import os

import optuna
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import LambdaLR, MultiplicativeLR, StepLR
from torch.utils.data import DataLoader
import numpy as np

from my_code.predict_ges_properites.GestPropDataset import GesturePropDataset


def weights_init_he(m):
    """Initialize the given linear layer using He initialization."""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        n = m.in_features * m.out_features
        # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0, np.sqrt(2 / n))
        # m.bias.data should be 0
        m.bias.data.fill_(0)


class PropPredictor(LightningModule):
    def __init__(self, hparams, dataset_root=None, test=None):
        super().__init__()

        if test is not None:
            hparams.Test = test

        self.data_root = hparams.data_root

        self.hparams = hparams

        # obtain datasets
        self.load_datasets()

        # define key parameters
        self.hparams = hparams
        self.kernel_size = hparams.CNN["input_dim"]
        self.input_dim = hparams.CNN["input_dim"]
        self.output_dim = hparams.CNN["output_dim"]
        self.hidden_dim = hparams.CNN["hidden_dim"]
        self.n_layers = hparams.CNN["n_layers"]

        self.loss_funct = nn.BCELoss()

        assert(self.kernel_size % 2 == 1)

        self.in_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(self.input_dim, self.hidden_dim, self.kernel_size, padding=int((self.kernel_size - 1) / 2))
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end_linear = torch.nn.Sequential(nn.Linear(hparams.CNN["hidden_dim"] * hparams.CNN["seq_length"],
                                                    hparams.CNN["output_dim"]), nn.Dropout(p = hparams.CNN["dropout"]),
                                                                                           nn.Sigmoid())
        end_linear = end_linear.apply(weights_init_he)
        self.end = end_linear

        for i in range(self.n_layers):
            dilation = 2 ** i
            padding = int((self.kernel_size*dilation - dilation)/2)

            in_layer = torch.nn.Conv1d(self.hidden_dim, self.hidden_dim, self.kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)


    def load_datasets(self):
        try:
            self.train_dataset = GesturePropDataset(self.data_root, "train", self.hparams.data_feat)
            self.val_dataset = GesturePropDataset(self.data_root, "val", self.hparams.data_feat)
        except FileNotFoundError as err:
            abs_data_dir = os.path.abspath(self.data_root)
            if not os.path.isdir(abs_data_dir):
                print(f"ERROR: The given dataset directory {abs_data_dir} does not exist!")
                print("Please, set the correct path with the --data_dir option!")
            else:
                print("ERROR: Missing data in the dataset!")
            exit(-1)


    def forward(self, batch):

        text_inp = batch["text"].float()

        # reshape
        input_seq_tr = torch.transpose(text_inp, dim0=2, dim1=1)

        h_seq = self.start(input_seq_tr)

        for i in range(self.n_layers):
            h_seq = self.in_layers[i](h_seq)

        # reshape
        hidden_flat = torch.flatten(h_seq, start_dim=1, end_dim=2) # dims=[1,2]) # hidden_tr_1, dim0=2, dim1=1)

        # final linear
        output = self.end(hidden_flat)

        return output


    def loss(self, prediction, label):

        loss_val = self.loss_funct(prediction, label)

        return loss_val


    def accuracy(self, prediction, truth):

        # convert from likelihood to labels
        prediction = prediction.round()

        acc_sum = 0

        for label in range(prediction.shape[1]):
            label_acc = torch.sum(prediction[:, label] == truth[:, label]) / prediction.shape[0]
            self.log('Acc/label_'+ str(label), label_acc)
            acc_sum += label_acc

        mean_acc = acc_sum / prediction.shape[1]

        return mean_acc


    def training_step(self, batch, batch_idx):

        predicted_prob = self(batch).float()
        true_lab = batch["property"].float()

        loss_array = self.loss(predicted_prob, true_lab)

        loss_value = torch.mean(loss_array).unsqueeze(-1) / batch["property"].shape[1]

        self.log('Loss/train', loss_value)

        if self.hparams.optuna and self.global_step > 20 and loss_value > 1000000:
            message = f"Trial was pruned since training loss > 1000"
            raise optuna.exceptions.TrialPruned(message)

        return loss_value


    def validation_step(self, batch, batch_idx):

        predicted_prob = self(batch).float()
        true_lab = batch["property"].float()

        loss_array = self.loss(predicted_prob, true_lab)

        loss_value = torch.mean(loss_array).unsqueeze(-1) / batch["property"].shape[1]

        self.log('Loss/val_loss', loss_value)

        acc = self.accuracy(predicted_prob, true_lab)

        self.log('Acc/mean_acc', acc)

        if self.hparams.optuna and self.global_step > 20 and loss_value > 1000000:
            message = f"Trial was pruned since loss > 1000"
            raise optuna.exceptions.TrialPruned(message)
        output = {"Loss/val_loss": loss_value}

        return output


    def configure_optimizers(self):
        lr_params = self.hparams.Optim
        optim_args = lr_params["args"][lr_params["name"]]
        optimizers = {"adam": Adam, "sgd": SGD, "rmsprop": RMSprop}
        # Define optimizer
        optimizer = optimizers[lr_params["name"]](
            self.parameters(), lr=self.hparams.lr, **optim_args
        )

        # Define Learning Rate Scheduling
        def lambda1(val):
            return lambda epoch: epoch // val

        sched_params = self.hparams.Optim["Schedule"]
        sched_name = sched_params["name"]
        if not sched_name:
            return optimizer

        sched_args = sched_params["args"][sched_name]

        if sched_name == "step":
            scheduler = StepLR(optimizer, **sched_args)
        elif sched_name == "multiplicative":
            scheduler = MultiplicativeLR(
                optimizer, lr_lambda=[lambda1(sched_args["val"])]
            )
        elif sched_name == "lambda":
            scheduler = LambdaLR(optimizer, lr_lambda=[lambda1(sched_args["val"])])
        else:
            raise NotImplementedError("Unimplemented Scheduler!")

        return [optimizer], [scheduler]


    # learning rate warm-up
    def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_i,
                       optimizer_closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):

        lr = self.hparams.lr
        # warm up lr
        warm_up = self.hparams.Optim["Schedule"]["warm_up"]
        if self.trainer.global_step < warm_up:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / warm_up)
            lr *= lr_scale
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        for pg in optimizer.param_groups:
            self.logger.log_metrics({"learning_rate": pg["lr"]}, self.global_step)

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()


    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=8,
            shuffle=True
        )
        return loader

    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=8,
            shuffle=True
        )
        return loader