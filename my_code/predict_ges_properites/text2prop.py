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
import matplotlib.pyplot as plt
import urllib.request

from my_code.predict_ges_properites.GestPropDataset import GesturePropDataset
from my_code.predict_ges_properites.classification_evaluation import evaluation
from my_code.predict_ges_properites.class_balanced_loss import ClassBalancedLoss


class PropPredictor(LightningModule):
    def __init__(self, hparams, fold, train_ids, val_ids, upsample=False, test=None):
        super().__init__()

        if test is not None:
            hparams.Test = test

        self.data_root = hparams.data_root
        self.upsample = upsample

        self.hparams = hparams

        # obtain datasets
        self.load_datasets()
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.fold = fold

        # define key parameters
        self.hparams = hparams
        self.kernel_size = hparams.CNN["kernel_size"]
        self.input_dim = hparams.CNN["input_dim"]
        self.output_dim = hparams.CNN["output_dim"]
        self.hidden_dim = hparams.CNN["hidden_dim"]
        self.n_layers = hparams.CNN["n_layers"]

        self.loss_funct = ClassBalancedLoss(self.class_freq, self.output_dim,
                                            self.hparams.Loss["beta"],  self.hparams.Loss["alpha"],
                                            self.hparams.Loss["gamma"])

        assert(self.kernel_size % 2 == 1)

        self.in_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(self.input_dim, self.hidden_dim, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        for i in range(self.n_layers):
            dilation = 2 ** i
            padding = int((self.kernel_size*dilation - dilation)/2)

            in_layer = torch.nn.Conv1d(self.hidden_dim, self.hidden_dim, self.kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end_linear = torch.nn.Sequential(nn.Linear(hparams.CNN["hidden_dim"],
                                                   hparams.CNN["output_dim"]), nn.Dropout(p=hparams.CNN["dropout"]))
        end_linear = end_linear.apply(self.weights_init)
        self.end = end_linear


    def weights_init(self, m):
        """Initialize the given linear layer using He initialization."""
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            n = m.in_features * m.out_features
            # m.weight.data should be zero
            m.weight.data.fill_(0.0)
            # m.bias.data
            m.bias.data.fill_(-np.log(self.hparams.CNN["output_dim"] - 1))


    def load_datasets(self):
        try:
            self.train_dataset = GesturePropDataset(self.data_root, "train_n_val", self.hparams.data_feat)
            self.val_dataset = GesturePropDataset(self.data_root, "train_n_val", self.hparams.data_feat)
            self.class_freq = self.train_dataset.get_freq()
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

        # take only the current time step
        seq_length = h_seq.shape[2]
        hid = h_seq[:, :, seq_length // 2 + 1]

        output = self.end(hid)

        return output


    def loss(self, prediction, label):

        loss_val = self.loss_funct(prediction, label,)

        return loss_val


    def accuracy(self, prediction, truth):

        # convert from likelihood to labels
        prediction = torch.sigmoid(prediction + 1e-6).round()

        # calculate metrics
        logs = evaluation(prediction.cpu(), truth.cpu(), self.output_dim)
        for metric in logs:
            self.log(metric + "/" + str(self.fold), logs[metric])


    def training_step(self, batch, batch_idx):

        predicted_prob = self(batch).float()
        true_lab = batch["property"][:, 2:].float() # ignore extra info, keep only the label

        loss_array = self.loss(predicted_prob, true_lab)

        loss_value = torch.mean(loss_array).unsqueeze(-1) / batch["property"].shape[1]

        self.log('Loss/train', loss_value)

        if self.hparams.optuna and self.global_step > 20 and loss_value > 1000000:
            message = f"Trial was pruned since training loss > 1000"
            raise optuna.exceptions.TrialPruned(message)

        return loss_value


    def validation_step(self, batch, batch_idx):

        prediction = self(batch).float()
        true_lab = batch["property"][:,2:].float()

        # plot sequnces
        if batch_idx == 2:
            x = batch["property"][:, 1].cpu()
            # convert from raw values to likelihood
            predicted_prob = torch.sigmoid(prediction + 1e-6)
            for feat in range(4):
                plt.ylim([0, 1])
                plt.plot(x, batch["property"][:, feat+2].cpu(), 'r--', x, predicted_prob[:, feat].cpu(), 'bs--')
                image_file_name = "fig/valid_res_"+str(self.current_epoch) + "_" + str(feat) + ".png"
                plt.savefig(fname=image_file_name)
                self.logger.experiment.log_image(image_file_name)
                plt.clf()

        loss_array = self.loss(prediction, true_lab)

        loss_value = torch.mean(loss_array).unsqueeze(-1) / batch["property"].shape[1]

        self.log('Loss/val_loss', loss_value)

        self.accuracy(prediction, true_lab)

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

        if self.upsample:
            # prepare to upsample under-represented classes
            n_features = self.output_dim

            max_freq = np.max(self.class_freq)
            multipliers = [int(max_freq // self.class_freq[feat]) for feat in range(n_features)]

            # upsample under-represented classes in the training set
            train_ids_upsampled = list(np.copy(self.train_ids))

            for frame_ind in range(self.train_ids.shape[0]):
                multipl_factor = 1
                for curr_feat in range(n_features):
                    if self.train_dataset.y_dataset[frame_ind, curr_feat + 2] == 1:  # first two numbers are containing extra info
                        multipl_factor = max(multipl_factor, multipliers[curr_feat]) # we take the highest multiplier of all features that are present in the frame
                if multipl_factor > 1:
                    train_ids_upsampled += [self.train_ids[frame_ind]] * multipl_factor

            self.train_ids = train_ids_upsampled

        train_subsampler = torch.utils.data.SubsetRandomSampler(self.train_ids)

        loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=8,
            sampler=train_subsampler
        )
        return loader

    def val_dataloader(self):

        val_sampler = torch.utils.data.SequentialSampler(self.val_ids)

        loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=8,
            sampler=val_sampler
        )
        return loader
