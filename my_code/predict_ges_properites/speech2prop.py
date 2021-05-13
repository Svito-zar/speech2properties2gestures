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
from my_code.predict_ges_properites.class_balanced_loss import ClassBalancedLoss, FocalLoss, BasicLoss


class ModalityEncoder(nn.Module):
    def __init__(self, modality, hparams):
        super().__init__()

        if modality == "audio":
            params = hparams.audio_enc
        elif modality == "text":
            params = hparams.text_enc
        else:
            raise NotImplementedError("The modality '", modality,"' is not implemented")

        # read all the params
        self.kernel_size = params["kernel_size"]
        self.input_dim = params["input_dim"]
        self.output_dim = params["output_dim"]
        self.hidden_dim = params["hidden_dim"]
        self.n_layers = params["n_layers"]
        self.dropout = params["dropout"]

        assert (self.kernel_size % 2 == 1)

        # define the network
        self.in_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(self.input_dim, self.hidden_dim, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        for i in range(self.n_layers):
            dilation = 2 ** i
            padding = int((self.kernel_size * dilation - dilation) / 2)

            in_layer = nn.Sequential(
                nn.Conv1d(self.hidden_dim, self.hidden_dim, self.kernel_size,
                          dilation=dilation, padding=padding),
                nn.BatchNorm1d(self.hidden_dim),
                nn.LeakyReLU()
            )

            self.in_layers.append(in_layer)

        self.end = nn.Linear(self.hidden_dim, self.output_dim)


    def forward(self, x):
        """
        Args:
            x:   input speech sequences [batch_size, sequence length, speech_modality_dim]

        Returns:
            speech encoding vectors [batch_size, modality_enc_dim]
        """

        # reshape
        input_seq_tr = torch.transpose(x, dim0=2, dim1=1)

        # encode
        h_seq = self.start(input_seq_tr)

        # apply dilated convolutions
        for i in range(self.n_layers):
            h_seq = self.in_layers[i](h_seq)

        # take only the current time step
        seq_length = h_seq.shape[2]
        hid = h_seq[:, :, seq_length // 2 + 1]

        # adjust the output dim
        return self.end(hid)


class Decoder(nn.Module):
    def __init__(self, enc_dim, hparams):
        super().__init__()

        # read params
        params = hparams.decoder
        self.input_dim = enc_dim
        self.output_dim = params["output_dim"]
        self.hidden_dim = params["hidden_dim"]
        self.n_layers = params["n_layers"]
        self.dropout = params["dropout"]
        self.params = hparams

        # define the network
        start = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.start = torch.nn.utils.weight_norm(start, name='weight')

        self.in_layers = torch.nn.ModuleList()
        for i in range(self.n_layers):
            in_layer = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_dim, self.hidden_dim), nn.Dropout(p=self.dropout),
                torch.nn.LeakyReLU())

            self.in_layers.append(in_layer)

        if hparams.data_feat == "Phase":
            # use Softmax
            self.end = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_dim, self.output_dim), nn.Dropout(p=self.dropout),
                torch.nn.Softmax())
        else:
            # stick to Sigmoid (which is actually integrated in the loss function)
            end_linear = torch.nn.Linear(self.hidden_dim, self.output_dim,
                                         nn.Dropout(p=self.dropout))
            self.end = end_linear.apply(self.weights_init)


    def forward(self, x):
        """
        Args:
            x:  speech encoding vectors [batch_size, speech_enc_dim]

        Returns:
            output: probabilities for gesture properties [batch_size, prop_dim]
        """

        hid = self.start(x)

        for i in range(self.n_layers):
            hid = self.in_layers[i](hid)

        return self.end(hid)


    def weights_init(self, m):
        """Initialize the given linear layer using special initialization."""
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            # m.weight.data should be zero
            m.weight.data.fill_(0.0)
            # m.bias.data
            m.bias.data.fill_(-np.log(self.output_dim - 1))


class PropPredictor(LightningModule):
    def __init__(self, hparams, fold, train_ids, val_ids, upsample=False):
        super().__init__()

        self.data_root = hparams.data_root
        self.upsample = upsample
        self.should_stop = False

        self.hparams = hparams

        # obtain datasets
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.load_datasets()
        self.fold = fold

        # define key parameters
        self.hparams = hparams
        self.sp_mod = hparams.speech_modality

        # Define modality encoders depending on the speech modality used
        enc_dim = 0
        if self.sp_mod == "text" or self.sp_mod == "both":
            self.text_enc = ModalityEncoder("text", hparams)
            enc_dim += hparams.text_enc["output_dim"]

        if self.sp_mod == "audio" or self.sp_mod == "both":
            self.audio_enc = ModalityEncoder("audio", hparams)
            enc_dim += hparams.audio_enc["output_dim"]

        # define the encoding -> output network
        self.decoder = Decoder(enc_dim, hparams)

        # define the loss function
        integrated_sigmoid = hparams.data_feat != "Phase"
        print("Integrated sigmoid: ", integrated_sigmoid)
        if hparams.CB["loss_type"] == "CB":
            print("\nUsing Class-Balancing Loss\n")
            self.loss_funct = ClassBalancedLoss(self.class_freq, self.decoder.output_dim,
                                                beta=self.hparams.Loss["beta"], alpha=self.hparams.Loss["alpha"],
                                                gamma=self.hparams.Loss["gamma"], integrated_sigmoid=integrated_sigmoid)

        elif hparams.CB["loss_type"] == "focal":
            print("\nUsing Focal Loss\n")
            self.loss_funct = FocalLoss(alpha=self.hparams.Loss["alpha"],
                                        gamma=self.hparams.Loss["gamma"],
                                        integrated_sigmoid = integrated_sigmoid)
        elif hparams.CB["loss_type"] == "normal":
            print("\nUsing Normal Loss\n")
            self.loss_funct = BasicLoss(integrated_sigmoid)
        else:
            raise NotImplementedError("The loss '", hparams.CB["loss_type"],"' is not implemented")

    def load_datasets(self):
        try:
            self.train_dataset = GesturePropDataset(self.data_root, "train_n_val", self.hparams.data_feat, self.hparams.speech_modality)
            self.val_dataset = GesturePropDataset(self.data_root, "train_n_val", self.hparams.data_feat, self.hparams.speech_modality, self.val_ids)
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

        if self.sp_mod == "text" or self.sp_mod == "both":
            input_text_seq = batch["text"].float()
            text_enc = self.text_enc(input_text_seq)
            enc = text_enc

        if self.sp_mod == "audio" or self.sp_mod == "both":
            input_audio_seq = batch["audio"].float()
            audio_enc = self.audio_enc(input_audio_seq)
            enc = audio_enc

        if self.sp_mod == "both":
            enc = torch.cat((text_enc, audio_enc), 1)

        output = self.decoder(enc)

        return output


    def loss(self, prediction, label):

        loss_val = self.loss_funct(prediction.float(), label.float())

        return loss_val


    def accuracy(self, prediction, truth):

        # convert from likelihood to labels
        prediction = (torch.sigmoid(prediction + 1e-6)).round()

        # calculate metrics
        logs = evaluation(prediction.cpu(), truth.cpu())

	# terminate training if there is nothing to validation on
        if len(logs) == 0:
            self.should_stop = True

        for metric in logs:
            self.log(metric + "/" + str(self.fold), logs[metric])


    def training_step(self, batch, batch_idx):
        prediction = self(batch).float()
        true_lab = batch["property"][:, 2:].float() # ignore extra info, keep only the label

        loss_array = self.loss(prediction, true_lab)

        loss_value = torch.mean(loss_array).unsqueeze(-1) / batch["property"].shape[1]

        self.log('Loss/train', loss_value)

        if self.hparams.optuna and self.global_step > 20 and loss_value > 1000000:
            message = f"Trial was pruned since training loss > 1000"
            raise optuna.exceptions.TrialPruned(message)

        return loss_value


    def training_epoch_end(self, training_step_outputs):
        # do something with all training_step outputs
        if self.should_stop:
            raise KeyboardInterrupt


    def validation_step(self, batch, batch_idx):

        prediction = self(batch).float()
        true_lab = batch["property"][:,2:].int()

        # plot sequences
        if batch_idx == 0:

            x = batch["property"][10:60, 1].cpu()
            # convert from raw values to likelihood
            predicted_prob = torch.sigmoid(prediction + 1e-6)
            for feat in range(self.decoder.output_dim):
                plt.ylim([-0.01, 1.01])
                plt.plot(x, batch["property"][10:60, feat+2].cpu(), 'r--', x, predicted_prob[10:60, feat].cpu(), 'bs--')
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
            n_features = self.decoder.output_dim

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
            pin_memory=True,
            sampler=train_subsampler
        )
        return loader

    def val_dataloader(self):

        # Validate on the whole dataset at once
        val_batch_size = len(self.val_dataset.y_dataset)

        loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=False
        )

        return loader