"""
This is the main model file for Speech2Prop prediction model
"""

import os

# import optuna
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import LambdaLR, MultiplicativeLR, StepLR
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import urllib.request

from my_code.predict_ges_properites.GestPropDataset import GesturePropDataset
from my_code.predict_ges_properites.classification_evaluation import evaluation
from my_code.predict_ges_properites.class_balanced_loss import ClassBalancedLoss, FocalLoss, BasicLoss

mpl.rcParams.update(mpl.rcParamsDefault)
# adjust parameters
rcParams = {'text.latex.preamble':r'\usepackage{amsmath}',
                'text.usetex': False,
                'savefig.pad_inches': 0.0,
                'figure.autolayout': False,
                'figure.constrained_layout.use': True,
                'figure.constrained_layout.h_pad':  0.05,
                'figure.constrained_layout.w_pad':  0.05,
                'figure.constrained_layout.hspace': 0.0,
                'figure.constrained_layout.wspace': 0.0,
                'font.size':        16,
                'axes.labelsize':   'small',
                'legend.fontsize':  'small',
                'xtick.labelsize':  'x-small',
                'ytick.labelsize':  'x-small',
                'mathtext.default': 'regular',
                'font.family' :     'sans-serif',
                'axes.labelpad': 1,
                #'xtick.direction': 'in',
                #'ytick.direction': 'in',
                'xtick.major.pad': 2,
                'ytick.major.pad': 2,
                'xtick.minor.pad': 2,
                'ytick.minor.pad': 2
               }
mpl.rcParams.update(rcParams)


# I guess all those modality encoders could be put in a separate file
class SimpleModalityEncoder(nn.Module):
    """
    The main modality encoder - dilated CNN without any skip or residual connections
    """
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


class ResidualModalityEncoder(nn.Module):
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
        self.res_skip_CNNs = torch.nn.ModuleList()
        self.res_skip_linear_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(self.input_dim, self.hidden_dim, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        for i in range(self.n_layers):
            dilation = 2 ** i
            padding = int((self.kernel_size * dilation - dilation) / 2)

            # normal hidden layer - is a dilated CNN
            in_layer = nn.Sequential(
                nn.Conv1d(self.hidden_dim, self.hidden_dim, self.kernel_size,
                          dilation=dilation, padding=padding),
                nn.LeakyReLU()
            )

            self.in_layers.append(in_layer)

            # last residual layer is not necessary
            if i < self.n_layers - 1:
                res_skip_channels = 2 * self.hidden_dim
            else:
                res_skip_channels = self.hidden_dim

            # Now ResLayer consist of CNN and MLP
            res_skip_cnn = nn.Sequential(
                torch.nn.Conv1d(self.hidden_dim, 2 * res_skip_channels, 1),
                nn.LeakyReLU(),
            )
            res_skip_linear = torch.nn.Linear(2 * res_skip_channels, res_skip_channels)

            res_skip_linear = res_skip_linear.apply(self.zero_init)
            self.res_skip_CNNs.append(res_skip_cnn)
            self.res_skip_linear_layers.append(res_skip_linear)

        self.end = nn.Linear(self.hidden_dim, self.output_dim)


    def forward(self, x):
        """
        Args:
            x:   input speech sequences [batch_size, sequence length, speech_modality_dim]

        Returns:
            speech encoding vectors [batch_size, modality_enc_dim]
        """

        # define output representation shape - [batch_size, speech_modality_dim]
        output = torch.zeros(x.shape[0], self.hidden_dim, device="cuda")

        # reshape
        input_seq_tr = torch.transpose(x, dim0=2, dim1=1)

        # encode into hiddden_dim channels
        h_seq = self.start(input_seq_tr)
        seq_length = h_seq.shape[2]

        # go through all the layers
        for i in range(self.n_layers):

            # apply dilated convolutions
            acts = self.in_layers[i](h_seq)

            # apply residual network
            res_skip_acts = self.apply_residual_net(acts, i)

            # half of the outputs from ResLayer go to hidden layer and half - directly to the output
            # both in a residual way (by summing)
            if i < self.n_layers - 1:
                # add first half of ResLayer output to the hidden layer
                h_seq = h_seq + res_skip_acts[:, :self.hidden_dim, :]
                # add the second half of ResLayer output to the output vector
                # take only the current time step
                output = output + res_skip_acts[:, self.hidden_dim:, seq_length // 2 + 1]
            else:
                # add the ResLayer output to the output vector
                # take only the current time step
                output = output + res_skip_acts[:, :, seq_length // 2 + 1]

        # adjust the output dim
        return self.end(output)


    def apply_residual_net(self, input_tensor, i):

        # apply residual CNN
        cnn_acts = self.res_skip_CNNs[i](input_tensor)

        # transpose from [B, D, T] to [B, T, D]
        transp_cnn_acts = torch.transpose(cnn_acts, 2, 1)

        # apply residual MLP
        transp_mlp_acts = self.res_skip_linear_layers[i](transp_cnn_acts)

        # transpose back from [B, T, D] to [B, D, T]
        mlp_acts = torch.transpose(transp_mlp_acts, 2, 1)

        return mlp_acts



    def zero_init(self, m):
        """Initialize the given linear layer using special initialization."""
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            # m.weight.data should be zero
            m.weight.data.fill_(0.0)
            # m.bias.data
            m.bias.data.fill_(0.0)


class MLP_ModalityEncoder(nn.Module):
    def __init__(self, modality, hparams):
        super().__init__()

        if modality == "audio":
            params = hparams.audio_enc
        elif modality == "text":
            params = hparams.text_enc
        else:
            raise NotImplementedError("The modality '", modality,"' is not implemented")

        # read all the params
        self.input_dim = params["input_dim"]
        self.output_dim = params["output_dim"]
        self.hidden_dim = params["hidden_dim"]
        self.n_layers = params["n_layers"]
        self.dropout = params["dropout"]

        # define the network

        start = torch.nn.Linear(self.input_dim * params["seq_length"], self.hidden_dim)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        self.in_layers = torch.nn.ModuleList()

        for i in range(self.n_layers):

            in_layer = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
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
        input_flattened = torch.flatten(x, start_dim=1)

        # encode
        hid = self.start(input_flattened)

        # apply dilated convolutions
        for i in range(self.n_layers):
            hid = self.in_layers[i](hid)

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
                torch.nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim), nn.Dropout(p=self.dropout),
                torch.nn.LeakyReLU())

            self.in_layers.append(in_layer)

        if hparams.data_feat == "Phase":
            # use Softmax
            self.end = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_dim, self.output_dim),
                nn.BatchNorm1d(self.output_dim), nn.Dropout(p=self.dropout),
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
        self.use_speaker_ID = hparams.use_speaker_ID

        if self.upsample:
             print("\nUpsampling will be used\n")
        else:
             print("\nNo Upsampling this time\n")

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
            self.text_enc = SimpleModalityEncoder("text", hparams)
            enc_dim += hparams.text_enc["output_dim"]

        if self.sp_mod == "audio" or self.sp_mod == "both":
            self.audio_enc = SimpleModalityEncoder("audio", hparams)
            enc_dim += hparams.audio_enc["output_dim"]

        if self.use_speaker_ID:
            enc_dim += 25

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
            self.train_dataset = GesturePropDataset(
                property_name = self.hparams.data_feat, 
                speech_modality = self.hparams.speech_modality,
                root_dir = self.hparams.data_root, 
                dataset_type = "train_n_val/"+self.hparams.data_type
            )

            self.val_dataset = GesturePropDataset(
                property_name = self.hparams.data_feat, 
                speech_modality = self.hparams.speech_modality,
                root_dir = self.hparams.data_root,
                dataset_type = "train_n_val/"+self.hparams.data_type
            )
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

        if self.use_speaker_ID:
            # convert IDs to OneHot vector
            speaker_ID = batch["property"][:, 0].long()
            speaker_OneHot = torch.zeros((speaker_ID.shape[0], 25))
            speaker_OneHot.to(self.device)
            speaker_OneHot[np.arange(speaker_ID.shape[0]), speaker_ID - 1] = 1

        if self.sp_mod == "text" or self.sp_mod == "both":
            input_text_seq = batch["text"].float()
            text_enc = self.text_enc(input_text_seq)
            enc = text_enc
            if self.use_speaker_ID:
               enc = torch.cat((text_enc, speaker_OneHot), 1)

        if self.sp_mod == "audio" or self.sp_mod == "both":
            input_audio_seq = batch["audio"].float()
            audio_enc = self.audio_enc(input_audio_seq)
            enc = audio_enc
            if self.use_speaker_ID:
               enc =  torch.cat((audio_enc, speaker_OneHot), 1)

        if self.sp_mod == "both":
            enc = torch.cat((text_enc, audio_enc), 1)
            if self.use_speaker_ID:
               enc =  torch.cat((text_enc, audio_enc, speaker_OneHot), 1)

        output = self.decoder(enc)

        return output


    def loss(self, prediction, label):

        loss_val = self.loss_funct(prediction.float(), label.float())

        return loss_val


    def accuracy(self, prediction, truth, train=False):

        # convert from likelihood to labels
        if self.hparams.data_feat != "Phase":
            prediction = (torch.sigmoid(prediction + 1e-6)).round()
            # calculate metrics
            logs = evaluation(prediction.cpu(), truth.cpu())
        else:
            for frame in range(prediction.shape[0]):
                # find most likely class
                max_classes = torch.argmax(prediction[frame])
                # collect binary predictions
                prediction[frame,:] = 0
                prediction[frame,max_classes] = 1
            # calculate metrics
            logs = evaluation(prediction.cpu(), truth.cpu(), macroF1=False)

        """
        # FOR DEBUG - take random samples instead:
        print("EVALUATING RANDOM")
        # treat Phase in a special way
        if self.hparams.data_feat == "Phase":
            rand_sample = np.random.multinomial(1, [0.0055, 0.12243, 0.40856, 0.147994, 0.308, 0.007443], size=prediction.shape[0])

            prediction = torch.zeros(prediction.shape)
            for frame in range(prediction.shape[0]):
                # find most likely class
                curr_class = np.argmax(rand_sample[frame])
                if curr_class == 5:
                    continue
                # set binary predictions
                prediction[frame, curr_class] = 1

            # calculate metrics
            logs = evaluation(prediction, truth, macroF1=False)
        else:
            #for G_Phrase
            rand_0 = np.random.binomial(1, 0.2905482155, size=prediction.shape[0])
            rand_1 = np.random.binomial(1, 0.1447061637, size=prediction.shape[0])
            rand_2 = np.random.binomial(1, 0.7202726131, size=prediction.shape[0])
            rand_3 = np.random.binomial(1, 0.1278471932, size=prediction.shape[0])

            # for G_semant
            rand_0 = np.random.binomial(1, 0.04728881449, size=prediction.shape[0])
            rand_1 = np.random.binomial(1, 0.1310157231, size=prediction.shape[0])
            rand_2 = np.random.binomial(1, 0.137053865, size=prediction.shape[0])
            rand_3 = np.random.binomial(1, 0.01942966461, size=prediction.shape[0])

            prediction = np.stack((rand_0, rand_1, rand_2, rand_3), axis=1)
            prediction = torch.from_numpy(prediction)

            # calculate metrics
            logs = evaluation(prediction.cpu(), truth.cpu(), macroF1=True)
        """

        # log statistics
        for feat in range(prediction.shape[1]):
            column = prediction[:, feat]
            self.log("freq/fold_" + str(self.fold)+ "_feat_" + str(feat), torch.sum(column) / prediction.shape[0])

        # terminate training if there is nothing to validation on
        if len(logs) == 0:
            self.should_stop = True

        for metric in logs:
            if train:
                self.log("train_" + metric + "/" + str(self.fold), logs[metric])
            else:
                self.log(metric + "/" + str(self.fold), logs[metric])


    def training_step(self, batch, batch_idx):
        prediction = self(batch).float()
        true_lab = batch["property"][:, 2:].int() # ignore extra info, keep only the label

        loss_array = self.loss(prediction, true_lab)

        loss_value = torch.mean(loss_array).unsqueeze(-1) / batch["property"].shape[1]

        self.log('Loss/train', loss_value)

        if self.hparams.optuna and self.global_step > 20 and loss_value > 1000000:
            message = f"Trial was pruned since training loss > 1000"
            raise optuna.exceptions.TrialPruned(message)

        return {"loss": loss_value, "prediction":prediction, "true_lab": true_lab}


    def training_epoch_end(self, training_step_outputs):
        # do something with all training_step outputs

        if self.should_stop:
            print("\nSTOPPPING")
            raise KeyboardInterrupt

        # calculate training accuracy
        #all_predictions = torch.cat([x["prediction"] for x in training_step_outputs])
        #all_true_labels = torch.cat([x["true_lab"] for x in training_step_outputs])
        #self.accuracy(all_predictions.detach(), all_true_labels, train=True)


    def validation_step(self, batch, batch_idx):

        prediction = self(batch).float()
        true_lab = batch["property"][:,2:].int()

        # plot sequences
        if batch_idx == -1:

            x = batch["property"][10:60, 1].cpu()
            # convert from raw values to likelihood
            predicted_prob = torch.sigmoid(prediction + 1e-6)
            for feat in range(self.decoder.output_dim):
                plt.figure(figsize=(5.196, 3.63), dpi=300)
                plt.ylim([-0.01, 1.01])
                plt.plot(x, batch["property"][10:60, feat+2].cpu(), 'r--', x, predicted_prob[10:60, feat].cpu(), 'bs--', markersize=1)
                image_file_name = "fig/valid_res_"+str(self.current_epoch) + "_" + str(feat) + "_1.jpg"
                plt.savefig(fname=image_file_name, dpi=600)
                self.logger.experiment.log_image(image_file_name)
                plt.clf()


        loss_array = self.loss(prediction, true_lab)

        loss_value = torch.mean(loss_array).unsqueeze(-1) / batch["property"].shape[1]

        self.log('Loss/val_loss', loss_value)

        if self.hparams.optuna and self.global_step > 20 and loss_value > 1000000:
            message = f"Trial was pruned since loss > 1000"
            raise optuna.exceptions.TrialPruned(message)
        output = {"Loss/val_loss": loss_value, "prediction":prediction, "true_lab": true_lab}

        return output

    def validation_epoch_end(self, val_step_outputs):
        all_predictions = torch.cat([x["prediction"] for x in val_step_outputs])
        all_true_labels = torch.cat([x["true_lab"] for x in val_step_outputs])
        self.accuracy(all_predictions.detach(), all_true_labels)


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
                    if self.train_dataset.property_dataset[frame_ind, curr_feat + 2] == 1:  # first two numbers are containing extra info
                        multipl_factor = max(multipl_factor, multipliers[curr_feat]) # we take the highest multiplier of all features that are present in the frame
                if multipl_factor > 1:
                    train_ids_upsampled += [self.train_ids[frame_ind]] * multipl_factor

            self.train_ids = train_ids_upsampled

        train_subsampler = torch.utils.data.SubsetRandomSampler(self.train_ids)

        loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            pin_memory=False,
            drop_last=True,
            sampler=train_subsampler
        )
        return loader

    def val_dataloader(self):

        val_subsampler = torch.utils.data.SubsetRandomSampler(self.val_ids)

        loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            pin_memory=False,
            shuffle=False,
            sampler=val_subsampler
        )

        return loader
