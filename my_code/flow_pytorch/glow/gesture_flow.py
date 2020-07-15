import random
from functools import reduce
import os
from os import path

import numpy as np
import optuna
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import LambdaLR, MultiplicativeLR, StepLR
from torch.utils.data import DataLoader
from my_code.flow_pytorch.glow.modules import GaussianDiag

from my_code.flow_pytorch.data.trinity_no_text import SpeechGestureDataset


from my_code.flow_pytorch.glow import (
    FeatureEncoder,
    Glow,
    calc_jerk,
    get_longest_history
)


class GestureFlow(LightningModule):
    def __init__(self, hparams, dataset_root=None, test=None):
        super().__init__()

        if dataset_root is not None:
            hparams.dataset_root = dataset_root

        if test is not None:
            hparams.Test = test

        if not hparams.Glow.get("rnn_type"):
            hparams.Glow["rnn_type"] = "gru"

        self.data_root = hparams.dataset_root

        self.hparams = hparams

        # obtain datasets
        self.load_datasets()

        # define key parameters
        self.hparams = hparams
        self.train_seq_len = hparams.Train["seq_len"]
        self.val_seq_len = hparams.Validation["seq_len"]
        self.best_jerk = torch.tensor(np.Inf)

        self.glow = Glow(hparams)

        # read important hparams
        self.past_context = self.hparams.Cond["Speech"]["prev_context"]
        self.future_context = self.hparams.Cond["Speech"]["future_context"]
        self.autoregr_hist_length = self.hparams.Cond["Autoregression"]["history_length"]

        # To reduce deminsionality of the speech encoding
        self.reduce_speech_enc = nn.Sequential(nn.Linear(int(self.hparams.Cond["Speech"]["dim"] * \
                                                             (self.past_context + self.future_context)),
                                                         self.hparams.Cond["Speech"]["enc_dim"]),
                                               nn.Tanh(), nn.Dropout(self.hparams.dropout))

        self.calculate_mean_pose()

    def calculate_mean_pose(self):
        self.mean_pose = np.mean(self.val_dataset.gesture, axis=(0, 1))
        np.save("./glow/utils/mean_pose.npy", self.mean_pose)

    def load_datasets(self):
        try:
            self.train_dataset = SpeechGestureDataset(self.hparams.data_root)
            scalings = self.train_dataset.get_scalers()
            self.val_dataset = SpeechGestureDataset(self.hparams.data_root, scalings, train=False)
        except FileNotFoundError as err:
            abs_data_dir = os.path.abspath(self.hparams.data_dir)
            if not os.path.isdir(abs_data_dir):
                print(f"ERROR: The given dataset directory {abs_data_dir} does not exist!")
                print("Please, set the correct path with the --data_dir option!")
            else:
                print("ERROR: Missing data in the dataset!")
            exit(-1)

    def create_conditioning(self, batch, time_st, train = True):

        # take audio with context
        curr_audio = batch["audio"][:, time_st - self.past_context:time_st + self.future_context]

        speech_concat = torch.flatten(curr_audio, start_dim=1)
        speech_cond_info = self.reduce_speech_enc(speech_concat)

        # Full teacher forcing
        if self.autoregr_hist_length > 0:
            if train:
                # Take several previous poses for conditioning
                prev_poses = batch["gesture"][:, time_st - self.autoregr_hist_length:time_st, :]
            else:
                # Use mean pose for conditioning
                # initialize all the previous poses with the mean pose
                init_poses = np.array([[self.mean_pose for length in range(self.autoregr_hist_length)]
                                      for it in range(batch["audio"].shape[0])])
                # we have to put these Tensors to the same device as the model because
                # numpy arrays are always on the CPU
                # store the 3 previous poses
                prev_poses = torch.from_numpy(init_poses).to(batch["audio"].device)

        else:
            prev_poses = 0

        if self.autoregr_hist_length > 0:
            if self.autoregr_hist_length == 3:
                pose_condition_info = torch.cat((prev_poses[:,-1], prev_poses[:,-2],
                                                 prev_poses[:,-3]), 1)
            elif self.autoregr_hist_length == 2:
                pose_condition_info = torch.cat((prev_poses[:,-1], prev_poses[:,-2]), 1)
            else:
                pose_condition_info = prev_poses[-1]

            # Todo: encode various conditioning

            curr_cond = torch.cat((speech_cond_info, pose_condition_info), 1)

        else:
            pose_condition_info = None
            curr_cond = speech_cond_info

        return curr_cond


    def inference(self, seq_len, batch):
        self.glow.init_rnn_hidden()

        model_output_shape = torch.zeros([batch["audio"].shape[0], self.hparams.Glow["distr_dim"]])

        produced_poses = None

        for time_st in range(self.past_context + self.autoregr_hist_length, seq_len - self.future_context):

            curr_cond = self.create_conditioning(batch, time_st, train = False)

            curr_output, _ = self.glow(
                condition=curr_cond,
                eps_std=self.hparams.Infer["eps"],
                reverse=True,
                output_shape=model_output_shape,
            )

            # add current frame to the total motion sequence
            if produced_poses is None:
                produced_poses = curr_output.unsqueeze(1)
            else:
                produced_poses = torch.cat((produced_poses, curr_output.unsqueeze(1)), 1)

        print("Produced: ", produced_poses.shape)

        return produced_poses

    def forward(self, batch):

        # Initialize and read main variables
        self.glow.init_rnn_hidden()

        loss = 0
        seq_len = batch["audio"].shape[1]

        z_seq = []
        losses = []

        for time_st in range(self.past_context + self.autoregr_hist_length, seq_len-self.future_context):

            curr_output = batch["gesture"][:, time_st, :]

            curr_cond = self.create_conditioning(batch, time_st)

            z_enc, objective = self.glow(x=curr_output, condition=curr_cond)
            tmp_loss = self.loss(objective, z_enc)
            losses.append(tmp_loss.cpu().detach())
            loss += torch.mean(tmp_loss)

            z_seq.append(z_enc.detach())

        return z_seq, (loss / len(z_seq)).unsqueeze(-1), losses

    def loss(self, objective, z):
        objective += GaussianDiag.logp_simplified(z)
        nll = (-objective) / float(np.log(2.0))
        return nll

    def alt_loss(self, objective, z):
        log_likelihood = objective + GaussianDiag.logp_simplified(z)
        nll = (-log_likelihood) / float(np.log(2.0))
        return nll

    def training_step(self, batch, batch_idx):
        _, loss, _ = self(batch)
        tb_log = {"Loss/train": loss}

        if self.hparams.optuna and self.global_step > 20 and loss > 0:
            message = f"Trial was pruned since loss > 0"
            raise optuna.exceptions.TrialPruned(message)

        return {"loss": loss, "log": tb_log}

    def validation_step(self, batch, batch_idx):
        z_seq, loss, _ = self(batch)
        if self.hparams.optuna and self.global_step > 20 and loss > 0:
            message = f"Trial was pruned since loss > 0"
            raise optuna.exceptions.TrialPruned(message)
        output = {"val_loss": loss}

        if batch_idx == 0:  #  and self.global_step > 0
            output["jerk"] = {}

            if self.hparams.Validation["check_invertion"]:
                # Test if the Flow works correctly
                output["det_check"] = self.test_invertability(z_seq, loss, batch)

            if self.hparams.Validation["scale_logging"]:
                self.log_scales()

        if self.hparams.Validation["inference"]:
            output["gesture_example"] = self.inference(self.hparams.Infer["seq_len"], batch)

        return output

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tb_logs = {"Loss/val": avg_loss}
        save_loss = avg_loss

        det_check = [x["det_check"] for x in outputs if x.get("det_check") is not None]

        if det_check:
            avg_det_check = torch.stack(det_check).mean()
            tb_logs["reconstruction/error_percentage"] = avg_det_check

        jerk = [x["jerk"] for x in outputs if x.get("jerk")]
        if jerk:
            gt_jerk_mean = [x["gt_mean"] for x in jerk]
            if gt_jerk_mean:
                tb_logs[f"jerk/gt_mean"] = torch.stack(gt_jerk_mean).mean()

            generated_jerk_mean = [x["generated_mean"] for x in jerk]
            if generated_jerk_mean:
                tb_logs[f"jerk/generated_mean"] = torch.stack(
                    generated_jerk_mean
                ).mean()
                percentage = tb_logs[f"jerk/generated_mean"] / tb_logs[f"jerk/gt_mean"]
                tb_logs[f"jerk/generated_mean_ratio"] = percentage

            if (
                tb_logs[f"jerk/generated_mean"] > 5
                and self.hparams.optuna
                and self.global_step > 20
            ):
                message = f"Trial was pruned since jerk > 5"
                raise optuna.exceptions.TrialPruned(message)
            if tb_logs[f"jerk/generated_mean"] < self.best_jerk:
                self.best_jerk = tb_logs[f"jerk/generated_mean"]
            else:
                save_loss + torch.tensor(np.Inf)

        if self.hparams.Validation["inference"]:

            # Save resulting gestures without teacher forcing
            sample_prediction = outputs[0]['gesture_example'][:3].cpu().detach().numpy()

            filename = f"val_result_ep{self.current_epoch + 1}_raw.npy"
            save_path = path.join(self.hparams.val_gest_dir, filename)
            np.save(save_path, sample_prediction)

        return {"save_loss": save_loss, "val_loss": avg_loss, "log": tb_logs}

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
    def optimizer_step(
        self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None
    ):
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
        optimizer.step()
        optimizer.zero_grad()

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True
        )
        return loader

    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False
        )
        return loader

    def test_invertability(self, z_seq, loss, data):

        reconstr_seq = []

        self.glow.init_rnn_hidden()
        backward_loss = 0

        for time_st, z_enc in enumerate(z_seq):
            condition = self.create_conditioning(data,time_st + self.past_context + self.autoregr_hist_length)

            reconstr, backward_objective = self.glow(
                z=z_enc, condition=condition, eps_std=1, reverse=True
            )

            backward_loss += torch.mean(
                self.loss(-backward_objective, z_enc)
            )  # , x.size(1)

            reconstr_seq.append(reconstr.detach())

        backward_loss = (backward_loss / len(z_seq)).unsqueeze(-1)

        error_percentage = (backward_loss - loss) * 100 / loss

        return torch.abs(error_percentage)