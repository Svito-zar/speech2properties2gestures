import random
from functools import reduce
import os

import numpy as np
import optuna
import torch
from pytorch_lightning import LightningModule
from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import LambdaLR, MultiplicativeLR, StepLR
from torch.utils.data import DataLoader
from my_code.flow_pytorch.glow.modules import GaussianDiag

from my_code.flow_pytorch.data.trinity_taras import SpeechGestureDataset


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


    def inference(self, seq_len, data=None):
        self.glow.init_rnn_hidden()

        output_shape = torch.zeros_like(data[:, 0, :])
        prev_poses = []

        for time_st in range(0, seq_len):
            condition =None

            output, _ = self.glow(
                condition=condition,
                eps_std=self.hparams.Infer["eps"],
                reverse=True,
                output_shape=output_shape,
            )

            prev_poses = torch.cat([prev_poses, output.unsqueeze(1)], dim=1)


        return prev_poses

    def forward(self, batch):
        self.glow.init_rnn_hidden()

        loss = 0
        seq_len = batch["audio"].shape[1]

        z_seq = []
        losses = []
        for time_st in range(0, seq_len):
            curr_output = batch["gesture"][:, time_st, :]
            curr_cond = batch["audio"][:, time_st, :]

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