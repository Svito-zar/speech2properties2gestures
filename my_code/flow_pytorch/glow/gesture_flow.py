import random
from functools import reduce
import os
from os import path

import numpy as np
import optuna
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import LambdaLR, MultiplicativeLR, StepLR
from torch.utils.data import DataLoader
from my_code.flow_pytorch.glow.modules import DiagGaussian, thops
from my_code.flow_pytorch.glow.models import Seq_Flow

from my_code.flow_pytorch.data.trinity_taras import SpeechGestureDataset, inv_standardize

from my_code.data_processing.visualization.motion_visualizer.generate_videos import visualize

import urllib.request
import h5py

def calc_jerk(sequence):
    # https://en.wikipedia.org/wiki/Finite_difference_coefficient
    # −1/2 	1 	0 	−1 	1/2
    # shift coefficients
    # 0     1   2    3   4
    x = sequence.cpu()
    jerk = -0.5 * x[:, :-4] + x[:, 1:-3] - x[:, 3:-1] + 0.5 * x[:, 4:]
    return jerk.abs().mean()



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

        self.seq_flow = Seq_Flow(hparams)

        # read important hparams
        self.past_context = self.hparams.Cond["Speech"]["prev_context"]
        self.future_context = self.hparams.Cond["Speech"]["future_context"]
        self.autoregr_hist_length = self.hparams.Cond["Autoregression"]["history_length"]

        self.mean_pose = np.zeros([self.hparams.Glow["distr_dim"] ], dtype=np.float)

    def load_datasets(self):
        try:
            self.train_dataset = SpeechGestureDataset(self.hparams.data_root)
            self.scalings = self.train_dataset.get_scalers()
            self.val_dataset = SpeechGestureDataset(self.hparams.data_root, train=False)
        except FileNotFoundError as err:
            abs_data_dir = os.path.abspath(self.hparams.data_dir)
            if not os.path.isdir(abs_data_dir):
                print(f"ERROR: The given dataset directory {abs_data_dir} does not exist!")
                print("Please, set the correct path with the --data_dir option!")
            else:
                print("ERROR: Missing data in the dataset!")
            exit(-1)


    def inference(self, batch):

        produced_poses, _, _ , mu, sigma, = self.seq_flow(batch, reverse=True)

        if self.hparams.Validation["scale_logging"]:
            self.log_scales(torch.mean(mu, dim=0), "test_mu", torch.mean(sigma,dim=0), "test_sigma")

        return produced_poses

    def forward(self, batch):

        z_seq, logdet, prior_nll, mu, sigma = self.seq_flow(batch)

        if self.hparams.Validation["scale_logging"]:
            self.log_scales(torch.mean(mu, dim=0), "mu", torch.mean(sigma,dim=0), "sigma")

        return z_seq, logdet, prior_nll


    def log_histogram(self, x, name):

        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_histogram(name, x, self.global_step)
        else:
            self.logger.experiment.log_histogram_3d(x, name, self.global_step)

    def log_scales(self, mu, mu_name, sigma, sigma_name):
        if not self.logger:
            return

        self.log_histogram(mu, mu_name)
        self.log_histogram(sigma, sigma_name)

    def derange_batch(self, batch_data):
        # Shuffle conditioning info
        permutation = torch.randperm(batch_data["audio"].size(1))

        mixed_up_batch = {}
        for modality in ["audio", "text", "gesture"]:
            mixed_up_batch[modality] = batch_data[modality][:, permutation]

        return mixed_up_batch


    def training_step(self, batch, batch_idx):

        _, logdet, prior_nll = self(batch)

        loss_array = prior_nll -logdet

        loss_value = torch.mean(loss_array).unsqueeze(-1) / batch["audio"].shape[1]

        deranged_batch = self.derange_batch(batch)
        _, deranged_log_det, deranged_pr_nll = self(deranged_batch)
        deranged_loss = torch.mean(deranged_pr_nll - deranged_log_det).unsqueeze(-1) / batch["audio"].shape[1]

        if random.randint(0, 5) == 1:
            self.log_histogram(prior_nll / batch["audio"].shape[1], "tr_logs/prior_nll")
            self.log_histogram(logdet / batch["audio"].shape[1], "tr_logs/logdet")

        self.log('Loss/train', loss_value)
        self.log('Loss/missmatched_nll', deranged_loss)

        if self.hparams.optuna and self.global_step > 20 and loss_value > 1000:
            message = f"Trial was pruned since training loss > 1000"
            raise optuna.exceptions.TrialPruned(message)

        return loss_value


    def validation_step(self, batch, batch_idx):
        z_seq, logdet, prior_nll = self(batch)
        loss_array = -logdet + prior_nll
        loss = torch.mean(loss_array).unsqueeze(-1) / batch["audio"].shape[1]

        self.log('val_loss', loss)

        if self.hparams.optuna and self.global_step > 20 and loss > 10000:
            message = f"Trial was pruned since loss > 1000"
            raise optuna.exceptions.TrialPruned(message)
        output = {"val_loss": loss}

        if batch_idx == 0:

            if self.hparams.Validation["check_invertion"]:
                # Test if the Flow works correctly
                output["det_check"], output["reconstr_check"] = self.test_invertability(z_seq, loss, batch)

        if self.hparams.Validation["inference"] and batch_idx == 0:

            output["gesture_example"] = self.inference(batch)

            gt_mean_jerk = calc_jerk(batch["gesture"])
            generated_mean_jerk = calc_jerk(output["gesture_example"])

            self.log("jerk/gt_mean", gt_mean_jerk)
            self.log("jerk/generated_mean", generated_mean_jerk)
            self.log("jerk/generated_mean_ratio", generated_mean_jerk / gt_mean_jerk)

            output["jerk"] = generated_mean_jerk


        return output


    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tb_logs = {"Loss/val": avg_loss}
        save_loss = avg_loss

        det_check = [x["det_check"] for x in outputs if x.get("det_check") is not None]

        self.log("val_loss", avg_loss)

        if det_check:
            avg_det_check = torch.stack(det_check).mean()
            self.log("reconstruction/nll_error_percentage", avg_det_check)

        reconstr_check = [x["reconstr_check"] for x in outputs if x.get("reconstr_check") is not None]

        if reconstr_check:
            avg_reconstr_check = torch.stack(reconstr_check).mean()
            self.log("reconstruction/reconstr_error_percentage", avg_reconstr_check)

        jerk = torch.stack([x["jerk"] for x in outputs if x.get("jerk")]).mean()
        if jerk:
            if (
                jerk > 1
                and self.global_step > 20
            ):
                message = f"Trial was pruned since jerk > 1"
                raise optuna.exceptions.TrialPruned(message)
            if jerk < self.best_jerk:
                self.best_jerk = jerk

        if self.hparams.Validation["inference"]:

            # Save resulting gestures without teacher forcing
            sample_prediction = outputs[0]['gesture_example'][:3].cpu().detach().numpy()

            sample_gesture = inv_standardize(sample_prediction, self.scalings[-1])

            save_dir = path.join(os.getcwd(), "videos/" + self.logger.version)

            # make dir for the results, if it does not exist yet
            if self.current_epoch == 0:
                for dir in [save_dir, save_dir + "/raw", save_dir + "/videos", save_dir + "/temp"]:
                    try:
                        os.mkdir(dir)
                    except OSError:
                        print("Creation of the directory %s failed" % dir)
                    else:
                        print("\nSuccessfully created the directory %s " % dir)

            self.save_prediction(sample_gesture, save_dir)

            mp4_filename = path.join(save_dir + "/videos", f"val_result_ep{self.current_epoch + 1}.mp4")

            external_ip = urllib.request.urlopen('https://v4.ident.me/').read().decode('utf8')
            experiment_id = self.logger.version
            file_name = f"val_result_ep{self.current_epoch + 1}.mp4"
            video_html = "http://"+str(external_ip)+":5103/"+str(experiment_id) + "/" + str(file_name)

            self.logger.experiment.log_html(
                f"{mp4_filename}<br><video src='{video_html}' width=640 controls></video> <br><br>"
            )


    def save_prediction(self, gestures, save_dir, raw=False, video=True):
        """
        Save the given gestures to the <generated_gestures_dir>/<phase> folder
        using the formats found in hparams.prediction_save_formats.

        The possible formats are: BVH file, MP4 video and raw numpy array.

        Args:
            gestures:  The output of the model
            phase:  Can be "training", "validation" or "test"
            filename:  The filename of the saved outputs (default: epoch_<current_epoch>.<extension>)
        """

        data_fps = 20

        npy_filename = path.join(save_dir, "raw/" + f"val_result_ep{self.current_epoch + 1}.npy")

        if video:

            mp4_filename = path.join(save_dir, "videos/" + f"val_result_ep{self.current_epoch + 1}.mp4")

            data_pipe = path.join(os.getcwd(), 'glow/utils/data_pipe.sav')

            temp_bvh = path.join(save_dir, 'temp/temp.bvh')

            visualize(
                gestures,
                bvh_file=temp_bvh,
                mp4_file=mp4_filename,
                npy_file=npy_filename,
                start_t=0,
                end_t=data_fps * self.hparams.Validation["seq_len"],
                data_pipe_dir=data_pipe)

            # Clean up the temporary files
            os.remove(temp_bvh)

        elif raw:
            raw_save_path = path.join(self.hparams.val_gest_dir, npy_filename)
            np.save(raw_save_path, gestures)


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
            shuffle=False
        )
        return loader

    def test_invertability(self, z_seq, loss, batch_data):

        reconstructed_poses, back_logdet, prior_nll , _, _ = self.seq_flow(batch_data, z_seq, reverse=True) # reverse should be true!

        backward_loss = back_logdet + prior_nll

        mean_backward_loss = torch.mean(backward_loss).unsqueeze(-1) / batch_data["audio"].shape[1]

        nll_error_percentage = (mean_backward_loss - loss) * 100 / loss


        X_origin = batch_data["gesture"][:3, self.past_context:self.past_context + 3, :3]
        X_reconstr = reconstructed_poses[:3, :3, :3]

        X_error = torch.mean(X_origin - X_reconstr)


        # DEBUG
        debug = False
        if debug:
            # print("\nX origin: ", batch_data["gesture"][:3,self.past_context :self.past_context+3,:3])
            # print("\nX reconstr: ", reconstructed_poses[:3, :3, :3])
            print("X error: ", X_error)

            print("\nLoss: ", loss)
            print("Backward Loss: ", mean_backward_loss)

            print("Error: ", nll_error_percentage)

        return torch.abs(nll_error_percentage), torch.abs(X_error)
