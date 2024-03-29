import json
import multiprocessing
import os
import shutil
import socket
from argparse import ArgumentParser, Namespace
from pprint import pprint
from datetime import datetime

import numpy as np
import optuna
import pytorch_lightning as pl
import torch
import yaml
from jsmin import jsmin
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Trainer, seed_everything

import os
from my_code.flow_pytorch.glow.gesture_flow import GestureFlow
from my_code.flow_pytorch.glow.utils import get_hparams
from my_code.misc.shared import BASE_DIR, CONFIG, DATA_DIR, RANDOM_SEED
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from my_code.flow_pytorch.params_tuning_configs import hparams_to_check as hparam_configs

seed_everything(RANDOM_SEED)


class FailedTrial(Exception):
    pass


class MyEarlyStopping(PyTorchLightningPruningCallback):
    def __init__(self, trial, monitor="val_loss", patience=2):
        super().__init__(trial, monitor=monitor)
        self.best_loss = torch.tensor(np.Inf)
        self.wait = 0
        self.patience = patience

    def on_validation_epoch_end(self, trainer, pl_module):
        super().validation_epoch_end(trainer, pl_module)
        #jerk = trainer.callback_metrics.get("jerk/generated_mean")
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None and val_loss > 0:
            message = f"Trial was pruned because val loss was too high {val_loss}."
            raise optuna.exceptions.TrialPruned(message)
        #if jerk is not None and jerk > 10:
        #    message = f"Trial was pruned because jerk was too high {jerk}."
        #    raise optuna.exceptions.TrialPruned(message)
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return True


parser = ArgumentParser()
parser.add_argument("hparams_file")
parser.add_argument("-n", type=int)
parser = Trainer.add_argparse_args(parser)
default_params = parser.parse_args()

parser2 = ArgumentParser()
parser2.add_argument("hparams_file")
parser2.add_argument("-n", type=int)
override_params, unknown = parser2.parse_known_args()

conf_name = (
    os.path.basename(override_params.hparams_file)
    .replace(".yaml", "")
    .replace(".json", "")
)


def prepare_hparams(trial):

    if override_params.hparams_file.endswith(".json"):
        hparams_json = json.loads(jsmin(open(override_params.hparams_file).read()))
    elif override_params.hparams_file.endswith(".yaml"):
        hparams_json = yaml.load(open(override_params.hparams_file))
    hparams_json["dataset_root"] = str(DATA_DIR)

    params = vars(default_params)
    params.update(hparams_json)
    params.update(vars(override_params))
    hparams = Namespace(**params)

    hparams.gpus = [0] # [0] # [0,1]

    return hparam_configs.hparam_options(hparams, trial)


def run(hparams, return_dict, trial, batch_size, current_date):

    log_path = os.path.join("logs", conf_name, f"{current_date}")
    if os.path.exists(log_path):
        shutil.rmtree(log_path)

    hparams.batch_size = batch_size

    trainer_params = vars(hparams).copy()

    trainer_params["checkpoint_callback"] = pl.callbacks.ModelCheckpoint(
        save_top_k=3, monitor="val_loss", mode="min"
    )


    if hparams.comet_logger["api_key"]:
        from pytorch_lightning.loggers import CometLogger

        trainer_params["logger"] = CometLogger(
            api_key=hparams.comet_logger["api_key"],
            project_name=hparams.comet_logger["project_name"],
            experiment_name=conf_name + current_date,
        )

    trainer_params["early_stop_callback"] = MyEarlyStopping(trial, monitor="val_loss")

    trainer_params = Namespace(**trainer_params)

    trainer = Trainer.from_argparse_args(trainer_params, deterministic=False, enable_pl_optimizer=True)
    model = GestureFlow(hparams)

    try:
        trainer.fit(model)
    except RuntimeError as e:
        if str(e).startswith("CUDA out of memory"):
            return_dict["OOM"] = True
            #raise FailedTrial("CUDA out of memory")
        else:
            return_dict["error"] = e
            #raise e
    except (optuna.exceptions.TrialPruned, Exception) as e:
        return_dict["error"] = e

    for key, item in trainer.callback_metrics.items():
        return_dict[key] = float(item)


def get_training_name():
    dt = datetime.now()
    return f"{dt.day}-{dt.month}_{dt.hour}-{dt.minute}-{dt.second}.{str(dt.microsecond)[:2]}"


def objective(trial):
    current_date = get_training_name()

    manager = multiprocessing.Manager()

    hparams = prepare_hparams(trial)
    batch_size = hparams.batch_size

    trial.set_user_attr("version", current_date)
    trial.set_user_attr("host", socket.gethostname())
    trial.set_user_attr("GPU", os.environ.get("CUDA_VISIBLE_DEVICES"))

    pprint(vars(hparams))

    while batch_size > 0:
        print(f"trying with batch_size {batch_size}")

        return_dict = manager.dict()
        p = multiprocessing.Process(
            target=run, args=(hparams, return_dict, trial, batch_size, current_date),
        )
        p.start()
        p.join()

        if return_dict.get("OOM"):
            new_batch_size = batch_size // 2
            if new_batch_size < 2:
                raise FailedTrial("batch size smaller than 2!")
            else:
                batch_size = new_batch_size
        #elif return_dict.get("error"):
            #raise return_dict.get("error")
        else:
            break

    trial.set_user_attr("batch_size", batch_size)

    for metric, val in return_dict.items():
        if metric == "val_loss" and isinstance(val, float):
            trial.set_user_attr(metric, float(val))

    return float(return_dict["val_loss"])


if __name__ == "__main__":
    conf_vars = {}
    if CONFIG["optuna"]["rdbs_storage"]:
        conf_vars["storage"] = optuna.storages.RDBStorage(
            url=CONFIG["optuna"]["rdbs_storage"],
        )

    study = optuna.create_study(
        **conf_vars,
        study_name=conf_name,
        direction="minimize",
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=override_params.n, catch=(FailedTrial,))

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
