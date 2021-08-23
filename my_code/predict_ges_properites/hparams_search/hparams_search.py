import json
import multiprocessing
import os
import shutil
import socket
from argparse import ArgumentParser, Namespace
from pprint import pprint
from datetime import datetime

import comet_ml
import numpy as np
import optuna
import pytorch_lightning as pl
import torch
import yaml
from jsmin import jsmin
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Trainer, seed_everything

import os
from my_code.predict_ges_properites.speech2prop import PropPredictor
from my_code.predict_ges_properites.cross_validation import get_hparams
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from my_code.predict_ges_properites.hparams_search import hparams_range_of_values as hparam_configs
from my_code.predict_ges_properites.GestPropDataset import GesturePropDataset

from sklearn.model_selection import KFold


class FailedTrial(Exception):
    pass

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

    params = vars(default_params)
    params.update(hparams_json)
    params.update(vars(override_params))
    hparams = Namespace(**params)

    hparams.gpus = 0 # [0] # [0,1]

    return hparam_configs.hparam_options(hparams, trial)


def run(hparams, return_dict, trial, batch_size, current_date):

    seed_everything(hparams.seed)

    log_path = os.path.join("logs", conf_name, f"{current_date}")
    if os.path.exists(log_path):
        shutil.rmtree(log_path)

    hparams.batch_size = batch_size

    trainer_params = vars(hparams).copy()

    trainer_params["checkpoint_callback"] = pl.callbacks.ModelCheckpoint(
        save_top_k=3, monitor="Loss/val_loss", mode="min"
    )


    if hparams.comet_logger["api_key"]:
        from pytorch_lightning.loggers import CometLogger

        trainer_params["logger"] = CometLogger(
            api_key=hparams.comet_logger["api_key"],
            project_name=hparams.comet_logger["project_name"],
            experiment_name="GestProp" + current_date
        )

    # Configuration K-fold cross validation
    k_folds = 10

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds)

    # Load dataset
    train_n_val_dataset = GesturePropDataset(hparams.data_feat, hparams.speech_modality, hparams.data_root, "train_n_val/no_zeros")

    # Start print
    print('--------------------------------')

    trainer_params = Namespace(**trainer_params)

    # Obtain a list of all the recordings present in the dataset
    recordings_ids = train_n_val_dataset.property_dataset[:, 0]
    recordings = np.unique(recordings_ids)

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(train_n_val_dataset)):

        if fold > 6:
            break

        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        # Make sure that sequences do not overlap
        indices_to_remove = []
        # Take 5% of each recording into the validation set
        for curr_record_id in recordings:
            curr_record_indices = np.where(recordings_ids == curr_record_id)[0]

            # find current train indices by an overlap
            train_curr_record_indices = list( set(curr_record_indices) & set(train_ids) )

            # find current test indices by an overlap
            test_curr_record_indices = list ( set(curr_record_indices) & set(test_ids) )

            # skip recordings which are not in train or val
            if len(train_curr_record_indices) == 0 or len(test_curr_record_indices) == 0:
                continue

            for tr_ind in range(len(train_curr_record_indices)):
                for test_ind in range(len(test_curr_record_indices)):
                    if abs(train_curr_record_indices[tr_ind] - test_curr_record_indices[test_ind]) < 20:
                        indices_to_remove.append(tr_ind)
                        # consider next tr_ind
                        break

        train_ids = np.delete(train_ids, indices_to_remove, axis=0)

        trainer = Trainer.from_argparse_args(trainer_params, deterministic=False, enable_pl_optimizer=True)
        model = PropPredictor(hparams, fold, train_ids, test_ids, upsample=hparams.CB["upsample"])

        try:
           trainer.fit(model)
        except RuntimeError as e:
           if str(e).startswith("CUDA out of memory"):
               return_dict["OOM"] = True
               raise FailedTrial("CUDA out of memory")
           else:
               return_dict["error"] = e
               raise e
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
        elif return_dict.get("error"):
            raise return_dict.get("error")
        else:
            break

    trial.set_user_attr("batch_size", batch_size)

    for metric, val in return_dict.items():
        if metric == "Loss/val_loss" and isinstance(val, float):
            trial.set_user_attr(metric, float(val))

    return float(return_dict["Loss/val_loss"])


if __name__ == "__main__":
    conf_vars = {}

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
