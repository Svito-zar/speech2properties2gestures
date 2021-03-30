import os
from argparse import ArgumentParser, Namespace
import yaml
import torch
import numpy as np

from my_code.predict_ges_properites.text2prop import PropPredictor
from my_code.predict_ges_properites.GestPropDataset import GesturePropDataset
from my_code.misc.shared import RANDOM_SEED
from pytorch_lightning import Trainer, seed_everything

seed_everything(RANDOM_SEED)

torch.set_default_tensor_type('torch.FloatTensor')


def get_hparams():
    parser = ArgumentParser()
    parser.add_argument("hparams_file")
    parser = Trainer.add_argparse_args(parser)
    default_params = parser.parse_args()

    parser2 = ArgumentParser()
    parser2.add_argument("hparams_file")
    override_params, unknown = parser2.parse_known_args()

    conf_name = os.path.basename(override_params.hparams_file)
    if override_params.hparams_file.endswith(".yaml"):
        hparams_json = yaml.load(open(override_params.hparams_file), Loader=yaml.FullLoader)
    else:
        raise("Can only work with yaml Hparameters file")

    params = vars(default_params)
    params.update(hparams_json)
    params.update(vars(override_params))

    hparams = Namespace(**params)

    return hparams, conf_name


if __name__ == "__main__":

    hparams, conf_name = get_hparams()

    assert os.path.exists(
        hparams.data_root
    ), "Failed to find root dir `{}` of dataset.".format(hparams.data_root)

    # Load dataset
    train_n_val_dataset = GesturePropDataset(hparams.data_root, "train_n_val", hparams.data_feat)
    class_freq = train_n_val_dataset.get_freq()

    if hparams.comet_logger["api_key"] != "None":
        from pytorch_lightning.loggers import CometLogger

        logger = CometLogger(
            api_key=hparams.comet_logger["api_key"],
            project_name=hparams.comet_logger["project_name"]
        )

    else:
        from pytorch_lightning import loggers as pl_loggers
        logger = pl_loggers.TensorBoardLogger('lightning_logs/')

    hparams.num_dataloader_workers = 0
    hparams.gpus = 0

    # Start print
    print('--------------------------------')

    # Obtain a list of all the recordings present in the dataset
    recordings_ids = train_n_val_dataset.y_dataset[:, 0]
    recordings = np.unique(recordings_ids)

    # K-fold Cross Validation model evaluation
    for curr_record_id in recordings:
        # Print
        print(f'Testing on hold out recording {curr_record_id}')
        print('--------------------------------')

        # Select all the indices with this recording for the validation set
        test_ids = np.where(recordings_ids == curr_record_id)

        # Select the rest indices for the training set
        train_ids = np.where(recordings_ids != curr_record_id)

        # Make sure that train_ids[0] does in fact contain all indices!
        assert len(train_ids[0]) > 0
        assert len(train_ids[0]) + len(test_ids[0]) == len(recordings_ids)

        # Define the model
        model = PropPredictor(hparams, curr_record_id, train_ids[0], test_ids[0], upsample=True)

        # Define the trainer
        trainer = Trainer.from_argparse_args(hparams, logger=logger) #, profiler="simple") # profiler="advanced"

        # Train
        trainer.fit(model)