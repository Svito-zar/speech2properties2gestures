import os

import comet_ml
import torch
import numpy as np

from my_code.predict_ges_properites.GestPropDataset import GesturePropDataset
from pytorch_lightning import seed_everything

from my_code.predict_ges_properites.cross_validation import K_fold_CV, leave_one_out_CV, get_hparams

torch.set_default_tensor_type('torch.FloatTensor')


if __name__ == "__main__":

    hparams, conf_name = get_hparams()

    seed_everything(hparams.seed)

    assert os.path.exists(
        hparams.data_root
    ), "Failed to find root dir `{}` of dataset.".format(hparams.data_root)

    # Load dataset
    train_n_val_dataset = GesturePropDataset(hparams.data_feat, hparams.speech_modality, hparams.data_root, "train_n_val/"+hparams.data_type)
    
    class_freq = train_n_val_dataset.get_freq()

    if hparams.comet_logger["api_key"] != "None":
        from pytorch_lightning.loggers import CometLogger

        logger = CometLogger(
            api_key=hparams.comet_logger["api_key"],
            project_name=hparams.comet_logger["project_name"]
        )

    else:
        from pytorch_lightning import loggers as pl_loggers
        logger = pl_loggers.TensorBoardLogger('lightning_logs/', version=str(hparams.data_feat))
    
    hparams.num_dataloader_workers = 8
    hparams.gpus = [1]

    # Start print
    print('--------------------------------')

    # Obtain a list of all the recordings present in the dataset
    recordings_ids = train_n_val_dataset.record_ids

    # K-fold Cross-Validation
    K_fold_CV(hparams, recordings_ids, logger, 20)

    # Leave-One-Out Cross-Validation
    #leave_one_out_CV(hparams, recordings_ids, logger)
