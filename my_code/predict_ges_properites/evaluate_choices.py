import os
import comet_ml
import torch
import numpy as np

from my_code.predict_ges_properites.GestPropDataset import GesturePropDataset
from pytorch_lightning import seed_everything

from my_code.predict_ges_properites.cross_validation import K_fold_CV, get_hparams

torch.set_default_tensor_type('torch.FloatTensor')


if __name__ == "__main__":

    hparams, conf_name = get_hparams()

    seed_everything(hparams.seed)

    assert os.path.exists(
        hparams.data_root
    ), "Failed to find root dir `{}` of dataset.".format(hparams.data_root)

    # Load dataset
    train_n_val_dataset = GesturePropDataset(
        property_name = hparams.data_feat,
        root_dir = hparams.data_root,
        speech_modality = hparams.speech_modality,
        dataset_type = "train_n_val/"+hparams.data_type
    )

    if hparams.comet_logger["api_key"] != "None":
        from pytorch_lightning.loggers import CometLogger

        logger = CometLogger(
            api_key=hparams.comet_logger["api_key"],
            project_name=hparams.comet_logger["project_name"]
        )

    else:
        from pytorch_lightning import loggers as pl_loggers
        logger = pl_loggers.TensorBoardLogger('lightning_logs/', version=str(hparams.data_feat))

    hparams.gpus = 0 # [1]

    # Start print
    print('--------------------------------')

    # Obtain a list of all the recordings present in the dataset
    recordings_ids = train_n_val_dataset.record_ids
    recordings = np.unique(recordings_ids)

    # Usual K-fold Cross Validation model evaluation
    K_fold_CV(hparams, recordings_ids, logger, 10, extra_shift=4)
