import os
from argparse import ArgumentParser, Namespace
import yaml
import torch
import numpy as np

from my_code.predict_ges_properites.text2prop import PropPredictor
from my_code.predict_ges_properites.GestPropDataset import GesturePropDataset
from my_code.misc.shared import RANDOM_SEED
from pytorch_lightning import Trainer, seed_everything

from sklearn.model_selection import KFold

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

    # Configuration K-fold cross validation
    k_folds = 10

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds)

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

    # prepare to upsample under-represented classes
    freq = train_n_val_dataset.get_freq()
    n_features = hparams.CNN["output_dim"]

    max_freq = np.max(freq)
    multipliers = [int(max_freq // freq[feat]) for feat in range(n_features)]

    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(train_n_val_dataset)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        if test_ids[-1] > 67436:
            continue

        # upsample under-represented classes in the training set
        train_ids_upsampled = list(np.copy(train_ids))

        for frame_ind in range(train_ids.shape[0]):
            multipl_factor = 1
            for curr_feat in range(n_features):
                if train_n_val_dataset.y_dataset[frame_ind, curr_feat + 2] == 1:  # first two numbers are containing extra info
                    multipl_factor = max(multipl_factor, multipliers[curr_feat])
            if multipl_factor > 1:
                train_ids_upsampled += [train_ids[frame_ind]] * multipl_factor

        # Define the model
        model = PropPredictor(hparams, fold, train_ids_upsampled, test_ids)

        # Define the trainer
        trainer = Trainer.from_argparse_args(hparams, logger=logger) #, profiler="simple") # profiler="advanced"

        # Train
        trainer.fit(model)