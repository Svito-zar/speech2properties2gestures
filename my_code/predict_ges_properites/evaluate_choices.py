import os
from argparse import ArgumentParser, Namespace
import yaml
import comet_ml
import torch
import numpy as np

from my_code.predict_ges_properites.speech2prop import PropPredictor
from my_code.predict_ges_properites.GestPropDataset import GesturePropDataset
from pytorch_lightning import Trainer, seed_everything

from sklearn.model_selection import KFold

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

    seed_everything(hparams.seed)

    assert os.path.exists(
        hparams.data_root
    ), "Failed to find root dir `{}` of dataset.".format(hparams.data_root)

    # Load dataset
    train_n_val_dataset = GesturePropDataset(
        property_name = hparams.data_feat,
        root_dir = hparams.data_root,
        speech_modality = hparams.speech_modality,
    )
    
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
    
    hparams.gpus = 0 # [1]

    # Start print
    print('--------------------------------')

    # Obtain a list of all the recordings present in the dataset
    recordings_ids = train_n_val_dataset.property_dataset[:, 0]
    recordings = np.unique(recordings_ids)

    # Configuration K-fold cross validation
    k_folds = 10

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=123456)

    # Usual K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(train_n_val_dataset)):

        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        # Make sure that sequences do not overlap
        indices_to_remove = []
        for curr_record_id in recordings:
            curr_record_indices = np.where(recordings_ids == curr_record_id)[0]

            # find current train indices by an overlap
            train_curr_record_indices = list(set(curr_record_indices) & set(train_ids))

            # find current test indices by an overlap
            test_curr_record_indices = list(set(curr_record_indices) & set(test_ids))

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

        # Make sure that train_ids[0] does in fact contain all indices!
        assert len(train_ids) > 0

        print("\n\ntrain IDs: ", train_ids[:20])

        # Define the model
        model = PropPredictor(hparams, "a" + str(fold), train_ids, test_ids, upsample=hparams.CB["upsample"])

        # Define the trainer
        trainer = Trainer.from_argparse_args(hparams, deterministic=False, enable_pl_optimizer=True, logger=logger)
        # Train
        trainer.fit(model)