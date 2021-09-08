import os
import yaml
import numpy as np
from my_code.predict_ges_properites.speech2prop import PropPredictor
from pytorch_lightning import Trainer
from argparse import ArgumentParser, Namespace


def K_fold_CV(hparams, recordings_ids, logger, fold_numb, extra_shift=0, catch_errors=False):
    """
    The classical K-fold Cross-Validation
    Args:
        hparams:         hyper-parameters of the model to evaluate
        recordings_ids:  list of recordings available in the dataset
        logger:          logger to log the experiment, usually CometML
        fold_numb:       number of the folds in the cross-validation
        extra_shift:     extra shift to make sure that we use different folds

    Returns:
        nothing, trains and evaluates a model instead

    """

    recordings = np.unique(recordings_ids)

    # K-fold mix x% of each Cross Validation model evaluation
    for fold in range(fold_numb):
        # Print
        print(f'Testing on hold out FOLD {fold}')
        print('--------------------------------')

        train_ids = []
        test_ids = []

        # Take 5% of each recording into the validation set
        for curr_record_id in recordings:
            curr_record_indices = np.where(recordings_ids == curr_record_id)[0]
            len_curr_ids = len(curr_record_indices)
            fraction = len_curr_ids // fold_numb

            # we don't want to take the same part of the recording all the time
            shift = int((curr_record_id + fold + extra_shift) % fold_numb)

            curr_test_ind = curr_record_indices[fraction * (shift): fraction * (shift + 1)]

            # make sure that sequences do not overlap by not using 20 closest sequences to the test data
            first_half_end = np.clip(fraction * (shift) - 20, 0, len(recordings_ids))
            curr_train_ids_1st_half = curr_record_indices[:first_half_end]
            second_half_start = np.clip(fraction * (shift + 1) + 20, 0, len(recordings_ids))
            curr_train_ids_2nd_half = curr_record_indices[second_half_start:]
            curr_train_ids = np.concatenate((curr_train_ids_1st_half, curr_train_ids_2nd_half))

            # Test the difference between any two indices in train and val is not smaller than 20
            for tr_ind in range(len(curr_train_ids)):
                for test_ind in range(len(curr_test_ind)):
                    assert abs(curr_train_ids[tr_ind] - curr_test_ind[test_ind]) >= 20

            if len(train_ids) == 0:
                train_ids = curr_train_ids
                test_ids = curr_test_ind
            else:
                train_ids = np.concatenate((train_ids, curr_train_ids))
                test_ids = np.concatenate((test_ids, curr_test_ind))

        # Make sure that train_ids[0] does in fact contain some indices!
        assert len(train_ids) > 0

        # Make sure train and test inds are not overlapping
        assert not any(np.isin(train_ids, test_ids))

        # Define the model
        model = PropPredictor(hparams, "a" + str(fold), train_ids, test_ids, upsample=False)

        # Define the trainer
        trainer = Trainer.from_argparse_args(hparams, deterministic=False, enable_pl_optimizer=True, logger=logger)

        # Train
        if catch_errors:
            try:
                trainer.fit(model)
            except RuntimeError as e:
                if str(e).startswith("CUDA out of memory"):
                    return_dict["OOM"] = True
                    raise FailedTrial("CUDA out of memory")
                elif isinstance(e, OSError) or str(e).find("memory"):
                    return_dict["memory"] = True
                    raise FailedTrial("CPU out of memory")
                else:
                    return_dict["error"] = e
            except (optuna.exceptions.TrialPruned, Exception) as e:
                return_dict["error"] = e

            for key, item in trainer.callback_metrics.items():
                return_dict[key] = float(item)
        else:
            trainer.fit(model)
def leave_one_out_CV(hparams, recordings_ids, logger):
    """
    Leave One Out Cross-validation
    Args:
        hparams:         hyper-parameters of the model to evaluate
        recordings_ids:  list of recordings available in the dataset
        logger:          logger to log the experiment, usually CometML

    Returns:

    """

    recordings = np.unique(recordings_ids)

    # K-fold LEAVE-ONE-OUT Cross Validation model evaluation
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

        # Make sure train and test inds are not overlapping
        assert not any(np.isin(train_ids[0], test_ids[0]))

        # Define the model
        model = PropPredictor(hparams, curr_record_id, train_ids[0], test_ids[0], upsample=hparams.CB["upsample"])

        # Define the trainer
        trainer = Trainer.from_argparse_args(hparams, deterministic=False, enable_pl_optimizer=True, logger=logger)
        # Train
        trainer.fit(model)


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