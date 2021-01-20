import os
import comet_ml
from my_code.flow_pytorch.glow.gesture_flow import GestureFlow
from my_code.flow_pytorch.glow.utils import get_hparams
from my_code.misc.shared import BASE_DIR, CONFIG, DATA_DIR, RANDOM_SEED
from pytorch_lightning import Trainer, seed_everything

seed_everything(RANDOM_SEED)


if __name__ == "__main__":
    hparams, conf_name = get_hparams()
    assert os.path.exists(
        hparams.data_root
    ), "Failed to find root dir `{}` of dataset.".format(hparams.dataset_root)

    if hparams.comet_logger["api_key"] != "None":
        from pytorch_lightning.loggers import CometLogger

        from comet_ml.api import API, APIExperiment

        logger = CometLogger(
            api_key=hparams.comet_logger["api_key"],
            project_name=hparams.comet_logger["project_name"]
        )
    else:
        from pytorch_lightning import loggers as pl_loggers
        logger = pl_loggers.TensorBoardLogger('lightning_logs/')


    hparams.num_dataloader_workers = 0
    hparams.gpus = 0

    model = GestureFlow(hparams)

    trainer = Trainer.from_argparse_args(hparams, logger=logger) #, profiler="simple") # profiler="advanced"

    trainer.fit(model)
