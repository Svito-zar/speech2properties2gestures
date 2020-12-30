import os
from my_code.flow_pytorch.glow.gesture_flow import GestureFlow
from my_code.flow_pytorch.glow.utils import get_hparams
from my_code.misc.shared import BASE_DIR, CONFIG, DATA_DIR, RANDOM_SEED
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers

seed_everything(RANDOM_SEED)


if __name__ == "__main__":
    hparams, conf_name = get_hparams()
    assert os.path.exists(
        hparams.data_root
    ), "Failed to find root dir `{}` of dataset.".format(hparams.dataset_root)

    hparams.num_dataloader_workers = 0
    hparams.gpus = 0

    model = GestureFlow(hparams)

    tb_logger = pl_loggers.TensorBoardLogger('lightning_logs/')

    trainer = Trainer.from_argparse_args(hparams, logger=tb_logger) # callbacks=callbacks)

    trainer.fit(model)
