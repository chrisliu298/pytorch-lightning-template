import argparse
import logging
import os
import warnings

import wandb
from easydict import EasyDict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.profiler import SimpleProfiler

from data import DataModule
from model import BaseModel


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", type=str, default="baseline-cifar10-resnet18")
    parser.add_argument("--verbose", action="store_true")
    config = EasyDict(vars(parser.parse_args()))

    seed_everything(config.seed)  # set seed for reproducibility
    if not config.verbose:
        os.environ["WANDB_SILENT"] = "True"
        warnings.filterwarnings("ignore")
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    # setup data module, model, and trainer
    datamodule = DataModule(config)
    model = BaseModel(config)
    trainer = Trainer(
        gpus=-1,
        callbacks=[],
        max_epochs=200,
        check_val_every_n_epoch=1,
        benchmark=True,
        logger=None,
        profiler=SimpleProfiler(),
    )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, verbose=config.verbose)
    wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
