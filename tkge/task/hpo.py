import torch

import time
import os
from collections import defaultdict
import argparse

from ax import *

from tkge.task.task import Task
from tkge.data.dataset import DatasetProcessor, SplitDataset
from tkge.train.sampling import NegativeSampler, NonNegativeSampler
from tkge.train.regularization import Regularizer, InplaceRegularizer
from tkge.common.config import Config
from tkge.common.utils import LocalConfig
from tkge.models.model import BaseModel
from tkge.models.loss import Loss
from tkge.models.fusion import TemporalFusion
from tkge.models.transformation import Transformation
from tkge.eval.metrics import Evaluation

from typing import Dict


class HPOTask(Task):
    @staticmethod
    def parse_arguments(parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Hyperparameter optimization"""
        subparser = parser.add_parser("hpo", description=description, help="search hyperparameter.")

        subparser.add_argument(
            "-c",
            "--config",
            type=str,
            help="specify configuration file path"
        )

        return subparser

    def __init__(self, config: Config):
        super(HPOTask, self).__init__(config=config)

        self.dataset = self.config.get("dataset.name")
        self.train_loader: torch.utils.data.DataLoader = None
        self.valid_loader: torch.utils.data.DataLoader = None

        self.sampler: NegativeSampler = None

        self.loss: Loss = None

        self.optimizer: torch.optim.optimizer.Optimizer = None
        self.lr_scheduler = None
        self.evaluation: Evaluation = None

        self.device = self.config.get("task.device")

    def _prepare(self):
        pass

    def main(self):
        # define the experiment
        experiment = None

        # generate trials/arms




