import torch

import time
import os
import argparse

from typing import Dict, List
from collections import defaultdict

from tkge.task.task import Task
from tkge.data.dataset import DatasetProcessor, SplitDataset
from tkge.train.sampling import NegativeSampler, NonNegativeSampler
from tkge.train.regularization import Regularizer, InplaceRegularizer
from tkge.train.optim import get_optimizer, get_scheduler
from tkge.common.config import Config
from tkge.models.model import BaseModel
from tkge.models.pipeline_model import PipelineModel
from tkge.models.loss import Loss
from tkge.eval.metrics import Evaluation


class ResumeTask(Task):
    @staticmethod
    def parse_arguments(parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Resume training"""
        subparser = parser.add_parser("resume", description=description, help="resume previous experiments.")

        subparser.add_argument(
            "-e",
            "--ex",
            type=str,
            help="specify the experiment folder",
            dest='experiment'
        )

        subparser.add_argument(
            "--overrides",
            action="store_true",
            default=False,
            help="override the hyper-parameter stored in checkpoint with a new configuration file"
        )

        subparser.add_argument(
            "-c",
            "--config",
            type=str,
            help="new configuration file"
        )

        return subparser

    def __init__(self):
        pass
