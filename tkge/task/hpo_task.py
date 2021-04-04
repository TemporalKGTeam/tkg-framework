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
from tkge.eval.metrics import Evaluation

from typing import Dict, Tuple


class TrialMetric(ax.Metric):
    def fetch_trial_data(self, trial):
        records = []

        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "mean": (params["x1"] + 2 * params["x2"] - 7) ** 2 + (2 * params["x1"] + params["x2"] - 5) ** 2,
                "sem": 0.0,
                "trial_index": trial.index,
            })



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

    def _prepare_experiment(self):
        # initialize a client
        self.ax_client = ax.AxClient()

        # define the search space
        hp_group = {}
        for hp in self.config.get("search.hyperparam"):
            hp_group.update({hp.name: ax.Parameter()})

        search_space = ax.SearchSpace(
            parameters=hp_group.values()
        )

        self.ax_client.create_experiment(
            name="hyperparam_search",
            search_space=search_space
        )

        sobol = Models.SOBOL(search_space=search_space)
        generator_run = sobol.gen(self.config.get("search.num_trials"))



    def _evaluate(self, parameters) -> Dict[str, Tuple[float, float]]:
        """
        evaluate a trial given parameters and return the metrics
        """

        # overwrite the config

        # initialize a trainer


        # train

        # evaluate
        return {"mrr": (0, 0.0)}



    def main(self):
        # define the experiment
        experiment = None

        # generate trials/arms


        # Metrics to evaluate trials






