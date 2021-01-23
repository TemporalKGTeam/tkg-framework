import torch

import time
import os
from collections import defaultdict
import argparse

from tkge.task.task import Task
from tkge.data.dataset import DatasetProcessor, SplitDataset
from tkge.train.sampling import NegativeSampler, NonNegativeSampler
from tkge.train.regularization import Regularizer, InplaceRegularizer
from tkge.common.config import Config
from tkge.models.model import BaseModel
from tkge.models.loss import Loss
from tkge.eval.metrics import Evaluation


class TestTask(Task):
    @staticmethod
    def parse_arguments(parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Eval a model"""
        subparser = parser.add_parser("eval", description=description, help="evaluate a model.")

        subparser.add_argument(
            "-c",
            "--config",
            type=str,
            help="specify configuration file path"
        )

        return subparser

    def __init__(self, config: Config):
        self.dataset = self.config.get("dataset.name")
        self.test_loader = None
        self.sampler = None
        self.model = None
        self.evaluation = None

        self.test_bs = self.config.get("test.batch_size")
        self.datatype = (['timestamp_id'] if self.config.get("dataset.temporal.index") else []) + (
            ['timestamp_float'] if self.config.get("dataset.temporal.float") else [])

        # TODO(gengyuan): passed to all modules
        self.device = self.config.get("task.device")

        self._prepare()

        self.test()

    def _prepare(self):
        self.config.log(f"Preparing datasets {self.dataset} in folder {self.config.get('dataset.folder')}")
        self.dataset = DatasetProcessor.create(config=self.config)

        self.config.log(f"Loading testing split data for loading")
        # TODO(gengyuan) load params
        self.test_loader = torch.utils.data.DataLoader(
            SplitDataset(self.dataset.get("test"), self.datatype + ['timestamp_id']),
            shuffle=False,
            batch_size=self.test_bs,
            num_workers=self.config.get("test.loader.num_workers"),
            pin_memory=self.config.get("test.loader.pin_memory"),
            drop_last=self.config.get("test.loader.drop_last"),
            timeout=self.config.get("test.loader.timeout")
        )

        self.onevsall_sampler = NonNegativeSampler(config=self.config, dataset=self.dataset, as_matrix=True)

        self.config.log(f"Loading model {self.config.get('model.name')}")
        self.model = BaseModel.create(config=self.config, dataset=self.dataset, device=self.device)
        model_path = self.config.get("test.model_path")
        model_state_dict = torch.load(model_path)

        self.model.load_state_dict(model_state_dict['state_dict'])

        self.config.log(f"Initializing evaluation")
        self.evaluation = Evaluation(config=self.config, dataset=self.dataset)

    def test(self):
        self.config.log("BEGIN TESTING")

        with torch.no_grad():
            self.model.eval()

            l = 0

            metrics = dict()
            metrics['head'] = defaultdict(float)
            metrics['tail'] = defaultdict(float)

            for batch in self.test_loader:
                bs = batch.size(0)
                dim = batch.size(1)
                l += bs

                samples_head, _ = self.onevsall_sampler.sample(batch, "head")
                samples_tail, _ = self.onevsall_sampler.sample(batch, "tail")

                samples_head = samples_head.to(self.device)
                samples_tail = samples_tail.to(self.device)

                batch_scores_head, _ = self.model.predict(samples_head)
                batch_scores_tail, _ = self.model.predict(samples_tail)

                batch_metrics = dict()
                batch_metrics['head'] = self.evaluation.eval(batch, batch_scores_head, miss='s')
                batch_metrics['tail'] = self.evaluation.eval(batch, batch_scores_tail, miss='o')

                for pos in ['head', 'tail']:
                    for key in batch_metrics[pos].keys():
                        metrics[pos][key] += batch_metrics[pos][key] * bs

            for pos in ['head', 'tail']:
                for key in metrics[pos].keys():
                    metrics[pos][key] /= l

            self.config.log(f"Metrics(head prediction) : {metrics['head'].items()}")
            self.config.log(f"Metrics(tail prediction) : {metrics['tail'].items()}")
