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
from tkge.common.config import Config
from tkge.models.model import BaseModel
from tkge.models.loss import Loss
from tkge.eval.metrics import Evaluation


class TrainTask(Task):
    @staticmethod
    def parse_arguments(parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Train a model"""
        subparser = parser.add_parser("train", description=description, help="train a model.")

        subparser.add_argument(
            "-c",
            "--config",
            type=str,
            help="specify configuration file path"
        )

        subparser.add_argument(
            "--resume",
            action="store_true",
            default=False,
            help="resume training from checkpoint in config file"
        )

        subparser.add_argument(
            "--overrides",
            action="store_true",
            default=False,
            help="override the hyper-parameter stored in checkpoint with the configuration file"
        )

        return subparser

    def __init__(self, config: Config):
        super().__init__(config)

        self.dataset: DatasetProcessor = self.config.get("dataset.name")
        self.train_loader: torch.utils.data.DataLoader = None
        self.valid_loader: torch.utils.data.DataLoader = None
        # self.test_loader = None
        self.sampler: NegativeSampler = None
        self.model: BaseModel = None
        self.loss: Loss = None
        self.optimizer: torch.optim.optimizer.Optimizer = None
        self.evaluation: Evaluation = None

        self.train_bs = self.config.get("train.batch_size")
        self.valid_bs = self.config.get("train.valid.batch_size")
        self.datatype = (['timestamp_id'] if self.config.get("dataset.temporal.index") else []) + (
            ['timestamp_float'] if self.config.get("dataset.temporal.float") else [])

        # TODO(gengyuan): passed to all modules
        self.device = self.config.get("task.device")

        self._prepare()

        # TODO optimizer should be added into modules

    def _prepare(self):
        self.config.log(f"Preparing datasets {self.dataset} in folder {self.config.get('dataset.folder')}...")
        self.dataset = DatasetProcessor.create(config=self.config)
        self.dataset.info()

        self.config.log(f"Loading training split data for loading")
        # TODO(gengyuan) load params
        self.train_loader = torch.utils.data.DataLoader(
            SplitDataset(self.dataset.get("train"), self.datatype),
            shuffle=True,
            batch_size=self.train_bs,
            num_workers=self.config.get("train.loader.num_workers"),
            pin_memory=self.config.get("train.loader.pin_memory"),
            drop_last=self.config.get("train.loader.drop_last"),
            timeout=self.config.get("train.loader.timeout")
        )

        self.valid_loader = torch.utils.data.DataLoader(
            SplitDataset(self.dataset.get("test"), self.datatype + ['timestamp_id']),
            shuffle=False,
            batch_size=self.valid_bs,
            num_workers=self.config.get("train.loader.num_workers"),
            pin_memory=self.config.get("train.loader.pin_memory"),
            drop_last=self.config.get("train.loader.drop_last"),
            timeout=self.config.get("train.loader.timeout")
        )

        self.config.log(f"Initializing negative sampling")
        self.sampler = NegativeSampler.create(config=self.config, dataset=self.dataset)
        self.onevsall_sampler = NonNegativeSampler(config=self.config, dataset=self.dataset, as_matrix=True)

        self.config.log(f"Creating model {self.config.get('model.name')}")
        self.model = BaseModel.create(config=self.config, dataset=self.dataset)
        self.model.to(self.device)

        self.config.log(f"Initializing loss function")
        self.loss = Loss.create(config=self.config)

        self.config.log(f"Initializing optimizer")
        # TODO  choose Adam or other types
        optim_dict = {
            'Adam': torch.optim.Adam,
            'Adagrad': torch.optim.Adagrad
        }
        self.optimizer = optim_dict[self.config.get("train.optimizer.type")](
            self.model.parameters(),
            lr=self.config.get("train.optimizer.lr"),
            weight_decay=self.config.get("train.optimizer.reg_lambda")
        )

        self.config.log((f"Initializeing regularizer"))
        self.regularizer = dict()
        self.inplace_regularizer = dict()

        if self.config.get("train.regularizer"):
            for name in self.config.get("train.regularizer"):
                self.regularizer[name] = Regularizer.create(self.config, name)

        if self.config.get("train.inplace_regularizer"):
            for name in self.config.get("train.inplace_regularizer"):
                self.inplace_regularizer[name] = InplaceRegularizer.create(self.config, name)

        self.config.log(f"Initializing evaluation")
        self.evaluation = Evaluation(config=self.config, dataset=self.dataset)

    def main(self):
        self.config.log("BEGIN TRANING")

        save_freq = self.config.get("train.checkpoint.every")
        eval_freq = self.config.get("train.valid.every")

        sample_target = self.config.get("negative_sampling.target")

        for epoch in range(1, self.config.get("train.max_epochs") + 1):
            self.model.train()

            # TODO early stopping conditions
            # 1. metrics 变化小
            # 2. epoch
            # 3. valid koss
            total_loss = 0.0
            train_size = self.dataset.train_size

            start = time.time()

            for pos_batch in self.train_loader:
                self.optimizer.zero_grad()

                samples, labels = self.sampler.sample(pos_batch, sample_target)

                samples = samples.to(self.device)
                labels = labels.to(self.device)

                scores, factors = self.model.fit(samples)

                # TODO (gengyuan) assertion: size of scores and labels should be matched
                assert scores.size() == labels.size(), f"Score's size {scores.shape} should match label's size {labels.shape}"
                loss = self.loss(scores, labels)

                assert not (factors and set(factors.keys()) - (set(self.regularizer) | set(
                    self.inplace_regularizer))), f"Regularizer name defined in model {set(factors.keys())} should correspond to that in config file"

                if factors:
                    for name, tensors in factors.items():
                        if name not in self.regularizer:
                            continue

                        if not isinstance(tensors, (tuple, list)):
                            tensors = [tensors]

                        reg_loss = self.regularizer[name](tensors)
                        loss += reg_loss

                loss.backward()
                self.optimizer.step()

                total_loss += loss.cpu().item()

                # TODO(gengyuan) inplace regularize
                if factors:
                    for name, tensors in factors.items():
                        if name not in self.inplace_regularizer:
                            continue

                        if not isinstance(tensors, (tuple, list)):
                            tensors = [tensors]

                        self.inplace_regularizer[name](tensors)

                # empty caches
                del samples, labels, scores, factors
                if self.device=="cuda":
                    torch.cuda.empty_cache()

            stop = time.time()
            avg_loss = total_loss / train_size

            self.config.log(f"Loss in iteration {epoch} : {avg_loss} comsuming {stop - start}s")

            if epoch % save_freq == 0:
                self.save_ckpt(epoch)

            if epoch % eval_freq == 0:
                with torch.no_grad():
                    self.model.eval()

                    counter = 0

                    metrics = dict()
                    metrics['head'] = defaultdict(float)
                    metrics['tail'] = defaultdict(float)

                    for batch in self.valid_loader:
                        bs = batch.size(0)
                        dim = batch.size(1)


                        batch = batch.to(self.device)

                        counter += bs

                        queries_head = batch.clone()[:, :-1]
                        queries_tail = batch.clone()[:, :-1]


                        # samples_head, _ = self.onevsall_sampler.sample(queries, "head")
                        # samples_tail, _ = self.onevsall_sampler.sample(queries, "tail")

                        # samples_head = samples_head.to(self.device)
                        # samples_tail = samples_tail.to(self.device)

                        queries_head[:, 0] = float('nan')
                        queries_tail[:, 2] = float('nan')

                        batch_scores_head = self.model.predict(queries_head)
                        assert list(batch_scores_head.shape) == [bs,
                                                           self.dataset.num_entities()], f"Scores {batch_scores_head.shape} should be in shape [{bs}, {self.dataset.num_entities()}]"

                        batch_scores_tail = self.model.predict(queries_tail)
                        assert list(batch_scores_tail.shape) == [bs,
                                                           self.dataset.num_entities()], f"Scores {batch_scores_head.shape} should be in shape [{bs}, {self.dataset.num_entities()}]"

                        # TODO (gengyuan): reimplement ATISE eval

                        # if self.config.get("task.reciprocal_relation"):
                        #     samples_head_reciprocal = samples_head.clone().view(-1, dim)
                        #     samples_tail_reciprocal = samples_tail.clone().view(-1, dim)
                        #
                        #     samples_head_reciprocal[:, 1] += 1
                        #     samples_head_reciprocal[:, [0, 2]] = samples_head_reciprocal.index_select(1, torch.Tensor(
                        #         [2, 0]).long().to(self.device))
                        #
                        #     samples_tail_reciprocal[:, 1] += 1
                        #     samples_tail_reciprocal[:, [0, 2]] = samples_tail_reciprocal.index_select(1, torch.Tensor(
                        #         [2, 0]).long().to(self.device))
                        #
                        #     samples_head_reciprocal = samples_head_reciprocal.view(bs, -1)
                        #     samples_tail_reciprocal = samples_tail_reciprocal.view(bs, -1)
                        #
                        #     batch_scores_head_reci, _ = self.model.predict(samples_head_reciprocal)
                        #     batch_scores_tail_reci, _ = self.model.predict(samples_tail_reciprocal)
                        #
                        #     batch_scores_head += batch_scores_head_reci
                        #     batch_scores_tail += batch_scores_tail_reci

                        batch_metrics = dict()

                        batch_metrics['head'] = self.evaluation.eval(batch, batch_scores_head, miss='s')
                        batch_metrics['tail'] = self.evaluation.eval(batch, batch_scores_tail, miss='o')

                        # TODO(gengyuan) refactor
                        for pos in ['head', 'tail']:
                            for key in batch_metrics[pos].keys():
                                metrics[pos][key] += batch_metrics[pos][key] * bs

                    for pos in ['head', 'tail']:
                        for key in metrics[pos].keys():
                            metrics[pos][key] /= counter

                    self.config.log(f"Metrics(head prediction) in iteration {epoch} : {metrics['head'].items()}")
                    self.config.log(f"Metrics(tail prediction) in iteration {epoch} : {metrics['tail'].items()}")

    def eval(self):
        # TODO early stopping

        raise NotImplementedError

    def save_ckpt(self, epoch):
        model = self.config.get("model.name")
        dataset = self.config.get("dataset.name")
        folder = self.config.get("train.checkpoint.folder")
        filename = f"epoch_{epoch}_model_{model}_dataset_{dataset}.ckpt"

        self.config.log(f"Save the model to {folder} as file {filename}")

        torch.save({'state_dict': self.model.state_dict()}, os.path.join(folder, filename))  # os.path.join(model, dataset, folder, filename))

    def load_ckpt(self, ckpt_path):
        raise NotImplementedError
