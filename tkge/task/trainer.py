import torch

import time
import os

from tkge.task.task import Task
from tkge.data.dataset import DatasetProcessor, SplitDataset
from tkge.train.sampling import NegativeSampler, NonNegativeSampler
from tkge.common.config import Config
from tkge.models.model import BaseModel
from tkge.models.loss import Loss
from tkge.eval.metrics import Evaluation


class TrainTask(Task):
    def __init__(self, config: Config):
        super().__init__()

        self.config = config

        self.dataset = self.config.get("dataset.name")
        self.train_loader = None
        self.valid_loader = None
        # self.test_loader = None
        self.sampler = None
        self.model = None
        self.loss = None
        self.optimizer = None
        self.evaluation = None

        self.train_bs = self.config.get("train.batch_size")
        self.valid_bs = self.config.get("train.valid.batch_size")
        self.datatype = (['timestamp_id'] if self.config.get("dataset.temporal.index") else []) + (
            ['timestamp_float'] if self.config.get("dataset.temporal.float") else [])

        # TODO(gengyuan): passed to all modules
        self.device = self.config.get("task.device")

        self._prepare()

        self.train()

        # TODO optimizer should be added into modules

    def _prepare(self):
        self.config.log(f"Preparing datasets {self.dataset} in folder {self.config.get('dataset.folder')}...")
        self.dataset = DatasetProcessor.create(config=self.config)

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
            SplitDataset(self.dataset.get("valid"), self.datatype),
            shuffle=False,
            batch_size=self.valid_bs,
            num_workers=self.config.get("train.loader.num_workers"),
            pin_memory=self.config.get("train.loader.pin_memory"),
            drop_last=self.config.get("train.loader.drop_last"),
            timeout=self.config.get("train.loader.timeout")
        )

        self.config.log(f"Initializing negative sampling")
        self.sampler = NegativeSampler.create(config=self.config, dataset=self.dataset)
        self.onevsall_sampler = NonNegativeSampler(config=self.config, dataset=self.dataset)

        self.config.log(f"Creating model {self.config.get('model.name')}")
        self.model = BaseModel.create(config=self.config, dataset=self.dataset, device=self.device)

        self.config.log(f"Initializing loss function")
        self.loss = Loss.create(config=self.config)

        self.config.log(f"Initializing optimizer")
        # TODO  choose Adam or other types
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.get("train.optimizer.lr"),
            weight_decay=self.config.get("train.optimizer.reg_lambda")
        )

        self.config.log(f"Initializing evaluation")
        self.evaluation = Evaluation(config=self.config, dataset=self.dataset)

    def train(self):
        self.config.log("BEGIN TRANING")

        save_freq = self.config.get("train.checkpoint.every")
        eval_freq = self.config.get("train.valid.every")

        regularizer = self.config.get("train.regularizer")

        for epoch in range(1, self.config.get("train.max_epochs") + 1):
            self.model.train()

            # TODO early stopping conditions
            # 1. metrics 变化小
            # 2. epoch
            # 3. valid koss
            total_loss = 0.0
            start = time.time()

            for pos_batch in self.train_loader:
                samples, labels = self.sampler.sample(pos_batch)

                scores, factors = self.model(samples)
                regs = None
                print(regularizer)

                # TODO(gengyuan) add regularizer
                loss = self.loss(scores, labels)
                loss.backward()
                self.optimizer.step()

                # TODO(gengyuan) inplace regularize
                total_loss += loss.cpu().item()

            self.config.log(f"Loss in iteration {epoch} : {total_loss} ")

            if epoch % save_freq == 0:
                self.save_ckpt(epoch)

            if epoch % eval_freq == 0:
                self.model.eval()

                scores_head = []
                scores_tail = []
                queries = []
                queries = []

                for batch in self.valid_loader:
                    samples_head, labels_head = self.onevsall_sampler.sample(batch, "head")
                    samples_tail, labels_tail = self.onevsall_sampler.sample(batch, "tail")

                    batch_scores_head, _ = self.model(samples_head)
                    batch_scores_tail, _ = self.model(samples_tail)

                    queries.append(batch)
                    scores_head.append(batch_scores_head)
                    scores_tail.append(batch_scores_tail)

                queries = torch.cat(queries, dim=0)
                scores_head = torch.cat(scores_head, dim=0)
                scores_tail = torch.cat(scores_tail, dim=0)

                metrics = dict()
                metrics['head'] = self.evaluation.eval(queries, scores_head, miss='s')
                metrics['tail'] = self.evaluation.eval(queries, scores_tail, miss='o')

                self.config.log(f"Metrics(head prediction) in iteration {epoch} : {metrics['head']}")
                self.config.log(f"Metrics(tail prediction) in iteration {epoch} : {metrics['tail']}")


def eval(self):
    # TODO early stopping

    raise NotImplementedError


def save_ckpt(self, epoch):
    model = self.config.get("model.name")
    dataset = self.config.get("dataset.name")
    folder = self.config.get("train.checkpoint.folder")
    filename = f"epoch:{epoch}_model:{model}_dataset:{dataset}.ckpt"

    self.config.log(f"Save the model to {folder} as file {filename}")

    torch.save(self.model, os.path.join(folder, filename))


def load_ckpt(self, ckpt_path):
    raise NotImplementedError
