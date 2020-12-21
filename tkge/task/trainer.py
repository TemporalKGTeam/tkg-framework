import torch

import time
import os

from tkge.data.dataset import Dataset
from tkge.train.sampling import NegativeSampler
from tkge.task.task import Task
from tkge.common.config import Config
from tkge.models.model import BaseModel
from tkge.models.loss import Loss


class TrainTask(Task):
    def __init__(self, config: Config):
        super().__init__()

        self.config = config

        self.dataset = self.config.get("data.name")
        # self.train_loader = None
        # self.valid_loader = None
        # self.test_loader = None
        self.sampler = None
        self.model = None
        self.loss = None
        self.optimizer = None

        self.batch_size = self.config.get("train.batch_size")

        self.device = self.config.get("task.device")

        self._prepare()

        self.train()

        # TODO optimizer should be added into modules

    def _prepare(self):
        self.config.log(f"Preparing datasets {self.dataset} in folder {self.config.get('data.folder')}...")
        self.dataset = Dataset.create(config=self.config)

        self.config.log(f"Loading training split data for loading")
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset.get_train(),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            pin_memory=self.config.get("train.pin_memory"),
        )

        # self.valid_loader = torch.utils.data.DataLoader(
        #     self.data.get_valid(),
        #     shuffle=False,
        #     batch_size=self.batch_size,
        #     num_workers=self.config.get()
        # )

        self.config.log(f"Initializing negative sampling")
        self.sampler = NegativeSampler.create(config=self.config, dataset=self.dataset)

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

    def train(self):
        self.config.log("BEGIN TRANING")
        self.model.train()

        for epoch in range(1, self.config.get("train.max_epochs") + 1):
            # TODO early stopping conditions
            # 1. metrics 变化小
            # 2. epoch
            # 3. valid koss
            total_loss = 0.0
            start = time.time()

            for pos_batch in self.train_loader:
                samples, labels = self.sampler.sample(pos_batch)

                scores = self.model(samples)

                loss = self.loss(scores, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.cpu().item()

            self.config.log(f"Loss in iteration {epoch} : {total_loss} ")

            if epoch % self.config.get("train.checkpoint.every") == 0:
                self.save_ckpt(epoch)

    def eval(self):
        # TODO early stopping

        raise NotImplementedError

    def save_ckpt(self, epoch):
        model = self.config.get("model.name")
        dataset = self.config.get("data.name")
        folder = self.config.get("train.checkpoint.folder")
        filename = f"epoch:{epoch}_model:{model}_dataset:{dataset}.ckpt"

        self.config.log(f"Save the model to {folder} as file {filename}")

        torch.save(self.model, os.path.join(folder, filename))

    def load_ckpt(self, ckpt_path):
        raise NotImplementedError
