from tkge.models.loss import Loss
from tkge.common.config import Config
from tkge.common.desc import *

import torch
import torch.nn.functional as F

from typing import Dict, Any


@Loss.register(name="log_rank_loss")
class LogRankLoss(Loss):
    device = DeviceParam('device')
    gamma = FloatParam('gamma')
    temp = FloatParam('temp')


    def __init__(self, config, reduction="sum", **kwargs):
        super().__init__(config)

        self.device = config.get("task.device")

        self._loss = torch.nn.SoftMarginLoss(reduction=reduction, **kwargs)

        self.gamma = self.config.get("train.loss.args.gamma")
        self.temp = self.config.get("train.loss.args.temp")

    def __call__(self, scores: torch.Tensor, labels: torch.Tensor, **kwargs):
        # TODO(gengyuan) kvsall not suppoorted
        batch_size = scores.size(0)

        # as matrix
        scores = self.gamma - scores


        scores_pos = scores[:, 0]
        scores_neg = scores[:, 1:]

        p = F.softmax(self.temp * scores_neg, dim=1)    #TODO(gengyuan) why dim=1
        loss_pos = torch.sum(F.softplus(-1 * scores_pos))
        loss_neg = torch.sum(p * F.softplus(scores_neg))
        loss = (loss_pos + loss_neg) / 2 / batch_size

        return loss

    @classmethod
    def _parse_config(cls, config: Config, load_default: bool = False) -> Config:
        mapping: Dict[str, str] = {
            "device": "task.device",
            "gamma": "train.loss.args.gamma",
            "temp": "train.loss.args.temp"
        }

        parsed_config = {}

        for k, v in mapping.items():
            parsed_config[k] = config.get(mapping[k])


