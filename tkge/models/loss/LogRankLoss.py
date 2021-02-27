from tkge.models.loss import Loss
from tkge.common.config import Config
from tkge.common.desc import *

import torch
import torch.nn.functional as F

from typing import Dict, Any


@Loss.register(name="log_rank_loss")
class LogRankLoss(Loss):
    _key_mapping: Dict[str, str] = {
        "device": "task.device",
        "gamma": "train.loss.gamma",
        "temp": "train.loss.temp"
    }

    device = DeviceParam(name='device', default_value='cuda')
    gamma = NumberParam(name='gamma', default_value=120)
    temp = NumberParam(name='temp', default_value=0.5)

    @property
    def key_mapping(self):
        return self._key_mapping

    def __init__(self, config, **kwargs):
        super().__init__(config)

        self._loss = torch.nn.SoftMarginLoss(**kwargs)

    def __call__(self, scores: torch.Tensor, labels: torch.Tensor, **kwargs):
        # TODO(gengyuan) kvsall not suppoorted
        batch_size = scores.size(0)

        # as matrix
        scores = self.gamma - scores

        scores_pos = scores[:, 0]
        scores_neg = scores[:, 1:]

        p = F.softmax(self.temp * scores_neg, dim=1)  # TODO(gengyuan) why dim=1
        loss_pos = torch.sum(F.softplus(-1 * scores_pos))
        loss_neg = torch.sum(p * F.softplus(scores_neg))
        loss = (loss_pos + loss_neg) / 2 / batch_size

        return loss

    # def _parse_config(self):
    #     load_default = self.config.get("global.load_default")
    #
    #     for k, v in self.fields_mapping.items():
    #         setattr(self, k, self.config.get(v))
