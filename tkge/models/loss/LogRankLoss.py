from tkge.models.loss import Loss

import torch
import torch.nn.functional as F


@Loss.register(name="log_rank_loss")
class LogRankLoss(Loss):
    def __init__(self, config, reduction="sum", **kwargs):
        super().__init__(config)

        self._device = config.get("task.device")

        self._loss = torch.nn.SoftMarginLoss(reduction=reduction, **kwargs)

        self.gamma = self.config.get("train.loss.gamma")
        self.temp = self.config.get("train.loss.temp")

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
