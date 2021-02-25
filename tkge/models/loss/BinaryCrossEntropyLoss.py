from tkge.models.loss import Loss
from tkge.common.config import Config

import torch


@Loss.register(name="binary_cross_entropy_loss")
class BinaryCrossEntropyLoss(Loss):
    def __init__(self, config: Config):
        super().__init__(config)

        self._device = config.get("task.device_type")
        self._train_type = config.get("train.type")
        self._loss = torch.nn.BCEWithLogitsLoss()

    def __call__(self, scores, labels, **kwargs):
        """Computes the loss given the scores and corresponding labels.

        `scores` is a batch_size x triples matrix holding the scores predicted by some
        model.

        `labels` is either (i) a batch_size x triples Boolean matrix holding the
        corresponding labels or (ii) a vector of positions of the (then unique) 1-labels
        for each row of `scores`.

        """

        if "negative_sampling" in self._train_type:
            return self._loss(scores, labels)

        elif self._train_type == "KvsAll":
            # TODO determine how to form pairs for margin ranking in KvsAll training
            # scores and labels are tensors of size (batch_size, num_entities)
            # Each row has 1s and 0s of a single sp or po tuple from training
            # How to combine them for pairs?
            # Each 1 with all 0s? Can memory handle this?
            raise NotImplementedError(
                "Margin ranking with KvsAll training not yet supported."
            )
        else:
            raise ValueError("train.type for margin ranking.")
