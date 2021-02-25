import torch
from torch import nn

from typing import Type, Callable, Dict
from collections import defaultdict
from functools import reduce

from tkge.common.registry import Registrable
from tkge.common.configurable import Configurable
from tkge.common.config import Config
from tkge.train.regularization import Regularizer


class TemporalFusion(Registrable, nn.Module):
    def __init__(self, config: Config):
        Registrable.__init__(config=config)
        nn.Module.__init__()

    @classmethod
    def create_from_name(cls, name: str):
        raise NotImplementedError


@TemporalFusion.register(name="concatenate_fusion")
class ConcatenateTemporalFusion(TemporalFusion):
    def __init__(self):
        super(ConcatenateTemporalFusion, self).__init__()

    def forward(self, operand1, operand2):
        return torch.cat([operand1, operand2], dim=1)


@TemporalFusion.register(name="addition_fusion")
class AdditionTemporalFusion(TemporalFusion):
    def __init__(self):
        super(AdditionTemporalFusion, self).__init__()

    def forward(self, operand1, operand2):
        return reduce(lambda t1, t2: t1 + t2, [operand1, operand2])


@TemporalFusion.register(name="elementwise_product_fusion")
class ElementwiseTemporalFusion(TemporalFusion):
    def __init__(self):
        super(ElementwiseTemporalFusion, self).__init__()

    def forward(self, operand1, operand2):
        return reduce(lambda t1, t2: t1 * t2, [operand1, operand2])


@TemporalFusion.register(name="reproject_fusion")
class ReprojectTemporalFusion(TemporalFusion):
    def __init__(self):
        super(ReprojectTemporalFusion, self).__init__()

    def forward(self, operand1, operand2):
        """
        input should be [static embedding, temporal embedding]
        """
        inner = torch.sum(operand1 * operand2, dim=-1, keepdim=True) / torch.sum(operand1 ** 2, dim=-1, keepdim=True)

        return operand1 - inner * operand2


@TemporalFusion.register(name="hidden_representaion_fusion")
class HiddenRepresentationCombination(TemporalFusion):
    """Base combination operator for hidden representation"""

    def __init__(self):
        super(HiddenRepresentationCombination, self).__init__()

    def forward(self, operand1, operand2):
        raise NotImplementedError


@TemporalFusion.register(name="diachronic_entity_fusion")
class DiachronicEntityFusion(HiddenRepresentationCombination):
    pass


@TemporalFusion.register(name="time_aware_fusion")
class TimeAwareFusion(HiddenRepresentationCombination):
    pass


@TemporalFusion.register(name="self_attention_fusion")
class SelfAttentioFusion(HiddenRepresentationCombination):
    pass
