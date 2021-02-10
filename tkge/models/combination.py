import torch
from torch import nn

from typing import Type, Callable, Dict
from collections import defaultdict
from functools import reduce

from tkge.common.registry import Registrable
from tkge.common.configurable import Configurable
from tkge.common.config import Config
from tkge.train.regularization import Regularizer


class CombinationOperator(nn.Module):
    def __init__(self):
        super(CombinationOperator, self).__init__()


class ConcatenateCombination(CombinationOperator):
    def __init__(self):
        super(ConcatenateCombination, self).__init__()

    def forward(self, *input):
        return torch.cat(input, dim=1)


class AdditionCombination(CombinationOperator):
    def __init__(self):
        super(AdditionCombination, self).__init__()

    def forward(self, *input):
        return reduce(lambda t1, t2: t1 + t2, input)


class ElementwiseCombination(CombinationOperator):
    def __init__(self):
        super(ElementwiseCombination, self).__init__()

    def forward(self, *input):
        return reduce(lambda t1, t2: t1 * t2, input)


class ReprojectCombination(CombinationOperator):
    def __init__(self):
        super(ReprojectCombination, self).__init__()

    def forward(self, *input):
        """
        input should be [static embedding, temporal embedding]
        """
        inner = torch.sum(input[0] * input[1], dim=-1, keepdim=True) / torch.sum(input[0] ** 2, dim=-1, keepdim=True)

        return input[0] - inner * input[1]


class HiddenRepresentationCombination(CombinationOperator):
    """Base combination operator for hidden representation"""

    def __init__(self):
        super(HiddenRepresentationCombination, self).__init__()

    def forward(self, *input):
        raise NotImplementedError


class Combination(nn.Module):
    def __init__(self):
        super(Combination, self).__init__()

    def forward(self, *input):
