import torch
from torch import nn
from torch import Tensor

from typing import Type, Callable, Dict
from collections import defaultdict

from tkge.common.registry import Registrable
from tkge.common.configurable import Configurable
from tkge.common.config import Config
from tkge.train.regularization import Regularizer


class Transformation(nn.Module):
    def __init__(self):
        super(Transformation, self).__init__()

    def forward(self, *input):
        raise NotImplementedError


class TranslationTransformation(Transformation):
    pass


class RotationTransformation(Transformation):
    pass


class RigidTransformation(Transformation):
    pass


class BilinearTransformation(Transformation):
    pass
