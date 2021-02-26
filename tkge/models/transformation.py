import torch
from torch import nn
from torch import Tensor

from typing import Type, Callable, Dict
from collections import defaultdict

from tkge.common.registrable import Registrable
from tkge.common.configurable import Configurable
from tkge.common.config import Config
from tkge.train.regularization import Regularizer


class Transformation(Registrable, nn.Module):
    def __init__(self, config: Config):
        Registrable.__init__(config=config)
        nn.Module.__init__()

    @classmethod
    def create_from_name(cls, name: str):
        raise NotImplementedError

    def forward(self, *input):
        raise NotImplementedError


@Transformation.register(name="translation_tf")
class TranslationTransformation(Transformation):
    pass

@Transformation.register(name="rotation_tf")
class RotationTransformation(Transformation):
    pass

@Transformation.register(name="rigid_tf")
class RigidTransformation(Transformation):
    pass

@Transformation.register(name="bilinear_tf")
class BilinearTransformation(Transformation):
    pass
