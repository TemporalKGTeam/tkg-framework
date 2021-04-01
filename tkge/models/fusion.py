import torch
from torch import nn

from typing import Type, Callable, Dict
from collections import defaultdict
from functools import reduce
from abc import ABC, abstractmethod

from tkge.common.registrable import Registrable
from tkge.common.configurable import Configurable
from tkge.common.config import Config
from tkge.common.error import ConfigurationError
from tkge.train.regularization import Regularizer


class TemporalFusion(ABC, nn.Module, Registrable, Configurable):
    def __init__(self, config: Config):
        nn.Module.__init__(self)
        Registrable.__init__(self)
        Configurable.__init__(self, config=config)

    @classmethod
    def create_from_name(cls, config: Config):
        fusion_type = config.get("model.fusion.type")
        kwargs = config.get("model.fusion.args")

        kwargs = kwargs if not isinstance(kwargs, type(None)) else {}

        if fusion_type in TemporalFusion.list_available():
            # kwargs = config.get("model.args")  # TODO: get all args params
            return TemporalFusion.by_name(fusion_type)(config, **kwargs)
        else:
            raise ConfigurationError(
                f"{fusion_type} specified in configuration file is not supported"
                f"implement your model class with `TemporalFusion.register(name)"
            )

    @abstractmethod
    def forward(self, operand1, operand2):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def embedding_constraint():
        raise NotImplementedError


@TemporalFusion.register(name="concatenate_fusion")
class ConcatenateTemporalFusion(TemporalFusion):
    def __init__(self, config: Config):
        super(ConcatenateTemporalFusion, self).__init__(config=config)

    def forward(self, operand1, operand2):
        res = {'real': torch.cat([operand1['real'], operand2['real']], dim=1)}

        return res

    @staticmethod
    def embedding_constraint():
        in_constraints = {'operand1': ['real'],
                          'operand2': ['real']}

        out_constraints = {'result': ['real']}

        return in_constraints, out_constraints


@TemporalFusion.register(name="addition_fusion")
class AdditionTemporalFusion(TemporalFusion):
    def __init__(self, config: Config):
        super(AdditionTemporalFusion, self).__init__(config=config)

    def forward(self, operand1, operand2):
        res = {'real': operand1['real'] + operand2['real']}

        return res

    @staticmethod
    def embedding_constraint():
        in_constraints = {'operand1': ['real'],
                          'operand2': ['real']}

        out_constraints = {'result': ['real']}

        return in_constraints, out_constraints


@TemporalFusion.register(name="elementwise_product_fusion")
class ElementwiseTemporalFusion(TemporalFusion):
    def __init__(self, config: Config):
        super(ElementwiseTemporalFusion, self).__init__(config=config)

    def forward(self, operand1, operand2):
        res = {'real': operand1['real'] * operand2['real']}

        return res

    @staticmethod
    def embedding_constraint():
        in_constraints = {'operand1': ['real'],
                          'operand2': ['real']}

        out_constraints = {'result': ['real']}

        return in_constraints, out_constraints


@TemporalFusion.register(name="complex_elementwise_product_fusion")
class ComplexElementwiseTemporalFusion(TemporalFusion):
    def __init__(self, config: Config):
        super(ComplexElementwiseTemporalFusion, self).__init__(config=config)

    def forward(self, operand1, operand2):
        """
        operand1 and operand2 should be complex embeddings
        """
        p = operand1['real'] * operand2['real'], \
            operand1['imag'] * operand2['real'], \
            operand1['real'] * operand2['imag'], \
            operand1['imag'] * operand2['imag']

        res = {'real': p[0] - p[3],
               'imag': p[1] + p[2]}

        return res

    @staticmethod
    def embedding_constraint():
        in_constraints = {'operand1': ['real', 'imag'],
                          'operand2': ['real', 'imag']}

        out_constraints = {'result': ['real', 'imag']}

        return in_constraints, out_constraints


@TemporalFusion.register(name="reproject_fusion")
class ReprojectTemporalFusion(TemporalFusion):
    def __init__(self, config: Config):
        super(ReprojectTemporalFusion, self).__init__(config=config)

    def forward(self, operand1, operand2):
        """
        input should be [static embedding, temporal embedding]
        """
        inner = torch.sum(operand1['real'] * operand2['real'], dim=1, keepdim=True) / torch.sum(operand1['real'] ** 2,
                                                                                                 dim=-1, keepdim=True)
        res = {'real': operand1['real'] - inner * operand2['real']}

        return res

    @staticmethod
    def embedding_constraint():
        in_constraints = {'operand1': ['real'],
                          'operand2': ['real']}

        out_constraints = {'result': ['real']}

        return in_constraints, out_constraints


@TemporalFusion.register(name="hidden_representaion_fusion")
class HiddenRepresentationCombination(TemporalFusion):
    """Base combination operator for hidden representation"""

    def __init__(self, config: Config):
        super(HiddenRepresentationCombination, self).__init__(config=config)

    def forward(self, operand1, operand2):
        raise NotImplementedError


@TemporalFusion.register(name="diachronic_entity_fusion")
class DiachronicEntityFusion(HiddenRepresentationCombination):
    def __init__(self, config: Config):
        super(DiachronicEntityFusion, self).__init__(config=config)

        self.time_nl = torch.sin


    def forward(self, operand1, operand2):
        """
        operand1 are entity embedding
        operand2 are timestamp index

        return a batch_size * dim embedding
        """

        time_emb = operand1['amps_y'] * self.time_nl(
            operand1['freq_y'] * operand2['level0'] + operand1['phi_y'])
        time_emb += operand1['amps_m'] * self.time_nl(
            operand1['freq_m'] * operand2['level1'] + operand1['phi_m'])
        time_emb += operand1['amps_d'] * self.time_nl(
            operand1['freq_d'] * operand2['level2'] + operand1['phi_d'])

        emb = torch.cat((operand1['ent_embs'], time_emb), 1)
        res = {'real': emb}

        return res

    @staticmethod
    def embedding_constraint():
        in_constraints = {
            'operand1': ['ent_embs', 'amps_y', 'amps_m', 'amps_d', 'freq_y', 'freq_m', 'freq_d', 'phi_y', 'phi_m',
                         'phi_d'],
            'operand2': ['year', 'month', 'day']}

        out_constraints = {'result': ['real']}

        return in_constraints, out_constraints


@TemporalFusion.register(name="time_aware_fusion")
class TimeAwareFusion(HiddenRepresentationCombination):
    pass


@TemporalFusion.register(name="self_attention_fusion")
class SelfAttentioFusion(HiddenRepresentationCombination):
    pass

