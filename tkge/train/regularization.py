from typing import Tuple

import torch
from torch import nn

from tkge.common.config import Config
from tkge.common.registry import Registrable


class Regularizer(nn.Module, Registrable):
    def __init__(self, config: Config):
        Registrable.__init__(self, config)
        nn.Module.__init__(self)

        self.device = self.get_option("device")

    def forward(self, factors: Tuple[torch.Tensor], **kwargs):
        raise NotImplementedError

    @staticmethod
    def create(config: Config, name: str):
        reg_type = config.get(f"regularize.{name}.type")

        if reg_type in Regularizer.list_available():
            return Regularizer.by_name(reg_type)(config, reg_type)

        else:
            raise ValueError(
                f"{reg_type} specified in configuration file is not supported"
                f"implement your regularizer class with `Regularizer.register(name)"
            )


class InplaceRegularizer(Regularizer):
    @staticmethod
    def create(config: Config, name: str):
        reg_type = config.get(f"regularize.{name}.type")

        if reg_type in InplaceRegularizer.list_available():
            return InplaceRegularizer.by_name(reg_type)(config, reg_type)

        else:
            raise ValueError(
                f"{reg_type} specified in configuration file is not supported"
                f"implement your inplace-regularizer class with `InplaceRegularizer.register(name)"
            )


# @Regularizer.register(name="lp_regularize")
# class LpRegularizer(Regularizer):
#     def __init__(self, config, name):
#         super().__init__(config, name)
#
#         self.p = self.get_option("regularize.args")
#
#     def forward(self, x, **kwargs):
#         dim = torch.as_tensor(x.shape[-1], dtype=torch.float, device=x.device)
#         if kwargs["p"] == 1:
#             # expected value of |x|_1 = d*E[x_i] for x_i i.i.d.
#             return x / dim
#         if kwargs["p"] == 2:
#             # expected value of |x|_2 when x_i are normally distributed
#             # cf. https://arxiv.org/pdf/1012.0621.pdf chapter 3.1
#             return x / dim.sqrt()
#         raise NotImplementedError(f'Lp regularization not implemented for p={self.p}')


# @Regularizer.register(name="power_h_regularize")
# class PowerHRegularizer(Regularizer):
#     def __init__(self, config, name):
#         super().__init__(config, name)
#
#     def forward(self, x, **kwargs):
#         dim = torch.as_tensor(x.shape[-1], dtype=torch.float, device=x.device)
#
#         value = x.abs().pow(kwargs["p"]).sum(dim=dim).mean()
#         if not kwargs["normalize"]:
#             return value
#         #
#         # dim = torch.as_tensor(x.shape[-1], dtype=torch.float, device=x.device)
#         return value / dim


# reference: https://github.com/facebookresearch/tkbc/blob/master/tkbc/regularizers.py
@Regularizer.register(name="n3_regularize")
class N3Reg(Regularizer):
    def __init__(self, config: Config, name: str):
        super().__init__(config)

        # TODO(gengyuan) add attribute automatically
        self.weight = self.config.get(f"regularizer.{name}.weight")

    def forward(self, factors: Tuple[torch.Tensor], **kwargs):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f) ** 3)

        return norm / factors[0].shape[0]


# reference: https://github.com/facebookresearch/kbc/blob/master/kbc/regularizers.py
@Regularizer.register(name="f2_regularize")
class F2Reg(Regularizer):
    def __init__(self, config: Config, name: str):
        super().__init__(config)

        self.weight = self.config.get(f"regularizer.{name}.weight")

    def forward(self, factors: Tuple[torch.Tensor], **kwargs):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(f ** 2)
        return norm / factors[0].shape[0]


@Regularizer.register(name="lambda3_regularize")
class Lambda3Reg(Regularizer):
    def __init__(self, config: Config, name: str):
        super().__init__(config)

        self.weight = self.config.get(f"regularizer.{name}.weight")

        # TODO(gengyuan): check whether all needed parameters are defined in config file; if not, print logging and load the default number
        # at the moment, throw errors

    def forward(self, factors: Tuple[torch.Tensor], **kwargs):
        ddiff = factors[1:] - factors[:-1]
        rank = int(ddiff.shape[1] / 2)
        diff = torch.sqrt(ddiff[:, :rank] ** 2 + ddiff[:, rank:] ** 2) ** 3
        return self.weight * torch.sum(diff) / (factors.shape[0] - 1)


@InplaceRegularizer.register(name="inplace_renorm_regularize")
class InplaceRenormReg(Regularizer):
    def __init__(self, config: Config, name: str):
        super().__init__(config)

        self.p = self.config.get(f"regularizer.{name}.p")
        self.dim = self.config.get(f"regularizer.{name}.dim")
        self.maxnorm = self.config.get(f"regularizer.{name}.maxnorm")

    def forward(self, factors: Tuple[torch.Tensor], **kwargs):
        for f in factors:
            f.weight.data.renorm_(p=self.p, dim=self.dim, maxnorm=self.maxnorm)

@InplaceRegularizer.register(name="inplace_uniform_regularize")
class InplaceClampReg(Regularizer):
    def __init__(self, config:Config, name:str):
        super().__init__(config)


        self.min = self.config.get(f"regularizer.{name}.min")
        self.max = self.config.get(f"regularizer.{name}.max")

    def forward(self, factors: Tuple[torch.Tensor], **kwargs):
        for f in factors:
            f.weight.data.clamp_(max=self.max, min=self.min)