import torch
from torch import nn
from torch import Tensor

from typing import Type, Callable, Dict
from collections import defaultdict

from tkge.common.registry import Registrable
from tkge.common.configurable import Configurable
from tkge.common.config import Config
from tkge.train.regularization import Regularizer


class BaseEmbedder(nn.Embedding):
    """
    Base class for all embedders of a fixed number of objects including entities, relations and timestamps
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 initializer: str = "uniform",
                 init_args: Dict = {"from_": 0, "to": 0},
                 reg: str = "renorm"):
        super().__init__(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        if initializer == "uniform":
            self.weight.data.uniform_(**init_args)
        else:
            raise NotImplementedError

        self.regularizer = Regularizer.create(reg)  # TODO: modify regularizer

    def forward(self, indexes: Tensor) -> Tensor:
        return self.embed(indexes)

    def embed(self, indexes: Tensor) -> Tensor:
        raise NotImplementedError

    def embed_all(self) -> Tensor:
        raise NotImplementedError

    def regularize(self):  # TODO should it be inplace?

        pass

    def split(self, sep):
        return self.embed(Tensor[range(sep)])

    def __getitem__(self, item):
        pass


# class BaseEmbedding(nn.Module, Registrable):
#     def __init__(self):
#         super(BaseEmbedding, self).__init__()
#
#         self.params: Dict[str, Tensor] = defaultdict(dict)
#
#     def build(self):
#         raise NotImplementedError
#
#         ## build up all embeddings needed at one time, and do the regularization
#         # self.params["new_key"] = nn.Parameter(num_emb, emb_dim)
#         # self.params["new_key"].init()
#
#     def forward(self, indexes):
#         raise NotImplementedError
#
#         ## return the score of indexed embedding
#         # return embedding_package
#
# class TkgeModel(nn.Module, Registrable):
#     def __init__(self, config: Config):
#         nn.Module().__init__()
#         Registrable().__init__(config=config)
#
#         # Customize
#         self._relation_embedder = BaseEmbedder()
#         self._entity_embedder = BaseEmbedder()
#
#     def forward(self):
#         # 模型主要代码写在这里
#         # return score
#         pass
#
#     def load_chpt(self):
#         pass





