import torch
from torch import nn
from torch import Tensor

from typing import Type, Callable, Dict
from collections import defaultdict

from tkge.common.registrable import Registrable
from tkge.common.configurable import Configurable
from tkge.common.config import Config
from tkge.train.regularization import Regularizer


class BaseEmbedding(nn.Embedding):
    def __init__(self, num: int, dim: int):
        super(BaseEmbedding, self).__init__(num_embeddings=num, embedding_dim=dim)

    def dim(self):
        raise NotImplementedError

    def num(self):
        raise NotImplementedError

    def num_embeddings(self):
        return self.weight.size(0)

    def embedding_dim(self):
        return self.weight.size(1)

    def norm(self):
        pass


class EntityEmbedding(BaseEmbedding):
    def __init__(self, num: int, dim: int, pos_aware: bool = False, interleave: bool = False,
                 expanded_dim_ratio: int = 1):
        self._num: int = num
        self._dim: int = dim
        self._pos_aware: bool = pos_aware
        self._interleave: bool = interleave

        expanded_num_ratio = 2 if self._pos_aware else 1

        super(EntityEmbedding, self).__init__(num * expanded_num_ratio, dim * expanded_dim_ratio)

    def dim(self):
        return self._dim

    def num(self):
        return self._num

    def get_by_index(self, index: torch.Tensor):
        """
        behave same as __index__ when pos_aware is false and return head and tail embeddings of the entities
        """
        pass


class RelationEmbedding(BaseEmbedding):
    def __init__(self, num: int, dim: int, reciprocal: bool = False, interleave: bool = False,
                 expanded_dim_ratio: int = 1):
        self._num: int = num
        self._dim: int = dim
        self._reciprocal: bool = reciprocal
        self._interleave: bool = interleave

        expanded_num_ratio = 2 if self._reciprocal else 1

        super(RelationEmbedding, self).__init__(num * expanded_num_ratio, dim * expanded_dim_ratio)

    def dim(self):
        return self._dim

    def num(self):
        return self._num

    def get_by_index(self, index: torch.Tensor):
        """
        behave same as __index__ when pos_aware is false and return original and reciprocal embeddings of the relations
        """
        pass


class TemporalEmbedding(BaseEmbedding):
    def __init__(self, num: int, dim: int):
        self._num: int = num
        self._dim: int = dim

        super(TemporalEmbedding, self).__init__(num, dim)

    def dim(self):
        return self._dim

    def num(self):
        return self._num

    def get_by_index(self, index: torch.Tensor):
        """
        retrieves embeddings by index
        """


class RealEntityEmbedding(EntityEmbedding):
    def __init__(self, num: int, dim: int, pos_aware: bool = False, interleave: bool = False):
        super(RealEntityEmbedding, self).__init__(num=num, dim=dim, pos_aware=pos_aware, interleave=interleave,
                                                  expanded_dim_ratio=1)


class ComplexEntityEmbedding(EntityEmbedding):
    def __init__(self, num: int, dim: int, pos_aware: bool = False, interleave: bool = False):
        super(ComplexEntityEmbedding, self).__init__(num=num, dim=dim, pos_aware=pos_aware, interleave=interleave,
                                                     expanded_dim_ratio=2)

    def re(self):
        pass

    def im(self):
        pass


class TranslationRelationEmbedding(RelationEmbedding):
    pass


class QuaternionRelationEmbedding(RelationEmbedding):
    pass


class DualQuaternionRelationEmbedding(RelationEmbedding):
    pass


# class BaseEmbedder(nn.Embedding):
#     """
#     Base class for all embedders of a fixed number of objects including entities, relations and timestamps
#     """
# 
#     def __init__(self,
#                  num_embeddings: int,
#                  embedding_dim: int,
#                  initializer: str = "uniform",
#                  init_args: Dict = {"from_": 0, "to": 0},
#                  reg: str = "renorm"):
#         super().__init__(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
#
#         if initializer == "uniform":
#             self.weight.data.uniform_(**init_args)
#         else:
#             raise NotImplementedError
#
#         self.regularizer = Regularizer.create(reg)  # TODO: modify regularizer
#
#     def forward(self, indexes: Tensor) -> Tensor:
#         return self.embed(indexes)
#
#     def embed(self, indexes: Tensor) -> Tensor:
#         raise NotImplementedError
#
#     def embed_all(self) -> Tensor:
#         raise NotImplementedError
#
#     def regularize(self):  # TODO should it be inplace?
#
#         pass
#
#     def split(self, sep):
#         return self.embed(Tensor[range(sep)])
#
#     def __getitem__(self, item):
#         pass

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
