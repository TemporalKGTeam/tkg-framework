import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

from enum import Enum
import os
from collections import defaultdict
from typing import Mapping, Dict, List, Any
import random

from tkge.common.registrable import Registrable
from tkge.common.config import Config
from tkge.common.error import ConfigurationError
from tkge.data.dataset import DatasetProcessor
from tkge.models.layers import LSTMModel
from tkge.models.model import BaseModel
from tkge.models.utils import *
from tkge.models.embedding_space import EntityEmbedding, RelationEmbedding, TemporalEmbedding
from tkge.models.fusion import TemporalFusion
from tkge.models.transformation import Transformation


@BaseModel.register(name='pipeline_model')
class PipelineModel(BaseModel):
    def __init__(self, config: Config, dataset: DatasetProcessor):
        super(PipelineModel, self).__init__(config=config, dataset=dataset)

        # self._embedding_space: EmbeddingSpace = EmbeddingSpace.from_config(config)
        self._entity_embeddings: EntityEmbedding = EntityEmbedding(config=config, dataset=dataset)
        self._relation_embeddings: RelationEmbedding = RelationEmbedding(config=config, dataset=dataset)

        if not isinstance(self.config.get('model.embedding.temporal'), type(None)):
            self._temporal_embeddings: TemporalEmbedding = TemporalEmbedding(config=config, dataset=dataset)

        self._fusion: TemporalFusion = TemporalFusion.create_from_name(config)
        self._transformation: Transformation = Transformation.create_from_name(config)

        self._fusion_operand: List = []

        self._inverse_scorer = self.config.get("model.scorer.inverse")

        # import pprint
        #
        # pprint.pprint({n: p.size() for n, p in self.named_parameters()})
        # assert False

    def forward(self, samples: torch.Tensor):
        # check the shape of input samples

        # get embeddings from embedding_space
        # {'s': embeddings of head embeddings,
        #  'p': embeddings of relation embeddings,
        #  'o': embeddings of tail embeddings,
        #  't': embeddings of temporal information}

        # spot_emb: Dict[torch.Tensor] = self._embedding_space(samples)
        head = samples[:, 0].long()
        rel = samples[:, 1].long()
        tail = samples[:, 2].long()

        temp = {}

        # if self.config.get('dataset.temporal.index') and not self.config.get('dataset.temporal.float'):
        #     if samples.size(1)==4:
        #         temp_index = samples[:, -1]
        #         temp.update(self._temporal_embeddings(temp_index.long()))
        #     else:
        #         temp_indexes = samples[:, 3:]
        #         for i in range(temp_indexes.size(1)):
        #             temp_embs = self._temporal_embeddings(temp_indexes[:, i:i + 1].long())
        #             temp_embs = {f"level{i}_{k}": v for k, v in temp_embs.items()}
        #             temp.update(temp_embs)

        # if self.config.get('dataset.temporal.float'):
        #     temp_float = samples[:, 3:-1] if self.config.get('dataset.temporal.index') else samples[:, 3:]
        #     for i in range(temp_float.size(1)):
        #         temp.update({f"level{i}": temp_float[:, i:i + 1]})

        if self.config.get('dataset.temporal.index') and not self.config.get('dataset.temporal.float'):
            temp_index = samples[:, -1]
            temp.update(self._temporal_embeddings(temp_index.long()))

        if self.config.get('dataset.temporal.float'):
            temp_float = samples[:, 3:-1] if self.config.get('dataset.temporal.index') else samples[:, 3:]
            for i in range(temp_float.size(1)):
                if not isinstance(self.config.get('model.embedding.temporal'), type(None)):
                    # TODO: dangerous
                    temp_embs = self._temporal_embeddings(temp_float[:, i:i + 1].long())
                    temp_embs = {f"level{i}_{k}": v for k, v in temp_embs.items()}
                    temp.update(temp_embs)
                else:
                    temp.update({f"level{i}": temp_float[:, i:i + 1]})

        spot_emb = {'s': self._entity_embeddings(head, 'head'),
                    'p': self._relation_embeddings(rel, inverse_relation=False),
                    'o': self._entity_embeddings(tail, 'tail'),
                    't': temp}

        if self._inverse_scorer:
            spot_emb_inv = {'s': self._entity_embeddings(tail, 'head'),
                            'p': self._relation_embeddings(rel, inverse_relation=True),
                            'o': self._entity_embeddings(head, 'tail'),
                            't': temp}

        # fusion

        # get encoded embeddings
        # {'s': embeddings of head embeddings,
        #  'p': embeddings of relation embeddings,
        #  'o': embeddings of tail embeddings}

        fuse_target: List = self.config.get('model.fusion.target')

        fused_spo_emb = self._fuse(spot_emb, fuse_target)

        if self._inverse_scorer:
            fused_spo_emb_inv = self._fuse(spot_emb_inv, fuse_target)

        # transformation
        # scores are vectors of input sample size

        scores = self._transformation(fused_spo_emb['s'], fused_spo_emb['p'], fused_spo_emb['o'])

        if self._inverse_scorer:
            scores_inv = self._transformation(fused_spo_emb_inv['s'], fused_spo_emb_inv['p'], fused_spo_emb_inv['o'])

            scores = (scores + scores_inv) / 2

        factors = {"entity_reg": list(self._entity_embeddings.parameters()),
                   "relation_reg": list(self._relation_embeddings.parameters())
                   }

        if hasattr(self, '_temporal_embeddings'):
            factors.update({'temporal_reg': list(getattr(self, '_temporal_embeddings').parameters())})

        return scores, factors

    def _fuse(self, spot_emb, fuse_target):
        fused_spo_emb = dict()
        if 'ent+temp' in fuse_target:
            fused_spo_emb['s'] = self._fusion(spot_emb['s'], spot_emb['t'])
            fused_spo_emb['o'] = self._fusion(spot_emb['o'], spot_emb['t'])
        else:
            fused_spo_emb['s'] = spot_emb['s']
            fused_spo_emb['o'] = spot_emb['o']
        if 'rel+temp' in fuse_target:
            fused_spo_emb['p'] = self._fusion(spot_emb['p'], spot_emb['t'])
        else:
            fused_spo_emb['p'] = spot_emb['p']
        return fused_spo_emb

    def predict(self, queries: torch.Tensor):
        assert torch.isnan(queries).sum(1).byte().all(), "Either head or tail should be absent."

        bs = queries.size(0)
        dim = queries.size(0)

        candidates = all_candidates_of_ent_queries(queries, self.dataset.num_entities())

        scores, _ = self.forward(candidates)
        scores = scores.view(bs, -1)

        return scores

    def fit(self, samples: torch.Tensor):
        bs = samples.size(0)
        dim = samples.size(1) // (1 + self.config.get("negative_sampling.num_samples"))

        samples = samples.view(-1, dim)

        scores, factors = self.forward(samples)
        scores = scores.view(bs, -1)

        return scores, factors


@BaseModel.register(name='translation_simple_model')
class TransSimpleModel(BaseModel):
    def __init__(self, config: Config, dataset: DatasetProcessor):
        super(TransSimpleModel, self).__init__(config=config, dataset=dataset)

        # self._embedding_space: EmbeddingSpace = EmbeddingSpace.from_config(config)
        self._entity_embeddings = EntityEmbedding(config=config, dataset=dataset)
        self._relation_embeddings = RelationEmbedding(config=config, dataset=dataset)

        if self.config.get('dataset.temporal.index'):
            self._temporal_embeddings = TemporalEmbedding(config=config, dataset=dataset)

        self._fusion: TemporalFusion = TemporalFusion.create_from_name(config)
        self._transformation: Transformation = Transformation.create_from_name(config)

        self._fusion_operand: List = []

        self._inverse_scorer = self.config.get("model.scorer.inverse")

        self.dropout = torch.nn.Dropout(p=self.config.get('model.p'))

        # import pprint
        #
        # pprint.pprint({n: p.size() for n, p in self.named_parameters()})
        # assert False

    def forward(self, samples: torch.Tensor):
        # check the shape of input samples

        # get embeddings from embedding_space
        # {'s': embeddings of head embeddings,
        #  'p': embeddings of relation embeddings,
        #  'o': embeddings of tail embeddings,
        #  't': embeddings of temporal information}

        # spot_emb: Dict[torch.Tensor] = self._embedding_space(samples)
        head = samples[:, 0].long()
        rel = samples[:, 1].long()
        tail = samples[:, 2].long()

        temp = {}

        if self.config.get('dataset.temporal.index'):
            temp_index = samples[:, -1]
            temp.update(self._temporal_embeddings(temp_index.long()))

        if self.config.get('dataset.temporal.float'):
            temp_float = samples[:, 3:-1] if self.config.get('dataset.temporal.index') else samples[:, 3:]
            for i in range(temp_float.size(1)):
                temp.update({f"level{i}": temp_float[:, i:i + 1]})

        spot_emb = {'s': self._entity_embeddings(head, 'head'),
                    'p': self._relation_embeddings(rel, inverse_relation=False),
                    'o': self._entity_embeddings(tail, 'tail'),
                    't': temp}

        if self._inverse_scorer:
            spot_emb_inv = {'s': self._entity_embeddings(tail, 'head'),
                            'p': self._relation_embeddings(rel, inverse_relation=True),
                            'o': self._entity_embeddings(head, 'tail'),
                            't': temp}

        # fusion

        # get encoded embeddings
        # {'s': embeddings of head embeddings,
        #  'p': embeddings of relation embeddings,
        #  'o': embeddings of tail embeddings}

        fuse_target: List = self.config.get('model.fusion.target')

        fused_spo_emb = self._fuse(spot_emb, fuse_target)

        if self._inverse_scorer:
            fused_spo_emb_inv = self._fuse(spot_emb_inv, fuse_target)

        # transformation
        # scores are vectors of input sample size

        # dropot
        # fused_spo_emb['s']['real'] = self.dropout(fused_spo_emb['s']['real'])
        # fused_spo_emb['p']['real'] = self.dropout(fused_spo_emb['p']['real'])
        # fused_spo_emb['o']['real'] = self.dropout(fused_spo_emb['o']['real'])
        scores = self._transformation(fused_spo_emb['s'], fused_spo_emb['p'], fused_spo_emb['o'])

        if self._inverse_scorer:
            # fused_spo_emb_inv['s']['real'] = self.dropout(fused_spo_emb_inv['s']['real'])
            # fused_spo_emb_inv['p']['real'] = self.dropout(fused_spo_emb_inv['p']['real'])
            # fused_spo_emb_inv['o']['real'] = self.dropout(fused_spo_emb_inv['o']['real'])
            scores_inv = self._transformation(fused_spo_emb_inv['s'], fused_spo_emb_inv['p'], fused_spo_emb_inv['o'])

            scores = (scores + scores_inv) / 2

        factors = {
            "n3": (torch.sqrt(self._entity_embeddings._head['real'].weight ** 2),
                   torch.sqrt(self._entity_embeddings._tail['real'].weight ** 2),
                   torch.sqrt(self._relation_embeddings._relation['real'].weight ** 2)),
            "lambda3": self._temporal_embeddings.get_weight('real')
        }

        return scores, factors

    def _fuse(self, spot_emb: Dict[str, Any], fuse_target: str):
        fused_spo_emb = dict()
        if 'ent+temp' in fuse_target:
            fused_spo_emb['s'] = self._fusion(spot_emb['s'], spot_emb['t'])
            fused_spo_emb['o'] = self._fusion(spot_emb['o'], spot_emb['t'])
        else:
            fused_spo_emb['s'] = spot_emb['s']
            fused_spo_emb['o'] = spot_emb['o']
        if 'rel+temp' in fuse_target:
            fused_spo_emb['p'] = self._fusion(spot_emb['p'], spot_emb['t'])
        else:
            fused_spo_emb['p'] = spot_emb['p']
        return fused_spo_emb

    def predict(self, queries: torch.Tensor):
        assert torch.isnan(queries).sum(1).byte().all(), "Either head or tail should be absent."

        bs = queries.size(0)
        dim = queries.size(0)

        candidates = all_candidates_of_ent_queries(queries, self.dataset.num_entities())

        scores, _ = self.forward(candidates)
        scores = scores.view(bs, -1)

        return scores

    def fit(self, samples: torch.Tensor):
        bs = samples.size(0)
        dim = samples.size(1) // (1 + self.config.get("negative_sampling.num_samples"))

        samples = samples.view(-1, dim)

        scores, factor = self.forward(samples)
        scores = scores.view(bs, -1)

        return scores, factor


if __name__ == '__main__':
    print(BaseModel.list_available())
