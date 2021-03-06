import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

from enum import Enum
import os
from collections import defaultdict
from typing import Mapping, Dict, List
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
        self._entity_embeddings = EntityEmbedding(config=config, dataset=dataset)
        self._relation_embeddings = RelationEmbedding(config=config, dataset=dataset)

        if self.config.get('dataset.temporal.index'):
            self._temporal_embeddings = TemporalEmbedding(config=config, dataset=dataset)

        self._fusion: TemporalFusion = TemporalFusion.create_from_name(config)
        self._transformation: Transformation = Transformation.create_from_name(config)

        self._fusion_operand: List = []

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
            temp_float = samples[:, 3:-1]
            for i in range(temp_float.size(1)):
                temp.update({f"level{i}": temp_float[:, i]})

        spot_emb = {'s': self._entity_embeddings(head, 'head'),
                    'p': self._relation_embeddings(rel),
                    'o': self._entity_embeddings(tail, 'tail'),
                    't': temp}

        # fusion

        # get encoded embeddings
        # {'s': embeddings of head embeddings,
        #  'p': embeddings of relation embeddings,
        #  'o': embeddings of tail embeddings}

        fuse_target: List = self.config.get('model.fusion.target')
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

        # transformation
        # scores are vectors of input sample size

        scores = self._transformation(fused_spo_emb['s'], fused_spo_emb['p'], fused_spo_emb['o'])

        return scores, None

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
