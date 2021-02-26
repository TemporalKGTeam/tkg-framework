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
from tkge.models.embedding_space import EmbeddingSpace
from tkge.models.fusion import TemporalFusion
from tkge.models.transformation import Transformation


@BaseModel.register(name='pipeline_model')
class PipelineModel(BaseModel):
    def __init__(self, config: Config, dataset: DatasetProcessor):
        super(PipelineModel, self).__init__(config=config, dataset=dataset)

        self._embedding_space: EmbeddingSpace = EmbeddingSpace.from_config(config)
        self._fusion: TemporalFusion = TemporalFusion.create_from_name(config.get("pipeline_model.fusion.type"))
        self._transformation: Transformation = Transformation.create_from_name(
            config.get("pipeline_model.transformation.type"))

        self._fusion_operand: List = []

    def _prepare_pipeline(self):
        pass

    def _prepare_embedding_space(self):
        pass

    def _fusion_step(self, spot_emb: Dict[torch.Tensor]) -> Dict[torch.Tensor]:
        fused_spo_emb = None

        return fused_spo_emb

    def _transformation_step(self, fused_spo_emb: Dict[torch.Tensor]) -> torch.Tensor:
        pass

    def forward(self, samples: torch.Tensor):
        # check the shape of input samples

        # get embeddings from embedding_space
        spot_emb: Dict[torch.Tensor] = self._embedding_space(samples)

        # fusion
        fused_spo_emb: Dict[torch.Tensor] = self._fusion_step(spot_emb)

        # transformation
        scores = self._transformation_step(fused_spo_emb)

        return scores

    def __repr__(self):
        return f"EmbeddingSpace: {type(self._embedding_space)}" \
               f"Fusion: {type(self._fusion)}" \
               f"Transformation: {type(self._transformation)}"





if __name__ == '__main__':
    print(BaseModel.list_available())
