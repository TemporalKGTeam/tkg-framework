import torch

from typing import List, Tuple, Dict
from collections import defaultdict

from tkge.common.config import Config
from tkge.common.configurable import Configurable
from tkge.common.error import ConfigurationError
from tkge.data.dataset import DatasetProcessor

import enum

SPOT = enum.Enum('spot', ('s', 'p', 'o', 't'))


class Evaluation(Configurable):
    def __init__(self, config: Config, dataset: DatasetProcessor):
        super().__init__(config=config)

        self.dataset = dataset

        self.filter = self.config.get("eval.filter")
        self.ordering = self.config.get("eval.ordering")
        self.k = self.config.get("eval.k")

        self.filtered_data = defaultdict(None)
        self.filtered_data['sp_'] = self.dataset.filter(type=self.filter, target='o')
        self.filtered_data['_po'] = self.dataset.filter(type=self.filter, target='s')

    def eval(self, queries, scores, miss='s'):
        metrics = {}


def ranking(scores: torch.Tensor, targets: torch.Tensor, filtered_index: torch.Tensor, ordering="optimistic"):
    query_size = scores.size(0)
    vocabulary_size = scores.size(1)

    target_scores = scores[range(query_size), targets].unsqueeze(1).repeat((1, vocabulary_size))

    scores[filtered_index[0], filtered_index[1]] = 0.0

    if ordering == "optimistic":
        comp = scores.gt(target_scores)
    else:
        comp = scores.ge(target_scores)

    ranks = comp.sum(1) + 1

    return ranks


def filter_query(queries: List[str], filtered_list: Dict[str, List], device: str = "cpu"):
    filtered_index = [[], []]
    for i, q in enumerate(queries):
        for j in filtered_list[q]:
            filtered_index[0].append(i)
            filtered_index[1].append(j)

    filtered_index = torch.Tensor(filtered_index).to(device)

    return filtered_index


def mean_ranking(scores: torch.Tensor, targets: torch.Tensor, ):
    ranks = ranking(scores, targets, )
    mr = torch.mean(ranks).item()

    return mr


def mean_reciprocal_ranking():
    ranks = ranking()
    mrr = torch.mean(1. / ranks).item()

    return mrr


def hits(queries, at: Tuple[int] = [1, 3, 10], miss='s', optimistics=False, filter="off"):
    # TODO(gengyuan) missing should be enum 's' 'p'
    assert miss in ["s", "p", "o"], "Only support s(ubject)/p(redicate)/o(bject) prediction task"

    ranks = mean_ranking()
    hits_at = list(map(lambda x: torch.mean((ranks <= x).float()).item(), at))
