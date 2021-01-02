import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(BASE_DIR)

import torch
import unittest
from unittest import mock
from tkge.eval.metrics import Evaluation


class MockEvaluation(Evaluation):
    def __init__(self, filter: str, ordering: str, tail_or_head: str):
        self.filter = filter
        self.ordering = ordering
        self.tail_or_head = tail_or_head


class TestEvaluation(unittest.TestCase):

    def test_ranks_raw_opt_head(self):
        eval = MockEvaluation(filter='off', ordering='pessimistic', tail_or_head='_po')
        torch.manual_seed(0)
        random_scores = torch.rand((1000, 1000))
        targets = torch.zeros((1000,)).long()
        filtered_mask = torch.zeros((1000, 1000)).long()

        ranks = eval.ranking(random_scores, targets, filtered_mask)

        ranks_gt = torch.load(f"{BASE_DIR}/tests/assets/test_ranks_raw_opt_head.pt")

        assert all(ranks == ranks_gt)


if __name__ == '__main__':
    unittest.main()
