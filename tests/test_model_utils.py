import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torch
import unittest

from tkge.models.utils import *


class TestEvaluation(unittest.TestCase):
    def test_all_candidates_of_ent_queries(self):
        nan = float('nan')
        q = torch.Tensor([[nan, 1, 2], [3, nan, 2], [3, 2, nan]])

        c = all_candidates_of_ent_queries(q, 4)
        t = torch.Tensor([[0., 1., 2.],
                          [1., 1., 2.],
                          [2., 1., 2.],
                          [3., 1., 2.],
                          [3., 0., 2.],
                          [3., 1., 2.],
                          [3., 2., 2.],
                          [3., 3., 2.],
                          [3., 2., 0.],
                          [3., 2., 1.],
                          [3., 2., 2.],
                          [3., 2., 3.]])

        assert (c == t).all()


if __name__ == '__main__':
    unittest.main()
