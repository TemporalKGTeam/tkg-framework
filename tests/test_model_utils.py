import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torch
import unittest

from tkge.models.utils import *
from tkge.common.error import *

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

    def test_forward_checking(self):
        @forward_checking
        def test_func1():
            return 1

        @forward_checking
        def test_func2():
            scores = torch.rand(3,3)
            scores[0,0] = torch.tensor(float('NaN'))
            return scores, []

        self.assertRaises(CodeError, test_func1)
        self.assertRaises(NaNError, test_func2)


if __name__ == '__main__':
    unittest.main()
