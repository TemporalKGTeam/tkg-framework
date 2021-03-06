import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torch
import unittest

from tkge.models.utils import *
from tkge.models.loss import *
from tkge.models.embedding import *


class TestEmbedding(unittest.TestCase):
    def test_ent_emb(self):
        num = 100
        dim = 128

        emb1 = EntityEmbedding(num=num, dim=dim)

        self.assertEqual(emb1.dim(), 128)
        self.assertEqual(emb1.num(), 100)
        self.assertEqual(emb1.weight.shape, torch.Size([100, 128]))

    def test_ent_emb_position(self):
        num = 100
        dim = 128

        emb = EntityEmbedding(num=num, dim=dim, pos_aware=True)

        self.assertEqual(emb.dim(), 128)
        self.assertEqual(emb.num(), 100)
        self.assertEqual(emb.embedding_dim, 128)
        self.assertEqual(emb.num_embeddings, 200)
        self.assertEqual(emb.weight.shape, torch.Size([200, 128]))

    def test_ent_emb_expanded_dim(self):
        num = 100
        dim = 128

        emb = EntityEmbedding(num=num, dim=dim, expanded_dim_ratio=4)

        self.assertEqual(emb.dim(), 128)
        self.assertEqual(emb.num(), 100)
        self.assertEqual(emb.weight.shape, torch.Size([100, 128 * 4]))

    def test_real_ent(self):
        num = 100
        dim = 128

        emb = RealEntityEmbedding(num=num, dim=dim)

        self.assertEqual(emb.dim(), 128)
        self.assertEqual(emb.num(), 100)
        self.assertEqual(emb.weight.shape, torch.Size([100, 128]))

    def test_complex_ent(self):
        num = 100
        dim = 128

        emb = ComplexEntityEmbedding(num=num, dim=dim)

        self.assertEqual(emb.dim(), 128)
        self.assertEqual(emb.num(), 100)
        self.assertEqual(emb.weight.shape, torch.Size([100, 256]))

    def test_rel(self):
        num = 100
        dim = 128

        emb = RelationEmbedding(num=num, dim=dim)

        self.assertEqual(emb.dim(), 128)
        self.assertEqual(emb.num(), 100)
        self.assertEqual(emb.weight.shape, torch.Size([100, 128]))

    def test_reciprocal_rel(self):
        num = 100
        dim = 128

        emb = RelationEmbedding(num=num, dim=dim, reciprocal=True)

        self.assertEqual(emb.dim(), 128)
        self.assertEqual(emb.num(), 100)
        self.assertEqual(emb.weight.shape, torch.Size([200, 128]))

    def test_rel(self):
        num = 100
        dim = 128

        emb = RelationEmbedding(num=num, dim=dim)

        self.assertEqual(emb.dim(), 128)
        self.assertEqual(emb.num(), 100)
        self.assertEqual(emb.weight.shape, torch.Size([100, 128]))


if __name__ == '__main__':
    unittest.main()
