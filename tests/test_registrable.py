import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torch
import unittest

from tkge.models.utils import *
from tkge.models.loss import *


class TestRegistrable(unittest.TestCase):
    def test_loss(self):
        print(Loss.list_available())


if __name__ == '__main__':
    unittest.main()
