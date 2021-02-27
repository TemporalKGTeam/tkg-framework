import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import unittest

from tkge.common.desc import *


class TestConfigurable(unittest.TestCase):
    def test_int(self):
        class Helper:
            with self.assertRaises(TypeError):
                i1 = IntegerParam('i1', 10)
                i2 = IntegerParam('i2', 1.3)


if __name__ == '__main__':
    unittest.main()
