import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import unittest

from tkge.common.paramtype import *


class TestConfigurable(unittest.TestCase):
    def test_int(self):
        class Helper:
            i1 = IntegerParam('i1', 10)
            # i2 = IntegerParam('i2', 1.3)
            n1 = NumberParam('n1', 150)
            device = DeviceParam('device', 'cuda')

            def __init__(self):
                self.i1 = 10

        c = Helper()
        # self.assertEqual(c.i1, 10)
        print(c.device)


if __name__ == '__main__':
    unittest.main()
