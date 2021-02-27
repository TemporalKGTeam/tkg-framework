import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import unittest

from tkge.common.config import Config

module_path = os.path.dirname(__file__)

class TestConfigurable(unittest.TestCase):
    def test_logrankloss(self):
        test_config = Config.create_from_yaml(module_path + '/test_loss_config.yaml')

        print(test_config)


if __name__ == '__main__':
    unittest.main()
