import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import unittest

from tkge.common.config import Config
from tkge.models.loss import Loss

module_path = os.path.dirname(__file__)


class TestConfigurable(unittest.TestCase):
    def test_logrankloss(self):
        test_config = Config.create_from_yaml(module_path + '/test_loss_config.yaml')
        test_config = Config.create_from_parent(parent_config=test_config, child_key='log_rank_loss1')

        loss = Loss.create(config=test_config)

        # print(loss.__dict__)
        self.assertEqual(loss.device, 'cpu')
        self.assertEqual(loss.gamma, 130)
        self.assertEqual(loss.temp, 0.5)

    def test_crossentropyloss(self):
        test_config = Config.create_from_yaml(module_path + '/test_loss_config.yaml')
        test_config = Config.create_from_parent(parent_config=test_config, child_key='cross_entropy_loss1')

        loss = Loss.create(config=test_config)

        # print(loss.__dict__)
        self.assertEqual(loss.device, 'cpu')

    def test_marginrankloss(self):
        test_config = Config.create_from_yaml(module_path + '/test_loss_config.yaml')
        test_config = Config.create_from_parent(parent_config=test_config, child_key='margin_rank_loss1')

        loss = Loss.create(config=test_config)

        # print(loss.__dict__)
        self.assertEqual(loss.device, 'cpu')
        self.assertEqual(loss.margin, 100)


if __name__ == '__main__':
    unittest.main()
