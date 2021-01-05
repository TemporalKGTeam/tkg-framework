import argparse

from tkge.task.trainer import TrainTask
from tkge.task.tester import TestTask
from tkge.common.config import Config

desc = 'Temporal KG Completion methods'
parser = argparse.ArgumentParser(description=desc)

# parser.add_argument('train', help='task type', type=bool)
parser.add_argument('-config', help='configuration file folder', type=str)
args = parser.parse_args()

config = Config(folder=args.config, load_default=False)  # TODO load_default is false

# trainer = TrainTask(config)
tester = TestTask(config)
