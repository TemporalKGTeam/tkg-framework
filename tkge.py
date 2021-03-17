import argparse
import os

from tkge.task.task import Task
from tkge.task.trainer import TrainTask
from tkge.task.tester import TestTask
from tkge.task.search import SearchTask
from tkge.task.resume import ResumeTask
from tkge.common.config import Config

desc = 'Temporal KG Completion methods'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument(
    "--version",
    "-V",
    action="version",
    version=f"work in progress"
)

# parser.add_argument('train', help='task type', type=bool)
# parser.add_argument('--config', help='configuration file', type=str)
# parser.add_argument('--help', help='help')

subparsers = parser.add_subparsers(title="task",
                                   description="valid tasks: train, evaluate, predict, search",
                                   dest="task")

# subparser train
parser_train = TrainTask.parse_arguments(subparsers)
parser_eval = TestTask.parse_arguments(subparsers)

args = parser.parse_args()

task_dict = {
    'train': TrainTask,
    'eval': TestTask,
    'search': SearchTask,
    'resume': ResumeTask
}

config_path = args.config if args.task is not 'resume' else os.path.join(args.experiment, 'config.yaml')
config = Config.create_from_yaml(config_path)  # TODO load_default is false
task = task_dict[args.task](config)

task.main()

# trainer = TrainTask(config)
# tester = TestTask(config)
