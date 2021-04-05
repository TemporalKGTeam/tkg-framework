import os
import argparse

import ax
import random
from ax.service.ax_client import AxClient

from tkge.task.task import Task
from tkge.task.train_task import TrainTask
from tkge.common.config import Config

from typing import Dict, Tuple


class HPOTask(Task):
    @staticmethod
    def parse_arguments(parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Hyperparameter optimization"""
        subparser = parser.add_parser("hpo", description=description, help="search hyperparameter.")

        subparser.add_argument(
            "-c",
            "--config",
            type=str,
            help="specify configuration file path"
        )

        return subparser

    def __init__(self, config: Config):
        super(HPOTask, self).__init__(config=config)

        self._prepare_experiment()


    def _prepare_experiment(self):
        # initialize a client
        self.ax_client = AxClient()

        # define the search space
        hp_group = self.config.get("hpo.hyperparam")

        self.ax_client.create_experiment(
            name="hyperparam_search",
            parameters=hp_group,
            objective_name="mrr",
            minimize=False,
        )


    def _evaluate(self, parameters, trial_id) -> Dict[str, Tuple[float, float]]:
        """
        evaluate a trial given parameters and return the metrics
        """

        self.config.log(f"Start trial {trial_id}")
        self.config.log(f"with parameters {parameters}")

        # overwrite the config
        trial_config: Config = self.config.clone()
        for k, v in parameters.items():
            trial_config.set(k, v)

        trial_config.create_trial(trial_id)

        # initialize a trainer
        trial_trainer: TrainTask = TrainTask(trial_config)

        # train
        trial_trainer.main()
        best_metric = trial_trainer.best_metric

        self.config.log(f"End trial {trial_id}")
        self.config.log(f"best metric achieved at {best_metric}")

        # evaluate
        return {"mrr": (best_metric, 0.0)}

    def main(self):
        # generate trials/arms
        for i in range(self.config.get("hpo.num_trials")):
            parameters, trial_index = self.ax_client.get_next_trial()
            self.ax_client.complete_trial(trial_index=trial_index, raw_data=self._evaluate(parameters, trial_index))

        best_parameters, values = self.ax_client.get_best_parameters()

        self.config.log("Search task finished.")
        self.config.log(f"Best parameter:"
                        f"{best_parameters}"
                        f""
                        f"Best metrics:"
                        f"{values}")

        result_df = self.ax_client.generation_strategy.trials_as_df
        result_df.to_pickle(os.path.join(self.config.ex_folder, 'trials_as_tf.pkl'))
        self.ax_client.save_to_json_file(filepath=self.config.ex_folder)
