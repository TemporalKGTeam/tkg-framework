from typing import Optional

from tkge.common.configurable import Configurable
from tkge.common.config import Config
from tkge.dataset.dataset import Dataset

class Task:
    @staticmethod
    def create(
        config: Config, dataset: Optional[Dataset] = None, parent_job=None, model=None
    ):
        "Create a new job."
        from kge.job import TrainingJob, EvaluationJob, SearchJob

        if dataset is None:
            dataset = Dataset.create(config)

        job_type = config.get("job.type")
        if job_type == "train":
            return TrainingJob.create(
                config, dataset, parent_job=parent_job, model=model
            )
        elif job_type == "search":
            return SearchJob.create(config, dataset, parent_job=parent_job)
        elif job_type == "eval":
            return EvaluationJob.create(
                config, dataset, parent_job=parent_job, model=model
            )
        else:
            raise ValueError("unknown job type")