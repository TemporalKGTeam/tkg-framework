from tkge.models.model import ATiSEModel
from tkge.data.dataset import DatasetProcessor
from tkge.common.config import Config

config =

dataset = DatasetProcessor.create(config=self.config)
model = BaseModel.create(config=self.config, dataset=self.dataset)