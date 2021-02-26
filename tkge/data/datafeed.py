from typing import Dict, List, Tuple

from tkge.data.dataset import DatasetProcessor
from tkge.common.config import Config
from tkge.common.registrable import Registrable
from tkge.common.error import ConfigurationError


class DataFeeder(Registrable):
    def __init__(self, config=Config, dataset=DatasetProcessor):
        super().__init__(config=config)


