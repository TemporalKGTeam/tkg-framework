import torch

import time
import os
import argparse

from typing import Dict, List
from collections import defaultdict

from tkge.task.task import Task
from tkge.data.dataset import DatasetProcessor, SplitDataset
from tkge.train.sampling import NegativeSampler, NonNegativeSampler
from tkge.train.regularization import Regularizer, InplaceRegularizer
from tkge.train.optim import get_optimizer, get_scheduler
from tkge.common.config import Config
from tkge.models.model import BaseModel
from tkge.models.pipeline_model import TransSimpleModel
from tkge.models.loss import Loss
from tkge.eval.metrics import Evaluation