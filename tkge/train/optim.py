import torch

from typing import Optional, Dict

from tkge.common.error import ConfigurationError


def get_optimizer(type: str, args: Optional[Dict]):
    optim_dict = {
        'Adam': torch.optim.Adam,
        'Adagrad': torch.optim.Adagrad
    }

    try:
        optimizer = optim_dict[type](**args)
    except KeyError:
        raise ConfigurationError(f"Optimizer type {type} specified in config file not supported.")

    return optimizer


def get_scheduler(optimizer: torch.optim.optimizer.Optimizer, type: str, args: Optional[Dict]):
    scheduler_dict = {
        'MultiStepLR': torch.optim.lr_scheduler.MultiStepLR,
        'StepLR': torch.optim.lr_scheduler.StepLR,
        'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR,
        'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR,
        'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'LambdaLR': torch.optim.lr_scheduler.LambdaLR
    }

    try:
        scheduler = scheduler_dict[type](optimizer, **args)
    except KeyError:
        raise ConfigurationError(f"Lr scheduler type {type} specified in config file not supported.")

    return scheduler
