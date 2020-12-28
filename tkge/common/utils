from typing import Tuple

import torch
import numpy as np


def repeat_interleave(inputs: torch.Tensor, n_tile: int, dim: int = None) -> torch.Tensor:
    init_dim = inputs.size(dim)
    repeat_idx = [1] * inputs.dim()
    repeat_idx[dim] = n_tile

    if hasattr(torch.Tensor, 'repeat_interleave'):
        return torch.Tensor.repeat_interleave(inputs, repeat_idx, dim)
    else:

        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))

        return torch.index_select(inputs, dim, order_index)
