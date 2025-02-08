import random

import numpy as np
import torch


def set_random_seed(seed: int = 42) -> None:
    """Ensure deterministic behavior."""
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
