import random
import numpy as np
import torch
from bm.config import Config


def set_seed(value: int = Config.random_seed):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def get_worker_seeder(seed_value: int = Config.random_seed):
    def seed_worker(worker_id):
        np.random.seed(seed_value)
        random.seed(seed_value)

    return seed_worker
