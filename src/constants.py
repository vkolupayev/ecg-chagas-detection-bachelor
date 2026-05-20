# import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # for deterministic pytorch
# OR run scripts by setting the environment variable via the shell
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m scripts_experiment.cv_ft_hubert

import random
import numpy as np
import torch

RANDOM_SEED = 42
LOG_LEVEL = "INFO"


def seed_everything(seed: int) -> None:
    r"""Sets the seed for generating random numbers in :pytorch:`PyTorch`,
    :obj:`numpy` and :python:`Python`.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


# torch.manual_seed() will seed the rng generator for workers.
# For further control generator and worker can be seeded
# def seeded_generator(seed):
#     g = torch.Generator()
#     g.manual_seed(seed)

# def seed_worker(worker_id):
#     worker_seed = RANDOM_SEED + worker_id
#     random.seed(worker_seed)
#     np.random.seed(worker_seed)
#     torch.manual_seed(worker_seed)
