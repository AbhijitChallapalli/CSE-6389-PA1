import os, random, numpy as np, torch

def set_all_seeds(seed: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic CUDA (if you switch to GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

