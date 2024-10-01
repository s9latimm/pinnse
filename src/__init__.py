import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

torch.manual_seed(42)

random.seed(42)

np.random.seed(42)

HIRES = 10

TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
ROOT_DIR = Path(__file__).parents[1].absolute()
OUTPUT_DIR = ROOT_DIR / 'output'

DEVELOP_MODE = True
