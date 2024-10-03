import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

torch.manual_seed(42)

random.seed(42)

np.random.seed(42)

HIRES: float = 10.

TIMESTAMP: str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
ROOT_DIR: Path = Path(__file__).parents[1].absolute()
OUTPUT_DIR: Path = ROOT_DIR / 'output'

DEVELOP_MODE: bool = True
