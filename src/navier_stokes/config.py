from datetime import datetime
from pathlib import Path

DEFAULT_NU = .1
DEFAULT_RHO = 1
DEFAULT_STEPS = 0

GRID = (100, 20)
GEOMETRY = (10, 2)
HIRES = 10

TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
ROOT_DIR = Path(__file__).parents[2].absolute()
OUTPUT_DIR = ROOT_DIR / 'output'

DEVELOP_MODE = False
