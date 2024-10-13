from src.nse.experiments.block import Block
from src.nse.experiments.curve import Curve
from src.nse.experiments.cylinder import Cylinder
from src.nse.experiments.empty import Empty
from src.nse.experiments.expand import Expand
from src.nse.experiments.slalom import Slalom
from src.nse.experiments.slit import Slit
from src.nse.experiments.step import Step
from src.nse.experiments.wing import Wing

EXPERIMENTS: dict[str, type[Empty | Step | Curve | Expand | Slalom | Block | Cylinder | Wing | Slit]] = {
    'empty': Empty,
    'step': Step,
    'curve': Curve,
    'expand': Expand,
    'slalom': Slalom,
    'block': Block,
    'cylinder': Cylinder,
    'wing': Wing,
    'slit': Slit,
}
