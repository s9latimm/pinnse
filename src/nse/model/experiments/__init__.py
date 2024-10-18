from src.nse.model.experiments.block import Block
from src.nse.model.experiments.cylinder import Cylinder
from src.nse.model.experiments.empty import Empty
from src.nse.model.experiments.slalom import Slalom
from src.nse.model.experiments.slit import Slit
from src.nse.model.experiments.step import Step
from src.nse.model.experiments.wing import Wing

EXPERIMENTS: dict[str, type[Empty | Step | Slalom | Block | Cylinder | Wing | Slit]] = {
    'empty': Empty,
    'step': Step,
    'slalom': Slalom,
    'block': Block,
    'slit': Slit,
    'cylinder': Cylinder,
    'wing': Wing,
}
