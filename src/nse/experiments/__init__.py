from src.nse.experiments.block import Block
from src.nse.experiments.cylinder import Cylinder
from src.nse.experiments.empty import Empty
from src.nse.experiments.experiment import NSEExperiment
from src.nse.experiments.slit import Slit
from src.nse.experiments.step import Step
from src.nse.experiments.wing import Wing

EXPERIMENTS: dict[str, type[Empty | Cylinder | Step | Block | Wing | Slit]] = {
    'empty': Empty,
    'cylinder': Cylinder,
    'step': Step,
    'block': Block,
    'wing': Wing,
    'slit': Slit,
}
