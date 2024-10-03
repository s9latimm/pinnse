import typing as tp

from src.nse.experiments.block import Block
from src.nse.experiments.experiment import NSEExperiment
from src.nse.experiments.step import Step
from src.nse.experiments.wing import Wing

EXPERIMENTS: tp.Dict[str, tp.Type[NSEExperiment]] = {
    'step': Step,
    'block': Block,
    'wing': Wing,
}
