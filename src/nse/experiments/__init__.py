from src.nse.experiments.block import Block
from src.nse.experiments.experiment import NSEExperiment
from src.nse.experiments.step import Step
from src.nse.experiments.wing import Wing

EXPERIMENTS: dict[str, type[Step | Block | Wing]] = {
    'step': Step,
    'block': Block,
    'wing': Wing,
}
