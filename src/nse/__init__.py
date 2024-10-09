from src.nse.experiments.block import Block
from src.nse.experiments.experiment import NSEExperiment
from src.nse.experiments.step import Step
from src.nse.experiments.wing import Wing

EXPERIMENTS: dict[str, type[Step | Block | Wing]] = {
    'step': Step,
    'block': Block,
    'wing': Wing,
}

DEFAULT_NU: float = 0.08
DEFAULT_RHO: float = 1.
DEFAULT_INTAKE: float = 5.
DEFAULT_STEPS: int = 1
