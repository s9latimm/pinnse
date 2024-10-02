import typing as t

from src.nse.experiments.experiment import NSEExperiment
from src.nse.experiments.step import Step

EXPERIMENTS: t.Dict[str, t.Type[NSEExperiment | Step]] = {
    'step': Step,
}
