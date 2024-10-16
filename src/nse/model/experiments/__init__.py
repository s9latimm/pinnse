from src.nse.model.experiments.block import Block
from src.nse.model.experiments.curve import Curve
from src.nse.model.experiments.cylinder import Cylinder
from src.nse.model.experiments.empty import Empty
from src.nse.model.experiments.expand import Expand
from src.nse.model.experiments.slalom import Slalom
from src.nse.model.experiments.slit import Slit
from src.nse.model.experiments.step import Step
from src.nse.model.experiments.wing import Wing

EXPERIMENTS: dict[str, type[Empty | Step | Curve | Expand | Slalom | Block
                            | Cylinder | Wing | Slit]] = {
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
