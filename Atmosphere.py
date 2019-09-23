from dataclasses import dataclass
from enum import Enum, auto
from typing import Sequence
import numpy as np

class ScaleType(Enum):
    Geometric = 0
    ColumnMass = auto()
    Tau5000 = auto()

@dataclass
class Atmosphere:
    scale: ScaleType
    depthScale: Sequence[float]
    temperature: Sequence[float]
    ne: Sequence[float]
    vlos: Sequence[float]
    vturb: Sequence[float]
    hydrogenPops: np.ndarray


