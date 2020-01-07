import lightweaver.constants as C
from copy import copy, deepcopy
import numpy as np
import os
from typing import Tuple
from dataclasses import dataclass
from enum import Enum, auto
from astropy import units
from specutils.utils.wcs_utils import vac_to_air as spec_vac_to_air, air_to_vac as spec_air_to_vac

@dataclass
class NgOptions:
    Norder: int = 0
    Nperiod: int = 0
    Ndelay: int = 0

class InitialSolution(Enum):
    Lte = auto()
    Zero = auto()
    EscapeProbability = auto()

def gaunt_bf(wvl, nEff, charge) -> float:
    # /* --- M. J. Seaton (1960), Rep. Prog. Phys. 23, 313 -- ----------- */
    # Copied from RH, ensuring vectorisation support 
    x = C.HC / (wvl * C.NM_TO_M) / (C.ERydberg * charge**2)
    x3 = x**(1.0/3.0)
    nsqx = 1.0 / (nEff**2 *x)

    return 1.0 + 0.1728 * x3 * (1.0 - 2.0 * nsqx) - 0.0496 * x3**2 \
            * (1.0 - (1.0 - nsqx) * (2.0 / 3.0) * nsqx)

class ConvergenceError(Exception):
    pass

_LwCodeLocation = None
def get_data_path():
    global _LwCodeLocation
    if _LwCodeLocation is None:
        _LwCodeLocation, _ = os.path.split(__file__)
    return _LwCodeLocation + '/../Data/'

def get_default_molecule_path():
    global _LwCodeLocation
    if _LwCodeLocation is None:
        _LwCodeLocation, _ = os.path.split(__file__)
    return _LwCodeLocation + '/../Molecules/'

def vac_to_air(wavelength: np.ndarray) -> np.ndarray:
    return spec_vac_to_air(wavelength * units.nm, method='edlen1966').value

def air_to_vac(wavelength: np.ndarray) -> np.ndarray:
    return spec_air_to_vac(wavelength * units.nm, scheme='iteration', method='edlen1966').value