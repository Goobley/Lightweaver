import lightweaver.constants as C
from copy import copy, deepcopy
import numpy as np
import os
from typing import Tuple, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from astropy import units
from specutils.utils.wcs_utils import vac_to_air as spec_vac_to_air, air_to_vac as spec_air_to_vac
from numba import njit
from scipy import special

@dataclass
class NgOptions:
    Norder: int = 0
    Nperiod: int = 0
    Ndelay: int = 0

class InitialSolution(Enum):
    Lte = auto()
    Zero = auto()
    EscapeProbability = auto()

def voigt_H(a, v):
    z = (v + 1j * a)
    return special.wofz(z).real

@njit
def planck(temp, wav):
    hc_Tkla = C.HC / (C.KBoltzmann * C.NM_TO_M * wav) / temp
    twohnu3_c2 = (2.0 * C.HC) / (C.NM_TO_M * wav)**3

    return twohnu3_c2 / (np.exp(hc_Tkla) - 1.0)

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

class ExplodingMatrixError(Exception):
    pass

_LwCodeLocation = None
def get_data_path():
    global _LwCodeLocation
    if _LwCodeLocation is None:
        _LwCodeLocation, _ = os.path.split(__file__)
    return _LwCodeLocation + '/Data/'

def get_default_molecule_path():
    global _LwCodeLocation
    if _LwCodeLocation is None:
        _LwCodeLocation, _ = os.path.split(__file__)
    return _LwCodeLocation + '/Data/DefaultMolecules/'

def vac_to_air(wavelength: np.ndarray) -> np.ndarray:
    return spec_vac_to_air(wavelength * units.nm, method='edlen1966').value

def air_to_vac(wavelength: np.ndarray) -> np.ndarray:
    return spec_air_to_vac(wavelength * units.nm, scheme='iteration', method='edlen1966').value

def convert_specific_intensity(wavelength: np.ndarray, specInt: np.ndarray, outUnits) -> units.quantity.Quantity:
    if not isinstance(wavelength, units.Quantity):
        wavelength = wavelength << units.nm

    if not isinstance(specInt, units.Quantity):
        specInt = specInt << units.J / units.s / units.m**2 / units.sr / units.Hz

    return specInt.to(outUnits, equivalencies=units.spectral_density(wavelength))

class CrswIterator:
    def __init__(self, initVal=1e3):
        self.val = initVal

    def __call__(self):
        self.val = max(1.0, self.val * 0.1**(1.0/self.val))
        return self.val

class UnityCrswIterator(CrswIterator):
    def __init__(self):
        super().__init__(1.0)

    def __call__(self):
        return self.val

def sequence_repr(x: Sequence) -> str:
    if isinstance(x, np.ndarray):
        return repr(x.tolist())

    return repr(x)

def view_flatten(x: np.ndarray) -> np.ndarray:
    y = x.view()
    y.shape = (x.size,)
    return y
