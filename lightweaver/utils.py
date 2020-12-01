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
    '''
    Container for the options related to Ng acceleration.
    Attributes
    ----------
    Norder : int, optional
        The order of the extrapolation to use (default: 0, i.e. none).
    Nperiod : int, optional
        The number of iterations to run between extrapolations.
    Ndelay : int, optional
        The number of iterations to run before starting acceleration.
    '''
    Norder: int = 0
    Nperiod: int = 0
    Ndelay: int = 0

class InitialSolution(Enum):
    '''
    Initial solutions to use for atomic populations, either LTE, Zero
    radiation (not yet supported), or second order escape probability.
    '''
    Lte = auto()
    Zero = auto()
    EscapeProbability = auto()

def voigt_H(a, v):
    '''
    Scalar Voigt profile.

    Parameters
    ----------
    a : float or array-like
        The a damping parameter to be used in the Voigt profile.
    v : float or array-like
        The position in the line profile in Doppler units.
    '''
    z = (v + 1j * a)
    return special.wofz(z).real

@njit
def planck(temp, wav):
    '''
    Planck black-body function B_nu(T) from wavelength.

    Parameters
    ----------
    temp : float or array-like
        Temperature [K]
    wav : float or array-like
        The wavelength at which to compute B_nu [nm].

    Returns
    -------
    result : float or array-like
        B_nu(T)
    '''
    hc_Tkla = C.HC / (C.KBoltzmann * C.NM_TO_M * wav) / temp
    twohnu3_c2 = (2.0 * C.HC) / (C.NM_TO_M * wav)**3

    return twohnu3_c2 / (np.exp(hc_Tkla) - 1.0)

def gaunt_bf(wvl, nEff, charge) -> float:
    '''
    Gaunt factor for bound-free transitions, from Seaton (1960), Rep. Prog.
    Phys. 23, 313, as used in RH.

    Parameters
    ----------
    wvl : float or array-like
        The wavelength at which to compute the Gaunt factor [nm].
    nEff : float
        Principal quantum number.
    charge : float
        Charge of free state.

    Returns
    -------
    result : float or array-like
        Gaunt factor for bound-free transitions.
    '''
    # /* --- M. J. Seaton (1960), Rep. Prog. Phys. 23, 313 -- ----------- */
    # Copied from RH, ensuring vectorisation support
    x = C.HC / (wvl * C.NM_TO_M) / (C.ERydberg * charge**2)
    x3 = x**(1.0/3.0)
    nsqx = 1.0 / (nEff**2 *x)

    return 1.0 + 0.1728 * x3 * (1.0 - 2.0 * nsqx) - 0.0496 * x3**2 \
            * (1.0 - (1.0 - nsqx) * (2.0 / 3.0) * nsqx)

class ConvergenceError(Exception):
    '''
    Raised by some iteration schemes, can also be used in user code.
    '''
    pass

class ExplodingMatrixError(Exception):
    '''
    Raised by the linear system matrix solver in the case of unsolvable
    systems.
    '''
    pass

_LwCodeLocation = None
def get_data_path():
    '''
    Returns the location of the Lightweaver support data.
    '''
    global _LwCodeLocation
    if _LwCodeLocation is None:
        _LwCodeLocation, _ = os.path.split(__file__)
    return _LwCodeLocation + '/Data/'

def get_default_molecule_path():
    '''
    Returns the location of the default molecules taken from RH.
    '''
    global _LwCodeLocation
    if _LwCodeLocation is None:
        _LwCodeLocation, _ = os.path.split(__file__)
    return _LwCodeLocation + '/Data/DefaultMolecules/'

def vac_to_air(wavelength: np.ndarray) -> np.ndarray:
    '''
    Convert vacuum wavelength to air.

    Parameters
    ----------
    wavelength : float or array-like or astropy.Quantity
        If no units then the wavelength is assumed to be in [nm], otherwise the provided units are used.

    Returns
    -------
    result : float or array-like or astropy.Quantity
        The converted wavelength in [nm].
    '''
    return spec_vac_to_air(wavelength << units.nm, method='edlen1966').value

def air_to_vac(wavelength: np.ndarray) -> np.ndarray:
    '''
    Convert air wavelength to vacuum.

    Parameters
    ----------
    wavelength : float or array-like or astropy.Quantity
        If no units then the wavelength is assumed to be in [nm], otherwise the provided units are used.

    Returns
    -------
    result : float or array-like or astropy.Quantity
        The converted wavelength in [nm].
    '''
    return spec_air_to_vac(wavelength << units.nm, scheme='iteration', method='edlen1966').value

def convert_specific_intensity(wavelength: np.ndarray,
                               specInt: np.ndarray, outUnits) -> units.quantity.Quantity:
    '''
    Convert a specific intensity between different units.

    Parameters
    ----------
    wavelength : np.ndarray or astropy.Quantity
        If no units are provided then this is assumed to be in nm.
    specInt : np.ndarray or astropy.Quantity
        If no units are provided then this is assumed to be in J/s/m2/sr/Hz,
        the default for Lightweaver.
    outUnits : str or astropy.Unit
        The units to convert specInt to e.g. 'erg/s/cm2/sr/A'

    Returns
    -------
    result : astropy.Quantity
        specInt converted to the desired units.
    '''
    if not isinstance(wavelength, units.Quantity):
        wavelength = wavelength << units.nm

    if not isinstance(specInt, units.Quantity):
        specInt = specInt << units.J / units.s / units.m**2 / units.sr / units.Hz

    return specInt.to(outUnits, equivalencies=units.spectral_density(wavelength))

class CrswIterator:
    '''
    Basic iterator to be used for controlling the scale of the collisional
    radiative switching (of Hummer & Voels) multiplicative paramter. Can be
    inherited to provide different behaviour. By default starts from a factor
    of 1e3 and scales this factor by 0.1**(1.0/value) each iteration, as is
    the default behaviour in RH.
    '''
    def __init__(self, initVal=1e3):
        self.val = initVal

    def __call__(self):
        self.val = max(1.0, self.val * 0.1**(1.0/self.val))
        return self.val

class UnityCrswIterator(CrswIterator):
    '''
    A specific case representing no collisional radiative switching (i.e.
    parameter always 1).
    '''
    def __init__(self):
        super().__init__(1.0)

    def __call__(self):
        return self.val

def sequence_repr(x: Sequence) -> str:
    '''
    Uniform representation of arrays and lists as lists for use in
    round-tripping AtomicModels.
    '''
    if isinstance(x, np.ndarray):
        return repr(x.tolist())

    return repr(x)

def view_flatten(x: np.ndarray) -> np.ndarray:
    '''
    Return a flattened view over an array, will raise an Exception if it
    cannot be represented as a flat array without copy.
    '''
    y = x.view()
    y.shape = (x.size,)
    return y
