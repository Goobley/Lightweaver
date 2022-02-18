import importlib
import os
from dataclasses import dataclass
from enum import Enum, auto
from os import path
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union

import numpy as np
from astropy import units
from numba import njit
from scipy import special
from scipy.integrate import trapezoid
from weno4 import weno4

import lightweaver.constants as C
from .simd_management import filter_usable_simd_impls

if TYPE_CHECKING:
    from .atomic_model import AtomicLine, AtomicModel

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

def get_code_location():
    '''
    Returns the directory containing the Lightweaver Python source.
    '''
    directory, _ = path.split(path.realpath(__file__))
    return directory

def get_data_path():
    '''
    Returns the location of the Lightweaver support data.
    '''
    return path.join(get_code_location(), 'Data') + path.sep

def get_default_molecule_path():
    '''
    Returns the location of the default molecules taken from RH.
    '''
    return path.join(get_code_location(), 'Data', 'DefaultMolecules') + path.sep

def filter_fs_iter_libs(libs: Sequence[str], exts: Sequence[str]) -> Sequence[str]:
    '''
    Filter a list of libraries (e.g. SimdImpl_{SimdType}.{pep3149}.so) with a
    valid collection of extensions. (As .so is a valid extension, we can't just
    check the end of the file name).
    '''
    result = []
    for libName in libs:
        libPrefix = libName.split('.')[0]
        for ext in exts:
            if libPrefix + ext == libName:
                result.append(libName)
    return result

def get_fs_iter_libs() -> Sequence[str]:
    '''
    Returns the paths of the default FsIterationScheme libraries usable on the
    current machine (due to available SIMD optimisations -- these are detected by NumPy).
    '''
    validExts = importlib.machinery.EXTENSION_SUFFIXES
    iterSchemesDir = path.join(get_code_location(), 'DefaultIterSchemes')
    schemes = [path.join(iterSchemesDir, x) for x in
                    filter_usable_simd_impls(
                        filter_fs_iter_libs(os.listdir(iterSchemesDir), validExts)
               )]
    return schemes

def vac_to_air(wavelength: np.ndarray) -> np.ndarray:
    '''
    Convert vacuum wavelength to air.

    Parameters
    ----------
    wavelength : float or array-like or astropy.Quantity
        If no units then the wavelength is assumed to be in [nm], otherwise the
        provided units are used.

    Returns
    -------
    result : float or array-like or astropy.Quantity
        The converted wavelength in [nm].
    '''
    # NOTE(cmo): Moved this import here as it's very slow
    ### HACK
    from specutils.utils.wcs_utils import vac_to_air as spec_vac_to_air
    return spec_vac_to_air(wavelength << units.nm, method='edlen1966').value

def air_to_vac(wavelength: np.ndarray) -> np.ndarray:
    '''
    Convert air wavelength to vacuum.

    Parameters
    ----------
    wavelength : float or array-like or astropy.Quantity
        If no units then the wavelength is assumed to be in [nm], otherwise the
        provided units are used.

    Returns
    -------
    result : float or array-like or astropy.Quantity
        The converted wavelength in [nm].
    '''
    # NOTE(cmo): Moved this import here as it's very slow
    ### HACK
    from specutils.utils.wcs_utils import air_to_vac as spec_air_to_vac
    return spec_air_to_vac(wavelength << units.nm, scheme='iteration',
                           method='edlen1966').value

def convert_specific_intensity(wavelength: np.ndarray,
                               specInt: np.ndarray,
                               outUnits) -> units.quantity.Quantity:
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

def check_shape_exception(a: np.ndarray, shape: Union[int, Tuple[int]],
                          ndim: Optional[int]=1, name: Optional[str]='array'):
    '''
    Ensure that an array matches the expected number of dimensions and shape.
    Raise a ValueError if not, quoting the array's name (if provided)

    Parameters
    ----------
    a : np.ndarray
        The array to verify.
    shape : int or Tuple[int]
        The length (for a 1D array), or shape for a multi-dimensional array.
    ndim : int, optional
        The expected number of dimensions (default: 1)
    name : str, optional
        The name to in any exception (default: array)

    '''
    if isinstance(shape, int):
        shape = (shape,)

    if a.ndim != ndim:
        raise ValueError(f'Array ({name}) does not have the expected number '
                         f'of dimensions: {ndim} (got: {a.ndim}).')

    if a.shape != shape:
        raise ValueError(f'Array ({name}) does not have the expected shape: '
                         f'{shape} (got: {a.shape}).')

def compute_radiative_losses(ctx) -> np.ndarray:
    '''
    Compute the radiative gains and losses for each wavelength in the grid
    used by the context. Units of J/s/m3/Hz. Includes
    background/contributions from overlapping lines. Convention of positive
    => radiative gain, negative => radiative loss.

    Parameters
    ----------
    ctx : Context
        A context with full depth-dependent data (i.e. ctx.depthData.fill =
        True set before the most recent formal solution).

    Returns
    -------
    loss : np.ndarray
        The radiative gains losses for each depth and wavelength in the
        simulation.
    '''
    atmos = ctx.kwargs['atmos']

    chiTot = ctx.depthData.chi
    S = (ctx.depthData.eta + (ctx.background.sca * ctx.spect.J)[:, None, None, :]) / chiTot
    Idepth = ctx.depthData.I
    loss = ((chiTot * (S - Idepth)) * 0.5).sum(axis=2).transpose(0, 2, 1) @ atmos.wmu

    return loss


def integrate_line_losses(ctx, loss : np.ndarray,
                          lines : Union['AtomicLine', Sequence['AtomicLine']],
                          extendGridNm: float=0.0) -> Union[Sequence[np.ndarray], np.ndarray]:
    '''
    Integrate the radiative gains and losses over the band associated with a
    line or list of lines. Units of J/s/m3. Includes background/contributions
    from overlapping lines. Convention of positive => radiative gain,
    negative => radiative loss.

    Parameters
    ----------
    ctx : Context
        A context with the full depth-dependent data (i.e. ctx.depthData.fill
        = True set before the most recent formal solution).
    loss : np.ndarray
        The radiative gains/losses for each wavelength and depth computed by
        `compute_radiative_losses`.
    lines : AtomicLine or list of AtomicLine
        The lines for which to compute losses.
    extendGridNm : float, optional
        Set this to a positive value to add an additional point at each end
        of the integration range to include a wider continuum/far-wing
        contribution. Units: nm, default: 0.0.

    Returns
    -------
    linesLosses : array or list of array
        The radiative gain/losses per line at each depth.
    '''
    from .atomic_model import AtomicLine

    if isinstance(lines, AtomicLine):
        lines = [lines]

    spect = ctx.kwargs['spect']

    lineLosses = []
    for line in lines:
        transId = line.transId
        grid = spect.transWavelengths[transId]
        blueIdx = spect.blueIdx[transId]
        blue = ctx.spect.wavelength[blueIdx]
        redIdx = blueIdx + grid.shape[0]
        red = ctx.spect.wavelength[redIdx-1]

        if extendGridNm != 0.0:
            wav = np.concatenate(((blue-extendGridNm,),
                                ctx.spect.wavelength[blueIdx:redIdx],
                                (red+extendGridNm,)))
        else:
            wav = ctx.spect.wavelength[blueIdx:redIdx]

        # NOTE(cmo): There's a sneaky transpose going on here for the integration
        lineLoss = np.zeros((loss.shape[1], wav.shape[0]))
        for k in range(loss.shape[1]):
            lineLoss[k, :] = weno4(wav, ctx.spect.wavelength, loss[:, k])
        lineLosses.append(trapezoid(lineLoss,
                                    (wav << units.nm).to(units.Hz,
                                                         equivalencies=units.spectral()).value)
                          )
    return lineLosses[0] if len(lineLosses) == 1 else lineLosses


def compute_contribution_fn(ctx, mu : int=-1, outgoing : bool=True) -> np.ndarray:
    '''
    Computes the contribution function for all wavelengths in the simulation,
    for a chosen angular index.

    Parameters
    ----------
    ctx : Context
        A context with the full depth-dependent data (i.e. ctx.depthData.fill
        = True set before the most recent formal solution).
    mu : Optional[int]
        The angular index to use (corresponding to the order of the angular
        quadratures in atmosphere), default: -1.
    outgoing : Optional[bool]
        Whether to compute the contribution for outgoing or incoming
        radiation (wrt to the atmosphere). Default: outgoing==True, i.e. to
        observer.

    Returns
    -------
    cfn : np.ndarray
        The contribution function in terms of depth and wavelength.
    '''
    upDown = 1 if outgoing else 0
    tau = np.zeros_like(ctx.depthData.chi[:, mu, upDown, :])
    chi = ctx.depthData.chi
    atmos = ctx.kwargs['atmos']

    # NOTE(cmo): Compute tau for all wavelengths
    tau[:, 0] = 1e-20
    for k in range(1, tau.shape[1]):
        tau[:, k] = tau[:, k-1] + 0.5 * (chi[:, mu, upDown, k] + chi[:, mu, upDown, k-1]) \
                                      * (atmos.height[k-1] - atmos.height[k])

    # NOTE(cmo): Source function.
    Sfn = ((ctx.depthData.eta
            + (ctx.background.sca * ctx.spect.J)[:, None, None, :])
           / chi)

    # NOTE(cmo): Contribution function for all wavelengths.
    cfn = ctx.depthData.chi[:, mu, upDown, :] / atmos.muz[mu] \
           * np.exp(-tau / atmos.muz[mu]) * Sfn[:, mu, upDown, :]

    return cfn


def compute_wavelength_edges(ctx) -> np.ndarray:
    '''
    Compute the edges of the wavelength bins associated with the wavelength
    array used in a simulation, typically used in conjunction with a plot
    using pcolormesh.

    Parameters
    ----------
    ctx : Context
        The context from which to construct the wavelength edges.

    Returns
    -------
    wlEdges : np.ndarray
        The edges of the wavelength bins.
    '''
    wav = ctx.spect.wavelength
    wlEdges = np.concatenate(((wav[0] - 0.5 * (wav[1] - wav[0]),),
                            0.5 * (wav[1:] + wav[:-1]),
                            (wav[-1] + 0.5 * (wav[-1] - wav[-2]),)
                            ))
    return wlEdges


def compute_height_edges(ctx) -> np.ndarray:
    '''
    Compute the edges of the height bins associated with the stratified
    altitude array used in a simulation, typically used in conjunction with a
    plot using pcolormesh.

    Parameters
    ----------
    ctx : Context
        The context from which to construct the height edges.

    Returns
    -------
    heightEdges : np.ndarray
        The edges of the height bins.
    '''
    atmos = ctx.kwargs['atmos']
    heightEdges = np.concatenate(((atmos.height[0] + 0.5 * (atmos.height[0] - atmos.height[1]),),
                                0.5 * (atmos.height[1:] + atmos.height[:-1]),
                                (atmos.height[-1] - 0.5 * (atmos.height[-2] - atmos.height[-1]),)))
    return heightEdges
