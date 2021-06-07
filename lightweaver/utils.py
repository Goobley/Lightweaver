import lightweaver.constants as C
from copy import copy, deepcopy
import numpy as np
import os
from typing import Union, Tuple, Sequence, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum, auto
from astropy import units
import astropy.units as u
from specutils.utils.wcs_utils import vac_to_air as spec_vac_to_air, air_to_vac as spec_air_to_vac
from numba import njit
from scipy import special
from weno4 import weno4
from scipy.integrate import trapezoid

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
    spect = ctx.kwargs['spect']
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
    atmos = ctx.kwargs['atmos']

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
                                    (wav << u.nm).to(u.Hz, equivalencies=u.spectral()).value))
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

def grotrian_diagram(atom : 'AtomicModel', ax=None, orbitalLabels=None,
                     vacuumWavelength=True, highlightLines=None, adjustTextPositions=False,
                     labelLevels=True, wavelengthLabels=True, continuumEdgeLabels=False,
                     fontSize=9, suppressLineLabels=None, rotateLabels=True):
    '''
    Produce a Grotrian (term) diagram for a model atom.

    Parameters
    ----------
    atom : AtomicModel
        The model atom for which to produce the diagram.
    ax : matplotlib.axis.Axes, optional
        The axes on which to create the plot.
    orbitalLabels : Optional[List[str]]
        The labels for the oribtals on the x-axis.
    vacuumWavelength : bool
        Whether to plot the wavelengths with their vacuum values, or convert
        to air. Default: True i.e. vacuum.
    highlightLines : Optional[List[int]]
        The indices of lines in atom.lines to highlight. These will be
        plotted in red, whereas the others will be plotted in green.
    adjustTextPositions: bool
        Whether to move the wavelength labels around to avoid the level
        labels and each other. Default: False.
    labelLevels: bool or List[str]
        Whether to label the levels (parsing the names for these from
        atom.levels[i].label if True), or uses the provided strings (one per
        level) if a List[str], or not label at all (False). Default: True.
    wavelengthLabels : bool
        Whether to label the wavelengths for each bound-bound transition.
        Default: True.
    continuumEdgeLabels : bool
        Label the continuum edge wavelengths (only if wavelengthLabels also True).
        Default: False.
    fontSize: float
        Font size to use for the annotations. Default: 9
    suppressLineLabels : Optional[List[int]]
        List of line indices to not print wavelength labels for. Default: None.
    rotateLabels : bool
        Whether to rotate the wavelength labels to lie along the line
        transitions, does not work well with `adjustTextPositions`. Default: True
    '''
    import matplotlib.pyplot as plt
    from copy import copy
    if adjustTextPositions:
        from adjustText import adjust_text

    if ax is None:
        ax = plt.gca()

    if highlightLines is None:
        highlightLines = []

    orbits = ['S', 'P', 'D', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'Q', 'R', 'T', 'U', 'V', 'W', 'X']

    levels = [copy(l) for l in atom.levels]
    levelTrueEEv = [level.E_eV for level in atom.levels]
    # NOTE(cmo): Just assume that it is, even if it has a defined L (e.g. for
    # polarisation reasons)
    # if atom.levels[-1].L is not None:
    #     raise ValueError('Last level should be ground term')

    ls = sorted(list(set([l.L for l in atom.levels[:-1]])))

    avoidLevelLabels = []
    # NOTE(cmo): Draw levels in
    MinLevelGap = 0.2
    for currentL in ls:
        currentEnergies = []
        for level in levels[:-1]:
            if level.L == currentL:
                e = level.E_eV
                if len(currentEnergies) != 0:
                    energyDiff = np.array([e - ee for ee in currentEnergies])
                    idx = np.argmin(np.abs(energyDiff))
                    if np.abs(energyDiff[idx]) < MinLevelGap:
                        e = currentEnergies[idx] + np.sign(energyDiff[idx]) * MinLevelGap
                ax.plot([level.L-0.25, level.L+0.25], [e, e], c='C0')
                currentEnergies.append(e)
                level.E = e * C.EV * C.CM_TO_M / C.HC



    # NOTE(cmo): Draw overlying cont
    ax.plot([min(ls)-0.25, max(ls)+0.25], [levels[-1].E_eV, levels[-1].E_eV], '--', c='C0')

    texts = []

    # NOTE(cmo): Draw b-f
    contPerCol = {L: 0 for L in ls}
    for cont in atom.continua:
        lu = levels[cont.j]
        ll = levels[cont.i]
        nc = contPerCol[ll.L]
        contPerCol[ll.L] += 1
        xLoc = ll.L + 0.05 * nc
        ax.annotate("", xy=(xLoc,lu.E_eV), xytext=(xLoc, ll.E_eV), arrowprops={'arrowstyle':'->','color':'y'})
        if wavelengthLabels and continuumEdgeLabels:
            textX = ll.L
            # textY = lu.E_eV - 0.05 * (ll.E_eV + lu.E_eV)
            textY = lu.E_eV - 0.5
            lambdaEdge = cont.lambdaEdge if vacuumWavelength else vac_to_air(cont.lambdaEdge)
            a = ax.annotate('%.2f' % lambdaEdge, xy=(textX, textY),
                            fontsize=fontSize, ha='center')
            texts.append(a)


    ax.relim()
    ax.autoscale_view()


    # NOTE(cmo): Draw b-b
    for idx, line in enumerate(atom.lines):
        lu = levels[line.j]
        ll = levels[line.i]
        lineColor = 'r' if idx in highlightLines else 'g'
        ax.annotate("", xy=(ll.L,ll.E_eV), xytext=(lu.L, lu.E_eV),arrowprops={'arrowstyle':'->','color': lineColor})
        if wavelengthLabels:
            if suppressLineLabels is not None and idx in suppressLineLabels:
                continue
            textX = 0.5 * (ll.L + lu.L)
            textY = 0.5 * (ll.E_eV + lu.E_eV)
            lambda0 = line.lambda0 if vacuumWavelength else vac_to_air(line.lambda0)
            if rotateLabels:
                angle = np.rad2deg(np.arctan2(lu.E_eV - ll.E_eV, lu.L - ll.L))
                if angle > 90:
                    angle -= 180.0
                a = ax.text(textX + 0.1 , textY - 0.1, '%.2f' % lambda0,# xy=(textX, textY),
                                fontsize=fontSize, rotation=angle, transform_rotates_text=True, ha='center', va='center')
            else:
                a = ax.annotate('%.2f' % lambda0, xy=(textX, textY),
                                fontsize=fontSize)
            texts.append(a)

    # NOTE(cmo): Label levels on top
    if labelLevels:
        if not isinstance(labelLevels, Sequence):
            labelsUsed = []
            for level in levels[:-1]:
                endIdx = [level.label.upper().rfind(x) for x in ['E', 'O']]
                maxIdx = max(endIdx)
                if maxIdx == -1:
                    raise ValueError("Unable to determine parity of level %s" % (repr(level)))
                label = level.label[:maxIdx+1].upper()
                words: List[str] = label.split()
                label = ' '.join(words[:-1])
                if any(label == l for l in labelsUsed):
                    continue
                labelsUsed.append(label)
                labelY = level.E_eV - 0.5 if level.E_eV != 0 else level.E_eV + 0.3
                labelX = level.L - 0.4 if level.E_eV != 0 else level.L - 0.25
                a = ax.annotate(label, xy=(labelX, labelY), color='r', fontsize=fontSize)
                avoidLevelLabels.append(a)
            a = ax.annotate(levels[-1].label, xy=(-0.25, levels[-1].E_eV + 0.04), color='b', fontsize=fontSize)
            # bbox={'boxstyle': 'round,pad=0.3', 'fc': 'w', 'alpha': 0.2})
            avoidLevelLabels.append(a)
        else:
            for i, level in enumerate(levels[:-1]):
                labelY = level.E_eV - 0.5 if level.E_eV != 0 else level.E_eV + 0.3
                labelX = level.L - 0.4 if level.E_eV != 0 else level.L - 0.25
                a = ax.annotate(labelLevels[i], xy=(labelX, labelY), fontsize=fontSize)
                avoidLevelLabels.append(a)
            a = ax.annotate(labelLevels[-1], xy=(-0.25, levels[-1].E_eV + 0.04), fontsize=fontSize, bbox={'boxstyle': 'round,pad=0.3', 'fc': 'w', 'alpha': 0.2})
            avoidLevelLabels.append(a)


    if adjustTextPositions and wavelengthLabels:
        adjust_text(texts, autoalign='xy', expand_text=(1.05, 1.05),
                    force_text=(0.05, 0.05), arrowprops={'arrowstyle': 'wedge'},
                    # bbox={'boxstyle': 'round,pad=0.3', 'fc': 'g', 'alpha': 0.2},
                    # avoid_self=False,
                    add_objects=avoidLevelLabels, lim=200 )

    if orbitalLabels is None:
        orbitalLabels = {}
        labelled = []
        for level in levels[:-1]:
            if level.L in labelled:
                continue
            labelled.append(level.L)
            label = level.label
            split = label.split()
            idx = len(split) - 1
            while idx >= 0:
                if len(split[idx]) > 1:
                    break
                idx -= 1

            orbitalLabels[level.L] = split[idx]
        orbitalLabels = [orbitalLabels[k] for k in sorted(orbitalLabels.keys())]
    if len(ls) != len(orbitalLabels):
        raise ValueError('Length of orbital labels does not match provided number of different Ls in the model atom')
    ax.set_xticks(ls)
    ax.set_xticklabels(orbitalLabels)

    ax.set_ylabel('Energy [eV]')
