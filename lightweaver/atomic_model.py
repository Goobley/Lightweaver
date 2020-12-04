from dataclasses import dataclass, field
from enum import Enum, auto
from fractions import Fraction
from typing import List, Tuple, Sequence, Optional, Any, Iterator, cast, TYPE_CHECKING, Callable
import re

import numpy as np
from weno4 import weno4
from parse import parse

import lightweaver.constants as Const
from .utils import gaunt_bf, sequence_repr
from .atomic_table import PeriodicTable, Element
from .barklem import Barklem
from .zeeman import compute_zeeman_components, ZeemanComponents
from .broadening import LineBroadening

if TYPE_CHECKING:
    from .atmosphere import Atmosphere
    from .atomic_set import SpeciesStateTable
    from .collisional_rates import CollisionalRates


@dataclass
class AtomicModel:
    '''
    Container class for the complete description of a model atom.

    Attributes
    ----------
    element : Element
        The element or ion represented by this model.
    levels : list of AtomicLevel
        The levels in use in this model.
    lines :  list of AtomicLine
        The atomic lines present in this model.
    continua : list of AtomicContinuum
        The atomic continua present in this model.
    collisions : list of CollisionalRates
        The collisional rates present in this model.
    '''
    element: Element
    levels: Sequence['AtomicLevel']
    lines: Sequence['AtomicLine']
    continua: Sequence['AtomicContinuum']
    collisions: Sequence['CollisionalRates']

    # @profile
    def __post_init__(self):
        for l in self.levels:
            l.setup(self)

        for l in self.lines:
            l.setup(self)

        for c in self.continua:
            c.setup(self)

        for c in self.collisions:
            c.setup(self)

    def __repr__(self):
        s = 'AtomicModel(element=%s,\n\tlevels=[\n' % repr(self.element)
        for l in self.levels:
            s += '\t\t' + repr(l) + ',\n'
        s += '\t],\n\tlines=[\n'
        for l in self.lines:
            s += '\t\t' + repr(l) + ',\n'
        s += '\t],\n\tcontinua=[\n'
        for c in self.continua:
            s += '\t\t' + repr(c) + ',\n'
        s += '\t],\n\tcollisions=[\n'
        for c in self.collisions:
            s += '\t\t' + repr(c) + ',\n'
        s += '])\n'
        return s

    # def __hash__(self):
    #     return hash(repr(self))

    def vBroad(self, atmos: 'Atmosphere') -> np.ndarray:
        '''
        Computes the atomic broadening velocity structure for a given
        atmosphere from the thermal motions and microturbulent velocity.
        '''
        vTherm = 2.0 * Const.KBoltzmann / (Const.Amu * PeriodicTable[self.element].mass)
        vBroad = np.sqrt(vTherm * atmos.temperature + atmos.vturb**2)
        return vBroad

    @property
    def transitions(self) -> Sequence['AtomicTransition']:
        '''
        List of all atomic transitions present on the model.
        '''
        return self.lines + self.continua # type: ignore

def reconfigure_atom(atom: AtomicModel):
    '''
    Re-perform all atomic set up after modifying parameters.
    '''
    atom.__post_init__()

def element_sort(atom: AtomicModel):
    return atom.element

@dataclass
class AtomicLevel:
    '''
    Description of atomic level in model atom.

    Attributes
    ----------
    E : float
        Energy above ground state [cm-1]
    g : float
        Statistical weight of level
    label : str
        Name for level
    stage : int
        Ionisation of level with 0 being neutral
    atom : AtomicModel
        AtomicModel that holds this level, will be initialised by the atom.
    J : Fraction, optional
        Total quantum angular momentum.
    L : int, optional
        Orbital angular momentum.
    S : Fraction, optional
        Spin.
    '''
    E: float
    g: float
    label: str
    stage: int
    atom: AtomicModel = field(init=False)
    J: Optional[Fraction] = None
    L: Optional[int] = None
    S: Optional[Fraction] = None

    def setup(self, atom):
        self.atom = atom

    def __hash__(self):
        return hash((self.E, self.g, self.label, self.stage, self.J, self.L, self.S))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AtomicLevel):
            return hash(self) == hash(other)
        return False

    @property
    def lsCoupling(self) -> bool:
        '''
        Returns whether the L-S coupling formalism can be applied to this
        level.
        '''
        if all(x is not None for x in (self.J, self.L, self.S)):
            J = cast(Fraction, self.J)
            L = cast(int, self.L)
            S = cast(Fraction, self.S)
            if J <= L + S:
                return True
        return False

    @property
    def E_SI(self):
        '''
        Returns E in Joule.
        '''
        return self.E * Const.HC / Const.CM_TO_M

    @property
    def E_eV(self):
        '''
        Returns E in electron volt.
        '''
        return self.E_SI / Const.EV

    def __repr__(self):
        s = 'AtomicLevel(E=%10.3f, g=%g, label="%s", stage=%d, J=%s, L=%s, S=%s)' % (self.E, self.g, self.label, self.stage, repr(self.J), repr(self.L), repr(self.S))
        return s

class LineType(Enum):
    '''
    Enum to show if the line should be treated in CRD or PRD.
    '''
    CRD = 0
    PRD = auto()

    def __repr__(self):
        if self == LineType.CRD:
            return 'LineType.CRD'
        elif self == LineType.PRD:
            return 'LineType.PRD'
        else:
            raise ValueError('Unknown LineType in LineType.__repr__')


@dataclass
class LineQuadrature:
    '''
    Describes the wavelength quadrature to be used for integrating properties
    associated with a line.
    '''
    def setup(self, line: 'AtomicLine'):
        pass

    def doppler_units(self, line: 'AtomicLine') -> np.ndarray:
        '''
        Return the quadrature in Doppler units.
        '''
        raise NotImplementedError

    def wavelength(self, line: 'AtomicLine', vMicroChar: float=Const.VMICRO_CHAR) -> np.ndarray:
        '''
        Return the quadrature in nm.
        '''
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError

@dataclass
class LinearCoreExpWings(LineQuadrature):
    """
    RH-Style line quadrature, with approximately linear core spacing and
    exponential wing spacing, by using a function of the form
    q(n) = a*(n + (exp(b*n)-1))
    with n \in [0, N) satisfying the following conditions:

     - q[0] = 0

     - q[(N-1)/2] = qcore

     - q[N-1] = qwing.

    """
    qCore: float
    qWing: float
    Nlambda: int
    beta: float = field(init=False)

    def __repr__(self):
        s = '%s(qCore=%g, qWing=%g, Nlambda=%d)' % (type(self).__name__,
             self.qCore, self.qWing, self.Nlambda)
        return s

    def __hash__(self):
        return hash((self.qCore, self.qWing, self.Nlambda))

    def setup(self, line: 'AtomicLine'):
        if self.qWing <= 2.0 * self.qCore:
            # Use linear scale to qWing
            print("Ratio of qWing / (2*qCore) <= 1\n Using linear spacing for transition %d->%d" % (line.j, line.i))
            self.beta = 1.0
        else:
            self.beta = self.qWing / (2.0 * self.qCore)

    def doppler_units(self, line: 'AtomicLine') -> np.ndarray:
        Nlambda = self.Nlambda // 2 if self.Nlambda % 2 == 1 else (self.Nlambda - 1) // 2
        Nlambda += 1
        beta = self.beta

        y = beta + np.sqrt(beta**2 + (beta - 1.0) * Nlambda + 2.0 - 3.0 * beta)
        b = 2.0 * np.log(y) / (Nlambda - 1)
        a = self.qWing / (Nlambda - 2.0 + y**2)
        nl = np.arange(Nlambda)
        q: np.ndarray = a * (nl + (np.exp(b * nl) - 1.0))

        NlambdaFull = 2 * Nlambda - 1
        result = np.zeros(NlambdaFull)
        Nmid = Nlambda - 1

        result[:Nmid][::-1] = -q[1:]
        result[Nmid+1:] = q[1:]
        return result

    def wavelength(self, line: 'AtomicLine', vMicroChar=Const.VMICRO_CHAR) -> np.ndarray:
        qToLambda = line.lambda0 * (vMicroChar / Const.CLight)
        result = self.doppler_units(line)
        result *= qToLambda
        result += line.lambda0
        return result


@dataclass
class AtomicTransition:
    '''
    Basic storage class for atomic transitions. Both lines and continua are
    derived from this.
    '''
    j: int
    i: int
    atom: AtomicModel = field(init=False)
    jLevel: AtomicLevel = field(init=False)
    iLevel: AtomicLevel = field(init=False)

    def setup(self, atom: AtomicModel):
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        if other is self:
            return True

        return repr(self) == repr(other)

    def wavelength(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def lambda0(self) -> float:
        raise NotImplementedError

    @property
    def lambda0_m(self) -> float:
        raise NotImplementedError

    @property
    def transId(self) -> Tuple[Element, int, int]:
        '''
        Unique identifier (transition ID) for transition (assuming one copy
        of each Element), used in creating a SpectrumConfiguration etc.
        '''
        return (self.atom.element, self.i, self.j)

@dataclass
class LineProfileState:
    '''
    Dataclass used to communicate line profile calculations from the backend
    to the frontend whilst allowing the backend to provide an overrideable
    optimised voigt implementation for the default case.

    Attributes
    ----------
    wavelength : np.ndarray
        Wavelengths at which to compute the line profile [nm]
    vlosMu : np.ndarray
        Bulk velocity projected onto each ray in the angular integration scheme [m/s] in an array of [Nmu, Nspace].
    atmos : Atmosphere
        The associated atmosphere.
    eqPops : SpeciesStateTable
        The associated populations for each species present in the simulation.
    default_voigt_callback : callable
        Computes the Voigt profile for the default case, takes the damping
        parameter aDamp and broadening velocity vBroad as arguments, and
        returns the line profile phi (in this case phi_num in the tech report).
    vBroad : np.ndarray, optional
        Cache to avoid recomputing vBroad every time. May be None.
    '''

    wavelength: np.ndarray
    vlosMu: np.ndarray
    atmos: 'Atmosphere'
    eqPops: 'SpeciesStateTable'
    default_voigt_callback: Callable[[np.ndarray, np.ndarray], np.ndarray]
    vBroad: Optional[np.ndarray]=None

@dataclass
class LineProfileResult:
    '''
    Dataclass for returning the line profile and associated data that needs
    to be saved (damping parameter and elastic collision rate) from the
    frontend to the backend.
    '''
    phi: np.ndarray
    aDamp: np.ndarray
    Qelast: np.ndarray


@dataclass(eq=False)
class AtomicLine(AtomicTransition):
    '''
    Base class for atomic lines, holding their specialised information over
    transitions.

    Attributes
    ----------
    f : float
        Oscillator strength.
    type : LineType
        Should the line be treated in PRD or CRD.
    quadrature : LineQuadrature
        Wavelength quadrature for integrating line properties over.
    broadening : LineBroadening
        Object describing the broadening processes to be used in conjunction
        with the quadrature to generate the line profile.
    gLandeEff : float, optional
        Optionally override LS-coupling (if available for this transition),
        and just directly set the effective Lande g factor (if it isn't).
    '''
    f: float
    type: LineType
    quadrature: LineQuadrature
    broadening: LineBroadening
    gLandeEff: Optional[float] = None

    def setup(self, atom: AtomicModel):
        self.atom = atom
        self.jLevel: AtomicLevel = self.atom.levels[self.j]
        self.iLevel: AtomicLevel = self.atom.levels[self.i]
        self.quadrature.setup(self)
        self.broadening.setup(self)

    def __repr__(self):
        s = '%s(j=%d, i=%d, f=%9.3e, type=%s, quadrature=%s, broadening=%s' % (
                type(self).__name__,
                self.j, self.i, self.f, repr(self.type),
                repr(self.quadrature), repr(self.broadening))
        if self.gLandeEff is not None:
            s += ', gLandeEff=%e' % self.gLandeEff
        s += ')'
        return s

    def __hash__(self):
        return hash(repr(self))

    def wavelength(self, vMicroChar=Const.VMICRO_CHAR) -> np.ndarray:
        '''
        Returns the wavelength grid for this transition based on the
        LineQuadrature.

        Parameters
        ----------
        vMicroChar : float, optional
            Characterisitc microturbulent velocity to assume when computing
            the line quadrature (default 3e3 m/s).
        '''
        return self.quadrature.wavelength(self, vMicroChar=vMicroChar)

    def zeeman_components(self) -> Optional[ZeemanComponents]:
        '''
        Returns the Zeeman components of a line, if possible or None.
        '''
        return compute_zeeman_components(self)

    def compute_phi(self, state: LineProfileState) -> LineProfileResult:
        '''
        Compute the line profile, intended to be called from the backend.
        '''
        raise NotImplementedError

    @property
    def overlyingContinuumLevel(self) -> AtomicLevel:
        '''
        Find the first overlying continuum level.
        '''
        Z = self.jLevel.stage + 1
        j = self.j
        ic = j + 1
        try:
            while self.atom.levels[ic].stage < Z:
                ic += 1
            cont = self.atom.levels[ic]
            return cont
        except:
            raise ValueError('No overlying continuum level found for line %s' % repr(self))

    @property
    def lambda0(self) -> float:
        '''
        Return the line rest wavelength [nm].
        '''
        return self.lambda0_m / Const.NM_TO_M

    @property
    def lambda0_m(self) -> float:
        '''
        Return the line rest wavelength [m].
        '''
        deltaE = self.jLevel.E_SI - self.iLevel.E_SI
        return Const.HC / deltaE

    @property
    def Aji(self) -> float:
        '''
        Return the Einstein A coefficient for this line.
        '''
        gRatio = self.iLevel.g / self.jLevel.g
        C: float = 2 * np.pi * (Const.QElectron / Const.Epsilon0) \
           * (Const.QElectron / Const.MElectron) / Const.CLight
        return C / self.lambda0_m**2 * gRatio * self.f

    @property
    def Bji(self) -> float:
        '''
        Return the Einstein B_{ji} coefficient for this line.
        '''
        return self.lambda0_m**3 / (2.0 * Const.HC) * self.Aji

    @property
    def Bij(self) -> float:
        '''
        Return the Einstein B_{ij} coefficient for this line.
        '''
        return self.jLevel.g / self.iLevel.g * self.Bji

    @property
    def polarisable(self) -> bool:
        '''
        Return whether sufficient information is available to compute full
        Stokes solutions for this line.
        '''
        return (self.iLevel.lsCoupling and self.jLevel.lsCoupling) or (self.gLandeEff is not None)


@dataclass(eq=False, repr=False)
class VoigtLine(AtomicLine):
    '''
    Specialised line profile for the default case of a Voigt profile.
    '''

    def damping(self, atmos: 'Atmosphere', eqPops: 'SpeciesStateTable',
                vBroad: Optional[np.ndarray]=None):
        '''
        Computes the damping parameter and elastic collision rate.

        Parameters
        ----------
        atmos : Atmosphere
            The atmosphere to consider.
        eqPops : SpeciesStateTable
            The populations in this atmosphere.
        vBroad : np.ndarray, optional
            The broadening velocity, will be used if passed, or computed
            using atom.vBroad if not.

        Returns
        -------
        aDamp : np.ndarray
            The Voigt damping parameter.
        Qelast : np.ndarray
            The rate of elastic collisions broadening the line -- needed for PRD.
        '''
        Qs = self.broadening.broaden(atmos, eqPops)

        if vBroad is None:
            vBroad = self.atom.vBroad(atmos)

        cDop = self.lambda0_m / (4.0 * np.pi)
        aDamp = (Qs.natural + Qs.Qelast) * cDop / vBroad
        return aDamp, Qs.Qelast

    def compute_phi(self, state: LineProfileState) -> LineProfileResult:
        '''
        Computes the line profile.

        In the case of a VoigtLine the line profile simply uses the
        default_voigt_callback from the backend.

        Parameters
        ----------
        state : LineProfileState
            The information from the backend

        Returns
        -------
        result : LineProfileResult
            The line profile, as well as the damping parameter 'a' and and
            the broadening velocity.
        '''
        vBroad = self.atom.vBroad(state.atmos) if state.vBroad is None else state.vBroad
        aDamp, Qelast = self.damping(state.atmos, state.eqPops, vBroad=vBroad)
        cb = state.default_voigt_callback
        # NOTE(cmo): This is affected by mypy #5485, so we ignore typing for now
        phi = state.default_voigt_callback(aDamp, vBroad) # type: ignore

        return LineProfileResult(phi=phi, aDamp=aDamp, Qelast=Qelast)

@dataclass(eq=False)
class AtomicContinuum(AtomicTransition):
    '''
    Base class for atomic continua.
    '''

    def setup(self, atom: AtomicModel):
        pass

    def __repr__(self):
        s = 'AtomicContinuum(j=%d, i=%d)' % (self.j, self.i)
        return s

    def __hash__(self):
        return hash(repr(self))

    def alpha(self, wavelength: np.ndarray) -> np.ndarray:
        '''
        Returns the cross-section as a function of wavelength

        Parameters
        ----------
        wavelength : np.ndarray
            The wavelengths at which to compute the cross-section

        Returns
        -------
        alpha : np.ndarray
            The cross-section for each wavelength
        '''
        raise NotImplementedError

    def wavelength(self) -> np.ndarray:
        '''
        The wavelength grid on which this continuum's cross section is defined.
        '''
        raise NotImplementedError

    @property
    def minLambda(self) -> float:
        '''
        The minimum wavelength at which this transition contributes.
        '''
        raise NotImplementedError

    @property
    def lambda0(self) -> float:
        '''
        The maximum (edge) wavelength at which this transition contributes [nm].
        '''
        return self.lambda0_m / Const.NM_TO_M

    @property
    def lambdaEdge(self) -> float:
        '''
        The maximum (edge) wavelength at which this transition contributes [nm].
        '''
        return self.lambda0

    @property
    def lambda0_m(self) -> float:
        '''
        The maximum (edge) wavelength at which this transition contributes [m].
        '''
        deltaE = self.jLevel.E_SI - self.iLevel.E_SI
        return Const.HC / deltaE

@dataclass(eq=False)
class ExplicitContinuum(AtomicContinuum):
    '''
    Specific version of atomic continuum with tabulated cross-section against
    wavelength. Interpolated using weno4.
    Attributes
    ----------
    wavelengthGrid : list of float
        Wavelengths at which cross-section is tabulated [nm].
    alphaGrid : list of float
        Tabulated cross-sections [m2].
    '''
    wavelengthGrid: Sequence[float]
    alphaGrid: Sequence[float]

    def setup(self, atom: AtomicModel):
        if self.j < self.i:
            self.i, self.j = self.j, self.i
        self.atom = atom
        self.wavelengthGrid = np.asarray(self.wavelengthGrid)
        if not np.all(np.diff(self.wavelengthGrid) > 0.0):
            raise ValueError('Wavelength array not monotonically increasing in continuum %s' % repr(self))
        self.alphaGrid = np.asarray(self.alphaGrid)
        self.jLevel = atom.levels[self.j]
        self.iLevel = atom.levels[self.i]
        if self.lambdaEdge > self.wavelengthGrid[-1]:
            wav = np.concatenate((self.wavelengthGrid, np.array([self.lambdaEdge])))
            self.wavelengthGrid = wav
            self.alphaGrid = np.concatenate((self.alphaGrid, np.array([self.alphaGrid[-1]])))

    def __repr__(self):
        s = 'ExplicitContinuum(j=%d, i=%d, wavelengthGrid=%s, alphaGrid=%s)' % (self.j, self.i,
        sequence_repr(self.wavelengthGrid), sequence_repr(self.alphaGrid))
        return s

    def alpha(self, wavelength: np.ndarray) -> np.ndarray:
        '''
        Computes cross-section as a function of wavelength.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelengths at which to compute the cross-section [nm].

        Returns
        -------
        alpha : np.ndarray
            Cross-section at associated wavelength.
        '''
        alpha = weno4(wavelength, self.wavelengthGrid, self.alphaGrid, left=0.0, right=0.0)
        alpha[wavelength < self.minLambda] = 0.0
        alpha[wavelength > self.lambdaEdge] = 0.0
        alpha[alpha < 0.0] = 0.0
        return alpha

    def wavelength(self) -> np.ndarray:
        '''
        Returns the wavelength grid at which this transition needs to be
        computed to be correctly integrated. Specific handling is added to
        ensure that it is treated properly close to the edge.
        '''
        grid = cast(np.ndarray, self.wavelengthGrid)
        edge = self.lambdaEdge
        result = np.copy(grid[(grid >= self.minLambda) & (grid <= edge)])
        # NOTE(cmo): If the last value before the edge is more than 0.1 nm away
        # then put the edge in.
        if edge - grid[-1] > 0.1:
            result = np.concatenate((result, (edge,)))
        return result

    @property
    def minLambda(self) -> float:
        '''
        The minimum wavelength at which this transition contributes.
        '''
        return self.wavelengthGrid[0]

@dataclass(eq=False)
class HydrogenicContinuum(AtomicContinuum):
    '''
    Specific case of a Hydrogenic continuum, approximately falling off as
    1/nu**3 towards higher frequencies (additional effects from Gaunt
    factor).

    Attributes
    ----------
    NlambaGen : int
        The number of points to generate for the wavelength grid.
    alpha0 : float
        The cross-section at the edge wavelength [m2].
    minWavelength : float
        The minimum wavelength below which this transition is assumed to no
        longer contribute [nm].
    '''
    NlambdaGen: int
    alpha0: float
    minWavelength: float

    def __repr__(self):
        s = 'HydrogenicContinuum(j=%d, i=%d, NlambdaGen=%d, alpha0=%g, minWavelength=%g)' % (self.j, self.i, self.NlambdaGen, self.alpha0, self.minWavelength)
        return s

    def setup(self, atom):
        if self.j < self.i:
            self.i, self.j = self.j, self.i
        self.atom = atom
        self.jLevel: AtomicLevel = atom.levels[self.j]
        self.iLevel: AtomicLevel = atom.levels[self.i]
        if self.minLambda >= self.lambda0:
            raise ValueError('Minimum wavelength is larger than continuum edge at %g [nm] in continuum %s' % (self.lambda0, repr(self)))

    def alpha(self, wavelength: np.ndarray) -> np.ndarray:
        '''
        Computes cross-section as a function of wavelength.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelengths at which to compute the cross-section [nm].

        Returns
        -------
        alpha : np.ndarray
            Cross-section at associated wavelength.
        '''
        Z = self.jLevel.stage
        nEff = Z * np.sqrt(Const.ERydberg / (self.jLevel.E_SI - self.iLevel.E_SI))
        gbf0 = gaunt_bf(self.lambda0, nEff, Z)
        gbf = gaunt_bf(wavelength, nEff, Z)
        alpha = self.alpha0 * gbf / gbf0 * (wavelength / self.lambda0)**3
        alpha[wavelength < self.minLambda] = 0.0
        alpha[wavelength > self.lambdaEdge] = 0.0
        return alpha

    def wavelength(self) -> np.ndarray:
        '''
        Returns the wavelength grid at which this transition needs to be
        computed to be correctly integrated.
        '''
        return np.linspace(self.minLambda, self.lambdaEdge, self.NlambdaGen)

    @property
    def minLambda(self) -> float:
        '''
        The minimum wavelength at which this transition contributes.
        '''
        return self.minWavelength