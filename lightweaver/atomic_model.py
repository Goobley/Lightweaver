from dataclasses import dataclass, field
from enum import Enum, auto
from fractions import Fraction
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Tuple, cast

import numpy as np
from weno4 import weno4

import lightweaver.constants as Const
from lightweaver.constants import VMICRO_CHAR

from .atomic_table import Element, PeriodicTable
from .broadening import LineBroadening
from .deprecation import deprecated_names
from .utils import gaunt_bf, sequence_repr
from .zeeman import ZeemanComponents, compute_zeeman_components

if TYPE_CHECKING:
    from .atmosphere import Atmosphere
    from .atomic_set import SpeciesStateTable
    from .collisional_rates import CollisionalRates


@deprecated_names
@dataclass
class AtomicModel:
    """
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
    """

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

    def v_broad(self, atmos: 'Atmosphere') -> np.ndarray:
        """
        Computes the atomic broadening velocity structure for a given
        atmosphere from the thermal motions and microturbulent velocity.
        """
        vTherm = 2.0 * Const.KBoltzmann / (Const.Amu * PeriodicTable[self.element].mass)
        v_broad = np.sqrt(vTherm * atmos.temperature + atmos.vturb**2)
        return v_broad

    @property
    def transitions(self) -> Sequence['AtomicTransition']:
        """
        List of all atomic transitions present on the model.
        """
        return self.lines + self.continua  # type: ignore


def reconfigure_atom(atom: AtomicModel):
    """
    Re-perform all atomic set up after modifying parameters.
    """
    atom.__post_init__()


def element_sort(atom: AtomicModel):
    return atom.element


@deprecated_names
@dataclass
class AtomicLevel:
    """
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
    """

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
    def LS_coupling(self) -> bool:
        """
        Returns whether the L-S coupling formalism can be applied to this
        level.
        """
        if all(x is not None for x in (self.J, self.L, self.S)):
            J = cast(Fraction, self.J)
            L = cast(int, self.L)
            S = cast(Fraction, self.S)
            if J <= L + S:
                return True
        return False

    @property
    def E_SI(self):
        """
        Returns E in Joule.
        """
        return self.E * Const.HC_CM

    @property
    def E_eV(self):
        """
        Returns E in electron volt.
        """
        return self.E_SI / Const.EV

    def __repr__(self):
        s = ('AtomicLevel(E=%10.3f, g=%g, label="%s", stage=%d, J=%s, L=%s, S=%s)') % (
            self.E,
            self.g,
            self.label,
            self.stage,
            repr(self.J),
            repr(self.L),
            repr(self.S),
        )
        return s


class LineType(Enum):
    """
    Enum to show if the line should be treated in CRD or PRD.
    """

    CRD = 0
    PRD = auto()

    def __repr__(self):
        if self == LineType.CRD:
            return 'LineType.CRD'
        elif self == LineType.PRD:
            return 'LineType.PRD'
        else:
            raise ValueError('Unknown LineType in LineType.__repr__')


@deprecated_names
@dataclass
class LineQuadrature:
    """
    Describes the wavelength quadrature to be used for integrating properties
    associated with a line.
    """

    def setup(self, line: 'AtomicLine'):
        pass

    def doppler_units(self, line: 'AtomicLine') -> np.ndarray:
        """
        Return the quadrature in Doppler units.
        """
        raise NotImplementedError

    def wavelength(self, line: 'AtomicLine', v_micro_char: float = Const.VMICRO_CHAR) -> np.ndarray:
        """
        Return the quadrature in nm.
        """
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError


@deprecated_names
@dataclass
class LinearQuadrature(LineQuadrature):
    """
    Simple linearly spaced wavelength grid. Primarily provided for CRTAF
    interaction.

    Nlambda : int
        The number of wavelength points in the wavelength grid (typically odd).
    delta_lambda : int
        The half-width of the grid (i.e. from core to one edge) [nm].
    """

    Nlambda: int
    delta_lambda: float

    def __repr__(self):
        s = '%s(Nlambda=%d, delta_lambda=%g)' % (
            type(self).__name__,
            self.Nlambda,
            self.delta_lambda,
        )
        return s

    def wavelength(self, line: 'AtomicLine', v_micro_char: float = Const.VMICRO_CHAR) -> np.ndarray:
        return np.linspace(
            line.lambda0 - self.delta_lambda, line.lambda0 + self.delta_lambda, self.Nlambda
        )

    def doppler_units(self, line: 'AtomicLine') -> np.ndarray:
        wavelength_grid = self.wavelength(line)
        v_micro_char = VMICRO_CHAR
        qToLambda = line.lambda0 * (v_micro_char / Const.CLight)
        return (wavelength_grid - line.lambda0) / qToLambda


@deprecated_names
@dataclass
class TabulatedQuadrature(LineQuadrature):
    """
    Tabulated wavelength quadrature. Primarily provided for CRTAF interaction.

    wavelength_grid : Sequence[float]
        The wavelength sample points [nm].
    """

    wavelength_grid: Sequence[float]

    def __repr__(self):
        s = '%s(wavelength_grid=%s)' % (type(self).__name__, sequence_repr(self.wavelength_grid))
        return s

    def wavelength(self, line: 'AtomicLine', v_micro_char: float = Const.VMICRO_CHAR) -> np.ndarray:
        return np.ascontiguousarray(self.wavelength_grid) + line.lambda0

    def doppler_units(self, line: 'AtomicLine') -> np.ndarray:
        wavelength_grid = self.wavelength(line)
        v_micro_char = VMICRO_CHAR
        qToLambda = line.lambda0 * (v_micro_char / Const.CLight)
        return (wavelength_grid - line.lambda0) / qToLambda


@deprecated_names
@dataclass
class LinearCoreExpWings(LineQuadrature):
    """
    RH-Style line quadrature, with approximately linear core spacing and
    exponential wing spacing, by using a function of the form
    q(n) = a*(n + (exp(b*n)-1))
    with n in [0, N) satisfying the following conditions:

     - q[0] = 0

     - q[(N-1)/2] = qcore

     - q[N-1] = qwing.

    If q_wing <= 2 * q_core, linear grid spacing will be used for this transition.
    """

    q_core: float
    q_wing: float
    Nlambda: int
    beta: float = field(init=False)

    def __repr__(self):
        s = '%s(q_core=%g, q_wing=%g, Nlambda=%d)' % (
            type(self).__name__,
            self.q_core,
            self.q_wing,
            self.Nlambda,
        )
        return s

    def __hash__(self):
        return hash((self.q_core, self.q_wing, self.Nlambda))

    def setup(self, line: 'AtomicLine'):
        if self.q_wing <= 2.0 * self.q_core:
            # Use linear scale to q_wing
            self.beta = 1.0
        else:
            self.beta = self.q_wing / (2.0 * self.q_core)

    def doppler_units(self, line: 'AtomicLine') -> np.ndarray:
        Nlambda = self.Nlambda // 2 if self.Nlambda % 2 == 1 else (self.Nlambda - 1) // 2
        Nlambda += 1
        beta = self.beta

        y = beta + np.sqrt(beta**2 + (beta - 1.0) * Nlambda + 2.0 - 3.0 * beta)
        b = 2.0 * np.log(y) / (Nlambda - 1)
        a = self.q_wing / (Nlambda - 2.0 + y**2)
        nl = np.arange(Nlambda)
        q: np.ndarray = a * (nl + (np.exp(b * nl) - 1.0))

        NlambdaFull = 2 * Nlambda - 1
        result = np.zeros(NlambdaFull)
        Nmid = Nlambda - 1

        result[:Nmid][::-1] = -q[1:]
        result[Nmid + 1 :] = q[1:]
        return result

    def wavelength(self, line: 'AtomicLine', v_micro_char=Const.VMICRO_CHAR) -> np.ndarray:
        qToLambda = line.lambda0 * (v_micro_char / Const.CLight)
        result = self.doppler_units(line)
        result *= qToLambda
        result += line.lambda0
        return result


@deprecated_names
@dataclass
class AtomicTransition:
    """
    Basic storage class for atomic transitions. Both lines and continua are
    derived from this.
    """

    j: int
    i: int
    atom: AtomicModel = field(init=False)
    j_level: AtomicLevel = field(init=False)
    i_level: AtomicLevel = field(init=False)

    def setup(self, atom: AtomicModel):
        if self.j < self.i:
            self.i, self.j = self.j, self.i
        self.atom = atom
        self.j_level: AtomicLevel = self.atom.levels[self.j]
        self.i_level: AtomicLevel = self.atom.levels[self.i]

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
    def trans_id(self) -> Tuple[Element, int, int]:
        """
        Unique identifier (transition ID) for transition (assuming one copy
        of each Element), used in creating a SpectrumConfiguration etc.
        """
        return (self.atom.element, self.i, self.j)


@deprecated_names
@dataclass
class LineProfileState:
    """
    Dataclass used to communicate line profile calculations from the backend
    to the frontend whilst allowing the backend to provide an overrideable
    optimised voigt implementation for the default case.

    Attributes
    ----------
    wavelength : np.ndarray
        Wavelengths at which to compute the line profile [nm]
    vlos_mu : np.ndarray
        Bulk velocity projected onto each ray in the angular integration scheme
        [m/s] in an array of [Nmu, Nspace].
    atmos : Atmosphere
        The associated atmosphere.
    eq_pops : SpeciesStateTable
        The associated populations for each species present in the simulation.
    default_voigt_callback : callable
        Computes the Voigt profile for the default case, takes the damping
        parameter a_damp and broadening velocity v_broad as arguments, and
        returns the line profile phi (in this case phi_num in the tech report).
    v_broad : np.ndarray, optional
        Cache to avoid recomputing v_broad every time. May be None.
    """

    wavelength: np.ndarray
    vlos_mu: np.ndarray
    atmos: 'Atmosphere'
    eq_pops: 'SpeciesStateTable'
    default_voigt_callback: Callable[[np.ndarray, np.ndarray], np.ndarray]
    v_broad: Optional[np.ndarray] = None


@deprecated_names
@dataclass
class LineProfileResult:
    """
    Dataclass for returning the line profile and associated data that needs
    to be saved (damping parameter and elastic collision rate) from the
    frontend to the backend.
    """

    phi: np.ndarray
    a_damp: np.ndarray
    Qelast: np.ndarray


@deprecated_names
@dataclass(eq=False)
class AtomicLine(AtomicTransition):
    """
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
    g_lande_eff : float, optional
        Optionally override LS-coupling (if available for this transition),
        and just directly set the effective Lande g factor (if it isn't).
    """

    f: float
    type: LineType
    quadrature: LineQuadrature
    broadening: LineBroadening
    g_lande_eff: Optional[float] = None

    def setup(self, atom: AtomicModel):
        super().setup(atom)
        self.quadrature.setup(self)
        self.broadening.setup(self)

    def __repr__(self):
        s = '%s(j=%d, i=%d, f=%9.3e, type=%s, quadrature=%s, broadening=%s' % (
            type(self).__name__,
            self.j,
            self.i,
            self.f,
            repr(self.type),
            repr(self.quadrature),
            repr(self.broadening),
        )
        if self.g_lande_eff is not None:
            s += ', g_lande_eff=%e' % self.g_lande_eff
        s += ')'
        return s

    def __hash__(self):
        return hash(repr(self))

    def wavelength(self, v_micro_char=Const.VMICRO_CHAR) -> np.ndarray:
        """
        Returns the wavelength grid for this transition based on the
        LineQuadrature.

        Parameters
        ----------
        v_micro_char : float, optional
            Characterisitc microturbulent velocity to assume when computing
            the line quadrature (default 3e3 m/s).
        """
        return self.quadrature.wavelength(self, v_micro_char=v_micro_char)

    def zeeman_components(self) -> Optional[ZeemanComponents]:
        """
        Returns the Zeeman components of a line, if possible or None.
        """
        return compute_zeeman_components(self)

    def compute_phi(self, state: LineProfileState) -> LineProfileResult:
        """
        Compute the line profile, intended to be called from the backend.
        """
        raise NotImplementedError

    @property
    def overlying_continuum_level(self) -> AtomicLevel:
        """
        Find the first overlying continuum level.
        """
        Z = self.j_level.stage + 1
        j = self.j
        ic = j + 1
        try:
            while self.atom.levels[ic].stage < Z:
                ic += 1
            cont = self.atom.levels[ic]
            return cont
        except IndexError:
            raise ValueError('No overlying continuum level found for line %s' % repr(self))

    @property
    def lambda0(self) -> float:
        """
        Return the line rest wavelength [nm].
        """
        return self.lambda0_m / Const.NM_TO_M

    @property
    def lambda0_m(self) -> float:
        """
        Return the line rest wavelength [m].
        """
        deltaE = self.j_level.E_SI - self.i_level.E_SI
        return Const.HC / deltaE

    @property
    def Aji(self) -> float:
        """
        Return the Einstein A coefficient for this line.
        """
        gRatio = self.i_level.g / self.j_level.g
        C: float = (
            2
            * np.pi
            * (Const.QElectron / Const.Epsilon0)
            * (Const.QElectron / Const.MElectron)
            / Const.CLight
        )
        return C / self.lambda0_m**2 * gRatio * self.f

    @property
    def Bji(self) -> float:
        """
        Return the Einstein B_{ji} coefficient for this line.
        """
        return self.lambda0_m**3 / (2.0 * Const.HC) * self.Aji

    @property
    def Bij(self) -> float:
        """
        Return the Einstein B_{ij} coefficient for this line.
        """
        return self.j_level.g / self.i_level.g * self.Bji

    @property
    def polarisable(self) -> bool:
        """
        Return whether sufficient information is available to compute full
        Stokes solutions for this line.
        """
        return (self.i_level.LS_coupling and self.j_level.LS_coupling) or (
            self.g_lande_eff is not None
        )


@deprecated_names
@dataclass(eq=False, repr=False)
class VoigtLine(AtomicLine):
    """
    Specialised line profile for the default case of a Voigt profile.
    """

    def damping(
        self,
        atmos: 'Atmosphere',
        eq_pops: 'SpeciesStateTable',
        v_broad: Optional[np.ndarray] = None,
    ):
        """
        Computes the damping parameter and elastic collision rate.

        Parameters
        ----------
        atmos : Atmosphere
            The atmosphere to consider.
        eq_pops : SpeciesStateTable
            The populations in this atmosphere.
        v_broad : np.ndarray, optional
            The broadening velocity, will be used if passed, or computed
            using atom.v_broad if not.

        Returns
        -------
        a_damp : np.ndarray
            The Voigt damping parameter.
        Qelast : np.ndarray
            The rate of elastic collisions broadening the line -- needed for PRD.
        """
        Qs = self.broadening.broaden(atmos, eq_pops)

        if v_broad is None:
            v_broad = self.atom.v_broad(atmos)

        cDop = self.lambda0_m / (4.0 * np.pi)
        a_damp = (Qs.natural + Qs.Qelast) * cDop / v_broad
        return a_damp, Qs.Qelast

    def compute_phi(self, state: LineProfileState) -> LineProfileResult:
        """
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
        """
        v_broad = self.atom.v_broad(state.atmos) if state.v_broad is None else state.v_broad
        a_damp, Qelast = self.damping(state.atmos, state.eq_pops, v_broad=v_broad)
        # NOTE(cmo): This is affected by mypy #5485, so we ignore typing for now
        phi = state.default_voigt_callback(a_damp, v_broad)  # type: ignore

        return LineProfileResult(phi=phi, a_damp=a_damp, Qelast=Qelast)


@deprecated_names
@dataclass(eq=False)
class AtomicContinuum(AtomicTransition):
    """
    Base class for atomic continua.
    """

    def setup(self, atom: AtomicModel):
        super().setup(atom)

    def __repr__(self):
        s = 'AtomicContinuum(j=%d, i=%d)' % (self.j, self.i)
        return s

    def __hash__(self):
        return hash(repr(self))

    def alpha(self, wavelength: np.ndarray) -> np.ndarray:
        """
        Returns the cross-section as a function of wavelength

        Parameters
        ----------
        wavelength : np.ndarray
            The wavelengths at which to compute the cross-section

        Returns
        -------
        alpha : np.ndarray
            The cross-section for each wavelength
        """
        raise NotImplementedError

    def wavelength(self) -> np.ndarray:
        """
        The wavelength grid on which this continuum's cross section is defined.
        """
        raise NotImplementedError

    @property
    def min_lambda(self) -> float:
        """
        The minimum wavelength at which this transition contributes.
        """
        raise NotImplementedError

    @property
    def lambda0(self) -> float:
        """
        The maximum (edge) wavelength at which this transition contributes [nm].
        """
        return self.lambda0_m / Const.NM_TO_M

    @property
    def lambda_edge(self) -> float:
        """
        The maximum (edge) wavelength at which this transition contributes [nm].
        """
        return self.lambda0

    @property
    def lambda0_m(self) -> float:
        """
        The maximum (edge) wavelength at which this transition contributes [m].
        """
        deltaE = self.j_level.E_SI - self.i_level.E_SI
        return Const.HC / deltaE

    @property
    def polarisable(self) -> bool:
        """
        Returns whether this continuum is polarisable, always False.
        """
        return False


@deprecated_names
@dataclass(eq=False)
class ExplicitContinuum(AtomicContinuum):
    """
    Specific version of atomic continuum with tabulated cross-section against
    wavelength. Interpolated using weno4.
    Attributes
    ----------
    wavelength_grid : list of float
        Wavelengths at which cross-section is tabulated [nm].
    alpha_grid : list of float
        Tabulated cross-sections [m2].
    """

    wavelength_grid: Sequence[float]
    alpha_grid: Sequence[float]

    def setup(self, atom: AtomicModel):
        super().setup(atom)
        self.wavelength_grid = np.asarray(self.wavelength_grid)  # type: ignore
        if not np.all(np.diff(self.wavelength_grid) > 0.0):
            raise ValueError(
                ('Wavelength array not monotonically increasing in continuum %s') % repr(self)
            )
        self.alpha_grid = np.asarray(self.alpha_grid)  # type: ignore
        if self.lambda_edge - self.wavelength_grid[-1] > 0.01:
            wav = np.concatenate((self.wavelength_grid, np.array([self.lambda_edge])))
            self.wavelength_grid = wav
            self.alpha_grid = np.concatenate((self.alpha_grid, np.array([self.alpha_grid[-1]])))

    def __repr__(self):
        s = 'ExplicitContinuum(j=%d, i=%d, wavelength_grid=%s, alpha_grid=%s)' % (
            self.j,
            self.i,
            sequence_repr(self.wavelength_grid),
            sequence_repr(self.alpha_grid),
        )
        return s

    def alpha(self, wavelength: np.ndarray) -> np.ndarray:
        """
        Computes cross-section as a function of wavelength.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelengths at which to compute the cross-section [nm].

        Returns
        -------
        alpha : np.ndarray
            Cross-section at associated wavelength.
        """
        alpha = weno4(wavelength, self.wavelength_grid, self.alpha_grid, left=0.0, right=0.0)
        alpha[wavelength < self.min_lambda] = 0.0
        alpha[wavelength > self.lambda_edge] = 0.0
        if np.any(alpha < 0.0):
            # NOTE(cmo): If weno4 has exploded to the extent that there are negatives, something has gone very wrong (e.g. overly sampled verticals in the cross-section resonances), so switch to linear interpolation.
            alpha = np.interp(
                wavelength, self.wavelength_grid, self.alpha_grid, left=0.0, right=0.0
            )
            alpha[wavelength < self.min_lambda] = 0.0
            alpha[wavelength > self.lambda_edge] = 0.0
            alpha[alpha < 0.0] = 0.0
        return alpha

    def wavelength(self) -> np.ndarray:
        """
        Returns the wavelength grid at which this transition needs to be
        computed to be correctly integrated. Specific handling is added to
        ensure that it is treated properly close to the edge.
        """
        grid = cast(np.ndarray, self.wavelength_grid)
        edge = self.lambda_edge
        result = np.copy(grid[(grid >= self.min_lambda) & (grid <= edge)])
        # NOTE(cmo): If the last value before the edge is more than 0.1 nm away
        # then put the edge in.
        if edge - result[-1] > 0.1:
            result = np.concatenate((result, (edge,)))
        return result

    @property
    def min_lambda(self) -> float:
        """
        The minimum wavelength at which this transition contributes.
        """
        return self.wavelength_grid[0]


@deprecated_names
@dataclass(eq=False)
class HydrogenicContinuum(AtomicContinuum):
    """
    Specific case of a Hydrogenic continuum, approximately falling off as
    1/nu**3 towards higher frequencies (additional effects from Gaunt
    factor).

    Attributes
    ----------
    NlambaGen : int
        The number of points to generate for the wavelength grid.
    alpha0 : float
        The cross-section at the edge wavelength [m2].
    min_wavelength : float
        The minimum wavelength below which this transition is assumed to no
        longer contribute [nm].
    """

    Nlambda_gen: int
    alpha0: float
    min_wavelength: float

    def __repr__(self):
        s = ('HydrogenicContinuum(j=%d, i=%d, Nlambda_gen=%d, alpha0=%g, min_wavelength=%g)') % (
            self.j,
            self.i,
            self.Nlambda_gen,
            self.alpha0,
            self.min_wavelength,
        )
        return s

    def setup(self, atom):
        super().setup(atom)
        if self.min_lambda >= self.lambda0:
            raise ValueError(
                ('Minimum wavelength is larger than continuum edge at %g [nm] in continuum %s')
                % (self.lambda0, repr(self))
            )

    def alpha(self, wavelength: np.ndarray) -> np.ndarray:
        """
        Computes cross-section as a function of wavelength.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelengths at which to compute the cross-section [nm].

        Returns
        -------
        alpha : np.ndarray
            Cross-section at associated wavelength.
        """
        Z = self.j_level.stage
        n_eff = Z * np.sqrt(Const.ERydberg / (self.j_level.E_SI - self.i_level.E_SI))
        gbf0 = gaunt_bf(self.lambda0, n_eff, Z)
        gbf = gaunt_bf(wavelength, n_eff, Z)
        alpha = self.alpha0 * gbf / gbf0 * (wavelength / self.lambda0) ** 3
        alpha[wavelength < self.min_lambda] = 0.0
        alpha[wavelength > self.lambda_edge] = 0.0
        return alpha

    def wavelength(self) -> np.ndarray:
        """
        Returns the wavelength grid at which this transition needs to be
        computed to be correctly integrated.
        """
        return np.linspace(self.min_lambda, self.lambda_edge, self.Nlambda_gen)

    @property
    def min_lambda(self) -> float:
        """
        The minimum wavelength at which this transition contributes.
        """
        return self.min_wavelength
