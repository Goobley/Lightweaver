from dataclasses import dataclass, field
from enum import Enum, auto
from fractions import Fraction
from typing import List, Sequence, Optional, Any, Iterator, cast, TYPE_CHECKING
import re

import numpy as np
from weno4 import weno4
from parse import parse

import lightweaver.constants as Const
from .utils import gaunt_bf, sequence_repr
from .atomic_table import PeriodicTable, Element
from .barklem import Barklem
from .zeeman import compute_zeeman_components, ZeemanComponents

if TYPE_CHECKING:
    from .atmosphere import Atmosphere
    from .atomic_set import SpeciesStateTable
    from .collisional_rates import CollisionalRates
    from .broadening import LineBroadening


@dataclass
class AtomicModel:
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
        s = 'AtomicModel(element="%s",\n\tlevels=[\n' % repr(self.element)
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

    def __hash__(self):
        return hash(repr(self))

    def vBroad(self, atmos: Atmosphere) -> np.ndarray:
        vTherm = 2.0 * Const.KBoltzmann / (Const.Amu * PeriodicTable[self.element].mass)
        vBroad = np.sqrt(vTherm * atmos.temperature + atmos.vturb**2)
        return vBroad

    @property
    def transitions(self) -> Sequence['AtomicTransition']:
        return self.lines + self.continua # type: ignore

def reconfigure_atom(atom: AtomicModel):
    atom.__post_init__()

def element_sort(atom: AtomicModel):
    return atom.element

# TODO(cmo): Tidy up comparisons
def avoid_recursion_eq(a, b) -> bool:
    if isinstance(a, np.ndarray):
        if not np.all(a == b):
            return False
    elif isinstance(a, AtomicModel):
        if a.element != b.element:
            return False
        if len(a.levels) != len(b.levels):
            return False
        if len(a.lines) != len(b.lines):
            return False
        if len(a.continua) != len(b.continua):
            return False
        if len(a.collisions) != len(b.collisions):
            return False
    else:
        if a != b:
            return False
    return True


def model_component_eq(a, b) -> bool:
    if a is b:
        return True

    if type(a) is not type(b):
        if not (isinstance(a, AtomicTransition) and isinstance(b, AtomicTransition)):
            raise NotImplemented
        else:
            return False

    ignoreKeys = ['interpolator']
    da = a.__dict__
    db = b.__dict__

    return all([avoid_recursion_eq(da[k], db[k]) for k in da.keys() if k not in ignoreKeys])

@dataclass
class AtomicLevel:
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
        if not any([x is None for x in [self.J, self.L, self.S]]):
            if self.J <= self.L + self.S:
                self.lsCoupling = True

    def __hash__(self):
        return hash((self.E, self.g, self.label, self.stage, self.J, self.L, self.S))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AtomicLevel):
            return hash(self) == hash(other)
        return False

    @property
    def lsCoupling(self) -> bool:
        if all(x is not None for x in (self.J, self.L, self.S)):
            J = cast(Fraction, self.J)
            L = cast(int, self.L)
            S = cast(Fraction, self.S)
            if J <= L + S:
                return True
        return False

    @property
    def E_SI(self):
        return self.E * Const.HC / Const.CM_TO_M

    @property
    def E_eV(self):
        return self.E_SI / Const.EV

    def __repr__(self):
        s = 'AtomicLevel(E=%f, g=%f, label="%s", stage=%d, J=%s, L=%s, S=%s)' % (self.E, self.g, self.label, self.stage, repr(self.J), repr(self.L), repr(self.S))
        return s

class LineType(Enum):
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
    def setup(self, line: 'AtomicLine'):
        pass

    def doppler_units(self, line: 'AtomicLine') -> np.ndarray:
        raise NotImplementedError

    def wavelength(self, line: 'AtomicLine', vMicroChar: float=Const.VMICRO_CHAR) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError

@dataclass
class LinearCoreExpWings(LineQuadrature):
    """
    RH-Style line quadrature.
    """
    qCore: float
    qWing: float
    Nlambda: int
    beta: float = field(init=False)

    def __repr__(self):
        s = '%s(qCore=%.4e, qWing=%.4e, Nlambda=%d)' % (type(self).__name__,
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
        return model_component_eq(self, other)

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
        return (self.atom.element, self.i, self.j)


@dataclass(eq=False)
class AtomicLine(AtomicTransition):
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
        s = '%s(j=%d, i=%d, f=%e, type=%s, quadrature=%s, broadening=%s' % (
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
        return self.quadrature.wavelength(self, vMicroChar=vMicroChar)

    def zeeman_components(self) -> Optional[ZeemanComponents]:
        return compute_zeeman_components(self)

    @property
    def overlyingContinuumLevel(self) -> AtomicLevel:
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
        return self.lambda0_m / Const.NM_TO_M

    @property
    def lambda0_m(self) -> float:
        deltaE = self.jLevel.E_SI - self.iLevel.E_SI
        return Const.HC / deltaE

    @property
    def Aji(self) -> float:
        gRatio = self.iLevel.g / self.jLevel.g
        C: float = 2 * np.pi * (Const.QElectron / Const.Epsilon0) \
           * (Const.QElectron / Const.MElectron) / Const.CLight
        return C / self.lambda0_m**2 * gRatio * self.f

    @property
    def Bji(self) -> float:
        return self.lambda0_m**3 / (2.0 * Const.HC) * self.Aji

    @property
    def Bij(self) -> float:
        return self.jLevel.g / self.iLevel.g * self.Bji

    @property
    def polarisable(self) -> bool:
        return (self.iLevel.lsCoupling and self.jLevel.lsCoupling) or (self.gLandeEff is not None)


@dataclass(eq=False)
class VoigtLine(AtomicLine):

    # TODO(cmo): Provide old interface for now to avoid needing to change all of this right now?
    def damping(self, atmos: Atmosphere, eqPops: SpeciesStateTable):
        Qs = self.damping(atmos, eqPops)

        vBroad = self.atom.vBroad(atmos)
        cDop = self.lambda0_m / (4.0 * np.pi)
        aDamp = (Qs.Qinelast + Qs.Qelast) * cDop / vBroad
        return aDamp, Qs.Qelast

@dataclass(eq=False)
class AtomicContinuum(AtomicTransition):

    def setup(self, atom: AtomicModel):
        pass

    def __repr__(self):
        s = 'AtomicContinuum(j=%d, i=%d)' % (self.j, self.i)
        return s

    def __hash__(self):
        return hash(repr(self))

    def alpha(self, wavelength: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def wavelength(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def minLambda(self) -> float:
        raise NotImplementedError

    @property
    def lambda0(self) -> float:
        return self.lambda0_m / Const.NM_TO_M

    @property
    def lambdaEdge(self) -> float:
        return self.lambda0

    @property
    def lambda0_m(self) -> float:
        deltaE = self.jLevel.E_SI - self.iLevel.E_SI
        return Const.HC / deltaE

@dataclass(eq=False)
class ExplicitContinuum(AtomicContinuum):
    wavelengthGrid: Sequence[float]
    alphaGrid: Sequence[float]

    def setup(self, atom: AtomicModel):
        if self.j < self.i:
            self.i, self.j = self.j, self.i
        self.atom = atom
        self.wavelengthGrid = np.asarray(self.wavelengthGrid)
        if not np.all(np.diff(self.wavelength) > 0.0):
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
        alpha = weno4(wavelength, self.wavelengthGrid, self.alphaGrid, left=0.0, right=0.0)
        alpha[wavelength < self.minLambda] = 0.0
        alpha[wavelength > self.lambdaEdge] = 0.0
        alpha[alpha < 0.0] = 0.0
        return alpha

    def wavelength(self) -> np.ndarray:
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
        return self.wavelengthGrid[0]

@dataclass(eq=False)
class HydrogenicContinuum(AtomicContinuum):
    alpha0: float
    minLambda: float
    NlambdaGen: int

    def __repr__(self):
        s = 'HydrogenicContinuum(j=%d, i=%d, alpha0=%e, minLambda=%f, NlambdaGen=%d)' % (self.j, self.i, self.alpha0, self.minLambda, self.NlambdaGen)
        return s

    def setup(self, atom):
        if self.j < self.i:
            self.i, self.j = self.j, self.i
        self.atom = atom
        self.jLevel: AtomicLevel = atom.levels[self.j]
        self.iLevel: AtomicLevel = atom.levels[self.i]
        if self.minLambda >= self.lambda0:
            raise ValueError('Minimum wavelength is larger than continuum edge at %f [nm] in continuum %s' % (self.lambda0, repr(self)))

    def alpha(self, wavelength: np.ndarray) -> np.ndarray:
        Z = self.jLevel.stage
        nEff = Z * np.sqrt(Const.ERydberg / (self.jLevel.E_SI - self.iLevel.E_SI))
        gbf0 = gaunt_bf(self.lambda0, nEff, Z)
        gbf = gaunt_bf(wavelength, nEff, Z)
        alpha = self.alpha0 * gbf / gbf0 * (wavelength / self.lambda0)**3
        alpha[wavelength < self.minLambda] = 0.0
        alpha[wavelength > self.lambdaEdge] = 0.0
        return alpha

    def wavelength(self) -> np.ndarray:
        return np.linspace(self.minLambda, self.lambdaEdge, self.NlambdaGen)