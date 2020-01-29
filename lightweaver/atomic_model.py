from dataclasses import dataclass, field
from enum import Enum, auto
from fractions import Fraction
from typing import List, Sequence, Optional, Any, Iterator, cast
import re

import numpy as np
from scipy.interpolate import interp1d
from parse import parse

import lightweaver.constants as Const
from .utils import gaunt_bf
from .atomic_table import AtomicTable, get_global_atomic_table
from .barklem import Barklem

class VdwBarklemIncompatible(Exception):
    pass


@dataclass
class AtomicModel:
    name: str
    levels: Sequence['AtomicLevel']
    lines: Sequence['AtomicLine']
    continua: Sequence['AtomicContinuum']
    collisions: Sequence['CollisionalRates']
    atomicTable: AtomicTable = field(default_factory=get_global_atomic_table)

    # @profile
    def __post_init__(self):
        for l in self.levels:
            l.setup(self)

        for l in self.lines:
            l.setup(self)
        
        # This is separate because all of the lines in 
        # an atom need to be initialised first
        for l in self.lines:
            l.setup_wavelength()

        for c in self.continua:
            c.setup(self)

        for c in self.collisions:
            c.setup(self)

    def __repr__(self):
        s = 'AtomicModel(name="%s",\n\tlevels=[\n' % self.name
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

    def replace_atomic_table(self, table: AtomicTable):
        self.atomicTable = table
        self.__post_init__()

    def vBroad(self, atmos):
        vTherm = 2.0 * Const.KBoltzmann / (Const.Amu * self.atomicTable[self.name].weight)
        vBroad = np.sqrt(vTherm * atmos.temperature + atmos.vturb**2)
        return vBroad

def reconfigure_atom(atom: AtomicModel):
    atom.__post_init__()

def avoid_recursion_eq(a, b) -> bool:
    if isinstance(a, np.ndarray):
        if not np.all(a == b):
            return False
    elif isinstance(a, AtomicModel):
        if a.name != b.name:
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
    lsCoupling: bool = False
    atom: AtomicModel = field(init=False)
    J: Optional[Fraction] = None
    L: Optional[int] = None
    S: Optional[Fraction] = None

    def setup(self, atom):
        self.atom = atom
        if not any([x is None for x in [self.J, self.L, self.S]]):
            if self.J <= self.L + self.S:
                self.lsCoupling = True
            
    def __eq__(self, other: object) -> bool:
        return model_component_eq(self, other)

    @property
    def E_SI(self):
        return self.E * Const.HC / Const.CM_TO_M

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
class VdwApprox:
    vals: Sequence[float]

    def setup(self, line: 'AtomicLine', table: AtomicTable):
        pass

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VdwApprox):
            return False

        return (self.vals == other.vals) and (type(self) is type(other))

@dataclass(eq=False)
class VdwUnsold(VdwApprox):
    def setup(self, line: 'AtomicLine', table: AtomicTable):
        self.line = line
        if len(self.vals) != 2:
            raise ValueError('VdwUnsold expects 2 coefficients (%s)' % repr(line))

        Z = line.jLevel.stage + 1
        j = line.j
        ic = j + 1
        while line.atom.levels[ic].stage < Z:
            ic += 1
        cont = line.atom.levels[ic]

        deltaR = (Const.ERydberg / (cont.E_SI - line.jLevel.E_SI))**2 \
                 - (Const.ERydberg / (cont.E_SI - line.iLevel.E_SI))**2
        fourPiEps0 = 4.0 * np.pi * Const.Epsilon0
        C625 = (2.5 * Const.QElectron**2 / fourPiEps0 * Const.ABarH / fourPiEps0 \
                   * 2 * np.pi * (Z * Const.RBohr)**2 / Const.HPlanck * deltaR)**0.4

        name = line.atom.name

        vRel35He = (8.0 * Const.KBoltzmann / (np.pi*Const.Amu * table[name].weight)\
                    * (1.0 + table[name].weight / table['He'].weight))**0.3
        vRel35H = (8.0 * Const.KBoltzmann / (np.pi*Const.Amu * table[name].weight)\
                    * (1.0 + table[name].weight / table['H'].weight))**0.3

        heAbund = table['He'].abundance
        self.cross = 8.08 * (self.vals[0] * vRel35H \
                             + self.vals[1] * heAbund * vRel35He) * C625

    def broaden(self, temperature, nHGround, broad):
        broad[:] = self.cross * temperature**0.3 * nHGround 


@dataclass(eq=False)
class VdwRidderRensbergen(VdwApprox):
    # NOTE(cmo): RidderRensbergen actually uses all 4 VdW coeffs
    def setup(self, line: 'AtomicLine', table: AtomicTable):
        self.line = line
        if len(self.vals) != 4:
            raise ValueError('VdwRidderRensbergen expects 4 coefficients (%s)' % (repr(line)))

        name = line.atom.name
        self.gammaCorrH = 1e-8 * Const.CM_TO_M**3 \
                        * (1.0 + table['H'].weight / table[name].weight)**self.vals[1]
        self.gammaCorrHe = 1e-9 * Const.CM_TO_M**3 \
                       * (1.0 + table['He'].weight / table[name].weight)**self.vals[3]
        self.HeAbund = table['He'].abundance

    def broaden(self, temperature, nHGround, broad):
        broad[:] = self.gammaCorrH * self.vals[0] * temperature**self.vals[1] \
                + self.gammaCorrHe * self.vals[2] * temperature**self.vals[3] \
                    * self.HeAbund
        broad *= nHGround

@dataclass(eq=False)
class VdwBarklem(VdwApprox):
    # NOTE(cmo): Since Helium is treated as per Unsold, only 3 vals are used
    def setup(self, line: 'AtomicLine', table: AtomicTable):
        self.line = line
        if len(self.vals) != 2:
            raise ValueError('VdwBarklem expects 2 coefficients (%s)' % (repr(line)))
        self.barklem = Barklem(table)
        try:
            newVals = self.barklem.get_active_cross_section(line.atom, line)
        except:
            raise VdwBarklemIncompatible
        
        self.vals = newVals

        Z = line.jLevel.stage + 1
        j = line.j
        ic = j + 1
        while line.atom.levels[ic].stage < Z:
            ic += 1
        cont = line.atom.levels[ic]

        deltaR = (Const.ERydberg / (cont.E_SI - line.jLevel.E_SI))**2 \
                 - (Const.ERydberg / (cont.E_SI - line.iLevel.E_SI))**2
        fourPiEps0 = 4.0 * np.pi * Const.Epsilon0
        C625 = (2.5 * Const.QElectron**2 / fourPiEps0 * Const.ABarH / fourPiEps0 \
                   * 2 * np.pi * (Z * Const.RBohr)**2 / Const.HPlanck * deltaR)**0.4

        name = line.atom.name

        vRel35He = (8.0 * Const.KBoltzmann / (np.pi*Const.Amu * table[name].weight)\
                    * (1.0 + table[name].weight / table['He'].weight))**0.3

        heAbund = table['He'].abundance
        self.cross = 8.08 * self.vals[2] * heAbund * vRel35He * C625

    def broaden(self, temperature, nHGround, broad):
        broad[:] = self.vals[0] * temperature**(0.5*(1.0-self.vals[1])) \
                    + self.cross * temperature**0.3
        broad *= nHGround
        
@dataclass
class AtomicTransition:
    def __eq__(self, other: object) -> bool:
        return model_component_eq(self, other)
    pass
    
@dataclass(eq=False)
class AtomicLine(AtomicTransition):
    j: int
    i: int
    f: float
    type: LineType
    NlambdaGen: int
    qCore: float
    qWing: float
    vdw: VdwApprox
    gRad: float
    stark: float
    gLandeEff: Optional[float] = None
    atom: AtomicModel = field(init=False)
    jLevel: AtomicLevel = field(init=False)
    iLevel: AtomicLevel = field(init=False)
    wavelength: np.ndarray = field(init=False)

    def __repr__(self):
        s = 'AtomicLine(j=%d, i=%d, f=%e, type=%s, NlambdaGen=%d, qCore=%f, qWing=%f, vdw=%s, gRad=%e, stark=%f' % (
                self.j, self.i, self.f, repr(self.type), self.NlambdaGen, self.qCore, self.qWing, repr(self.vdw), 
                self.gRad, self.stark)
        if self.gLandeEff is not None:
            s += ', gLandeEff=%f' % self.gLandeEff
        s += ')'
        return s

    def __hash__(self):
        return hash(repr(self))

    @property
    def Nlambda(self):
        return self.wavelength.shape[0]

def fraction_range(start: Fraction, stop: Fraction, step: Fraction=Fraction(1,1)) -> Iterator[Fraction]:
    while start < stop:
        yield start
        start += step

@dataclass
class ZeemanComponents:
    alpha: np.ndarray
    strength: np.ndarray
    shift: np.ndarray

@dataclass(eq=False)
class VoigtLine(AtomicLine):
    def setup(self, atom):
        if self.j < self.i:
            self.i, self.j = self.j, self.i

        self.atom: AtomicModel = atom
        self.jLevel: AtomicLevel = self.atom.levels[self.j]
        self.iLevel: AtomicLevel = self.atom.levels[self.i]

        try:
            self.vdw.setup(self, atom.atomicTable)
        except VdwBarklemIncompatible:
            print("Unable to treat line %d->%d of atom %s with Barklem broadening, using Unsold." % (self.j, self.i, self.atom.name))
            vals = self.vdw.vals
            self.vdw = VdwUnsold(vals)
            self.vdw.setup(self, atom.atomicTable)

    def __repr__(self):
        s = 'VoigtLine(j=%d, i=%d, f=%e, type=%s, NlambdaGen=%d, qCore=%f, qWing=%f, vdw=%s, gRad=%e, stark=%f' % (
                self.j, self.i, self.f, repr(self.type), self.NlambdaGen, self.qCore, self.qWing, repr(self.vdw), 
                self.gRad, self.stark)
        if self.gLandeEff is not None:
            s += ', gLandeEff=%f' % self.gLandeEff
        s += ')'
        return s

    def linear_stark_broaden(self, atmos):
        # We don't need to read n from the label, we can use the fact that for H -- which is the only atom we have here, g=2n^2
        nUpper = int(np.round(np.sqrt(0.5*self.jLevel.g)))
        nLower = int(np.round(np.sqrt(0.5*self.iLevel.g)))

        a1 = 0.642 if nUpper - nLower == 1 else 1.0
        C = a1 * 0.6 * (nUpper**2 - nLower**2) * Const.CM_TO_M**2
        GStark = C * atmos.ne**(2.0/3.0)
        return GStark

    def stark_broaden(self, atmos):
        if self.stark > 0.0:
            weight = self.atom.atomicTable[self.atom.name].weight
            C = 8.0 * Const.KBoltzmann / (np.pi * Const.Amu * weight)
            Cm = (1.0 + weight / (Const.MElectron / Const.Amu))**(1.0/6.0)
            # NOTE(cmo): 28.0 is average atomic weight
            Cm += (1.0 + weight / (28.0))**(1.0/6.0)

            Z = self.iLevel.stage + 1
            ic = self.i + 1
            while self.atom.levels[ic].stage < Z and ic < len(self.atom.levels):
                ic += 1
            if self.atom.levels[ic].stage == self.iLevel.stage:
                raise ValueError('Cant find overlying cont: %s' % repr(self))

            E_Ryd = Const.ERydberg / (1.0 + Const.MElectron / (weight * Const.Amu))
            neff_l = Z * np.sqrt(E_Ryd / (self.atom.levels[ic].E_SI - self.iLevel.E_SI))
            neff_u = Z * np.sqrt(E_Ryd / (self.atom.levels[ic].E_SI - self.jLevel.E_SI))

            C4 = Const.QElectron**2 / (4.0 * np.pi * Const.Epsilon0) \
                * Const.RBohr \
                * (2.0 * np.pi * Const.RBohr**2 / Const.HPlanck) / (18.0 * Z**4) \
                * ((neff_u * (5.0 * neff_u**2 + 1.0))**2 \
                    - (neff_l * (5.0 * neff_l**2 + 1.0))**2)
            cStark23 = 11.37 * (self.stark * C4)**(2.0/3.0)

            vRel = (C * atmos.temperature)**(1.0/6.0) * Cm
            stark = cStark23 * vRel * atmos.ne
        elif self.stark < 0.0:
            stark = np.abs(self.stark) * atmos.ne
        else:
            stark = np.zeros(atmos.Nspace)

        if self.atom.name.upper().strip() == 'H':
            stark += self.linear_stark_broaden(atmos)
        return stark

    def setup_wavelength(self):
        # self.xrd = []

        # # This is just a hack to keep the search happy
        # self.wavelength = np.zeros(1)
        # # NOTE(cmo): This is currently disabled -- we're not gonna do XRD anyway
        # if self.type == LineType.PRD and False:

        #     thisIdx = self.atom.lines.index(self)

        #     for i, l in enumerate(self.atom.lines):
        #         if l.type == LineType.PRD and l.j == self.j and l.i != self.i:
        #             self.xrd.append(l)

        #             if i > thisIdx:
        #                 waveratio = self.xrd[-1].lambda0 / self.lambda0
        #                 self.xrd[-1].qWing = waveratio * self.qWing

        #     if len(self.xrd) > 0:
        #         print("Found %d subordinate PRD lines, for line %d->%d of atom %s" % \
        #             (len(self.xrd), self.j, self.i, self.atom.name))

        # Compute default lambda grid
        Nlambda = self.NlambdaGen // 2 if self.NlambdaGen % 2 == 1 else (self.NlambdaGen - 1) // 2
        Nlambda += 1
        # Nlambda = self.Nlambda // 2 if self.Nlambda % 2 == 1 else (self.Nlambda + 1) // 2

        if self.qWing <= 2.0 * self.qCore:
            # Use linear scale to qWing
            print("Ratio of qWing / (2*qCore) <= 1\n Using linear spacing for transition %d->%d" % (self.j, self.i))
            beta = 1.0
        else:
            beta = self.qWing / (2.0 * self.qCore)

        y = beta + np.sqrt(beta**2 + (beta - 1.0) * Nlambda + 2.0 - 3.0 * beta)
        b = 2.0 * np.log(y) / (Nlambda - 1)
        a = self.qWing / (Nlambda - 2.0 + y**2)
        self.a = a
        self.b = b
        self.y = y
        nl = np.arange(Nlambda)
        self.q: np.ndarray = a * (nl + (np.exp(b * nl) - 1.0))

        qToLambda = self.lambda0 * (Const.VMICRO_CHAR / Const.CLight)
        NlambdaFull = 2 * Nlambda - 1
        line = np.zeros(NlambdaFull)
        Nmid = Nlambda - 1

        line[Nmid] = self.lambda0
        line[:Nmid][::-1] = self.lambda0 - qToLambda * self.q[1:]
        line[Nmid+1:] = self.lambda0 + qToLambda * self.q[1:]
        self.wavelength = line

    def zeeman_components(self) -> Optional[ZeemanComponents]:
        # Just do basic anomalous Zeeman splitting
        if self.gLandeEff is not None:
            alpha = np.array([-1, 0, 1], dtype=np.int32)
            strength = np.ones(3)
            shift = alpha * self.gLandeEff
            return ZeemanComponents(alpha, strength, shift)
        
        # Do LS coupling
        if self.iLevel.lsCoupling and self.jLevel.lsCoupling:
            # Mypy... you're a pain sometimes... (even if you are technically correct)
            Jl = cast(Fraction, self.iLevel.J)
            Ll = cast(int, self.iLevel.L)
            Sl = cast(Fraction, self.iLevel.S)
            Ju = cast(Fraction, self.jLevel.J)
            Lu = cast(int, self.jLevel.L)
            Su = cast(Fraction, self.jLevel.S)

            gLl = lande_factor(Jl, Ll, Sl)
            gLu = lande_factor(Ju, Lu, Su)
            alpha = []
            strength = []
            shift = []
            norm = np.zeros(3)
            
            for ml in fraction_range(-Jl, Jl+1):
                for mu in fraction_range(-Ju, Ju+1):
                    if abs(ml - mu) <= 1.0:
                        alpha.append(int(ml - mu))
                        shift.append(gLl*ml - gLu*mu)
                        strength.append(zeeman_strength(Ju, mu, Jl, ml))
                        norm[alpha[-1]+1] += strength[-1]
            alpha = np.array(alpha, dtype=np.int32)
            strength = np.array(strength)
            shift = np.array(shift)
            strength /= norm[alpha + 1]

            return ZeemanComponents(alpha, strength, shift)

        return None



    def polarised_wavelength(self, bChar: Optional[float]=None) -> Sequence[float]:
        ## NOTE(cmo): bChar in TESLA
        if any([not self.iLevel.lsCoupling, not self.jLevel.lsCoupling]) or \
           (self.gLandeEff is None):
            print("Can't treat line %d->%d with polarization" % (self.j, self.i))
            return self.wavelength

        if bChar is None:
            bChar = Const.B_CHAR
        # /* --- When magnetic field is present account for denser
        #        wavelength spacing in the unshifted \pi component, and the
        #        red and blue-shifted circularly polarized components.
        #        First, get characteristic Zeeman splitting -- ------------ */
        gLandeEff = effective_lande(self)
        qBChar = gLandeEff * (Const.QElectron / (4.0 * np.pi * Const.MElectron)) * \
                  (self.lambda0_m) * bChar / Const.VMICRO_CHAR

        if 0.5 * qBChar >= self.qCore:
            print("Characteristic Zeeman splitting qBChar (=%f) >= 2*qCore for transition %d->%d" % (qBChar, self.j, self.i))

        Nlambda = self.q.shape[0]
        NB = np.searchsorted(self.q, 0.5 * qBChar)
        qBShift = 2 * self.q[NB]
        qB = np.zeros(self.q.shape[0] + 2 * NB)
        qB[:Nlambda] = self.q

        nl = np.arange(NB+1, 2*NB+1)
        qB[NB+1:2*NB+1] = qBShift - self.a * (2*NB - nl + \
                                        (np.exp(self.b * (2 * NB - nl)) -1.0))
        nl = np.arange(2*NB+1, Nlambda+2*NB)
        qB[2*NB+1:] = qBShift + self.a * (nl - 2*NB  + \
                                     (np.exp(self.b * (nl - 2 * NB)) - 1.0))
        line = np.zeros(2 * qB.shape[0] - 1)
        Nmid = qB.shape[0] - 1
        qToLambda = self.lambda0 * (Const.VMICRO_CHAR / Const.CLight)
        line[Nmid] = self.lambda0
        line[:Nmid][::-1] = self.lambda0 - qToLambda * qB[1:]
        line[Nmid+1:] = self.lambda0 + qToLambda * qB[1:]
        return line

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

    def damping(self, atmos, vBroad, hGround):
        aDamp = np.zeros(atmos.Nspace)
        Qelast = np.zeros(atmos.Nspace)

        self.vdw.broaden(atmos.temperature, hGround, aDamp)
        Qelast += aDamp

        Qelast += self.stark_broaden(atmos)

        cDop = self.lambda0_m / (4.0 * np.pi)
        aDamp = (self.gRad + Qelast) * cDop / vBroad
        return aDamp, Qelast

def zeeman_strength(Ju: Fraction, Mu: Fraction, Jl: Fraction, Ml: Fraction) -> float:
    alpha  = int(Ml - Mu)
    dJ = int(Ju - Jl)

    # These parameters are x2 those in del Toro Iniesta (p. 137), but we normalise after the fact, so it's fine

    if dJ == 0: # jMin = ju = jl
        if alpha == 0: # pi trainsitions
            s = 2.0 * Mu**2
        elif alpha == -1: # sigma_b transitions
            s = (Ju + Mu) * (Ju - Mu + 1.0)
        elif alpha == 1: # sigma_r transitions
            s = (Ju - Mu) * (Ju + Mu + 1.0)
    elif dJ == 1: # jMin = jl, Mi = Ml
        if alpha == 0: # pi trainsitions
            s = 2.0 * ((Jl + 1)**2 - Ml**2)
        elif alpha == -1: # sigma_b transitions
            s = (Jl + Ml + 1) * (Jl + Ml + 2.0)
        elif alpha == 1: # sigma_r transitions
            s = (Jl - Ml + 1.0) * (Jl - Ml + 2.0)
    elif dJ == -1: # jMin = ju, Mi = Mu
        if alpha == 0: # pi trainsitions
            s = 2.0 * ((Ju + 1)**2 - Mu**2)
        elif alpha == -1: # sigma_b transitions
            s = (Ju - Mu + 1) * (Ju - Mu + 2.0)
        elif alpha == 1: # sigma_r transitions
            s = (Ju + Mu + 1.0) * (Ju + Mu + 2.0)
    else:
        raise ValueError('Invalid dJ: %d' % dJ)

    return float(s)

def lande_factor(J: Fraction, L: int, S: Fraction) -> float:
    if J == 0.0:
        return 0.0
    return float(1.5 + (S * (S + 1.0) - L * (L + 1)) / (2.0 * J * (J + 1.0)))

def effective_lande(line: AtomicLine):
    if line.gLandeEff is not None:
        return line.gLandeEff

    i = line.iLevel
    j = line.jLevel
    if any(x is None for x in [i.J, i.L, i.S, j.J, j.L, j.S]):
        raise ValueError('Cannot compute gLandeEff as gLandeEff not set and some of J, L and S None for line %s'%repr(line))
    gL = lande_factor(i.J, i.L, i.S) # type: ignore
    gU = lande_factor(j.J, j.L, j.S) # type: ignore

    return 0.5 * (gU + gL) + \
           0.25 * (gU - gL) * (j.J * (j.J + 1.0) - i.J * (i.J + 1.0)) # type: ignore


@dataclass(eq=False)
class AtomicContinuum(AtomicTransition):
    j: int
    i: int
    atom: AtomicModel = field(init=False)
    jLevel: AtomicLevel = field(init=False)
    iLevel: AtomicLevel = field(init=False)
    wavelength: np.ndarray = field(init=False)
    alpha: np.ndarray = field(init=False)

    def __repr__(self):
        s = 'AtomicContinuum(j=%d, i=%d)' % (self.j, self.i)
        return s

    def compute_alpha(self, wavelength) -> np.ndarray:
        pass

    def __hash__(self):
        return hash(repr(self))

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
    alphaGrid: Sequence[Sequence[float]]

    def setup(self, atom):
        if self.j < self.i:
            self.i, self.j = self.j, self.i
        self.atom = atom
        lambdaAlpha = np.array(self.alphaGrid).T
        self.wavelength = np.copy(lambdaAlpha[0, ::-1])
        if not np.all(np.diff(self.wavelength) > 0.0):
            raise ValueError('Wavelength array not monotonically increasing in continuum %s' % repr(self))
        self.alpha = np.copy(lambdaAlpha[1, ::-1])
        self.jLevel: AtomicLevel = atom.levels[self.j]
        self.iLevel: AtomicLevel = atom.levels[self.i]
        if self.lambdaEdge > self.wavelength[-1]:
            wav = np.concatenate((self.wavelength, np.array([self.lambdaEdge])))
            self.wavelength = wav
            self.alpha = np.concatenate((self.alpha, np.array([self.alpha[-1]])))

    def __repr__(self):
        s = 'ExplicitContinuum(j=%d, i=%d, alphaGrid=%s)' % (self.j, self.i, repr(self.alphaGrid))
        return s

    def compute_alpha(self, wavelength) -> np.ndarray:
        alpha = interp1d(self.wavelength, self.alpha, kind=3, bounds_error=False, fill_value=0.0)(wavelength)
        alpha[wavelength < self.minLambda] = 0.0
        alpha[wavelength > self.lambdaEdge] = 0.0
        if np.any(alpha < 0.0):
            alpha = interp1d(self.wavelength, self.alpha, bounds_error=False, fill_value=0.0)(wavelength)
        return alpha

    @property
    def Nlambda(self) -> int:
        return self.wavelength.shape[0]

    @property
    def lambda0(self) -> float:
        return self.lambda0_m / Const.NM_TO_M

    @property
    def lambdaEdge(self) -> float:
        return self.lambda0

    @property
    def minLambda(self) -> float:
        return self.wavelength[0]

    @property
    def lambda0_m(self) -> float:
        deltaE = self.jLevel.E_SI - self.iLevel.E_SI
        return Const.HC / deltaE

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
        self.wavelength = np.linspace(self.minLambda, self.lambdaEdge, self.NlambdaGen)
        self.alpha = self.compute_alpha(self.wavelength)

    def compute_alpha(self, wavelength) -> np.ndarray:
        # if self.atom.name.strip() != 'H':
        # NOTE(cmo): As it should be, the general case is equivalent for H
        Z = self.jLevel.stage
        nEff = Z * np.sqrt(Const.ERydberg / (self.jLevel.E_SI - self.iLevel.E_SI))
        gbf0 = gaunt_bf(self.lambda0, nEff, Z)
        gbf = gaunt_bf(wavelength, nEff, Z)
        alpha = self.alpha0 * gbf / gbf0 * (wavelength / self.lambda0)**3
        alpha[wavelength < self.minLambda] = 0.0
        alpha[wavelength > self.lambdaEdge] = 0.0
        return alpha
        # else:
        #     sigma0 = 32.0 / (3.0 * np.sqrt(3.0)) * Const.Q_ELECTRON**2 / (4.0 * np.pi * Const.EPSILON_0) / (Const.M_ELECTRON * Const.CLIGHT) * Const.HPLANCK / (2.0 * Const.E_RYDBERG)
        #     nEff = np.sqrt(Const.E_RYDBERG / (self.jLevel.E_SI - self.iLevel.E_SI))
        #     gbf = gaunt_bf(wavelength, nEff, self.iLevel.stage+1)
        #     sigma = sigma0 * nEff * gbf * (wavelength / self.lambdaEdge)**3
        #     sigma[wavelength < self.minLambda] = 0.0
        #     sigma[wavelength > self.lambdaEdge] = 0.0
        #     return sigma

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

    @property
    def Nlambda(self) -> int:
        return self.wavelength.shape[0]

@dataclass
class CollisionalRates:
    j: int
    i: int
    atom: AtomicModel = field(init=False)

    def __repr__(self):
        s = 'CollisionalRates(j=%d, i=%d, temperature=%s, rates=%s)' % (self.j, self.i, repr(self.temperature), repr(self.rates))
        return s

    def setup(self, atom):
        pass

    def compute_rates(self, atmos, nstar, Cmat):
        pass

    def __eq__(self, other: object) -> bool:
        return model_component_eq(self, other)

    def __getstate__(self):
        state = self.__dict__.copy()
        try:
            del state['interpolator']
        except KeyError:
            pass

        return state