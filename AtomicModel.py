from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Sequence, Optional
import re

import numpy as np
from scipy.interpolate import interp1d
from parse import parse

import Constants as Const
from Utils import gaunt_bf
from AtomicTable import AtomicTable
from Barklem import Barklem

# TODO(cmo): Handle Stark (+ Linear stark for H) broadening in the same way as VdW, i.e. move all of Broaden to Python AtomicModel
# TODO(cmo): Add collisional beam rates as a default empty field to AtomicModel
# TODO(cmo): Prune Kurucz lines after loading all of them, based on the ones we have detailed models for. This lets us remove the element checks from rlk_opacity!
# May need to compute the RLK unsold cross-section in Python, where we have access to all the element stuff
# RlkProfile needs the atomic weight
# The easiest option may instead be to fill a CElement struct for these functions, especially the ICE stuff

# Process of setting up a simulation:
# Load/Update abundances/AtomicTable
# Load Atomic Models -- select active and initial solutions
# Load Background Molecular Models
# Setup Atmosphere
# Load and prune rlk lines
# Set up computational atoms
# Set up CAtomicTable -- backrefs
# Set up computational molecule
# Set up spectrum/wvl contributions/activeset i.e. Spectrum
# LTE/initial/ICE pops
# Compute background 
# Compute line profiles for active -- damping and collisions from Python atoms
# Solve!




@dataclass
class PrincipalQuantum:
    S: float
    L: int
    J: float

class CompositeLevelError(Exception):
    pass

@dataclass
class AtomicModel:
    name: str
    levels: Sequence['AtomicLevel']
    lines: Sequence['AtomicLine']
    continua: Sequence['AtomicContinuum']
    collisions: Sequence['CollisionalRates']

    # We need lots of periodic table data to calculate lots of stuff. This lets us override the default AtomicTable if abundaces/metallicity is changed.
    # TODO(cmo):
    # i.e. a function will replace each atom with another with the same name, levels, lines, continua, collisions, but a different atomic table
    atomicTable: AtomicTable = field(default_factory=AtomicTable)

    def __post_init__(self):
        for l in self.levels:
            l.setup(self)

        for l in self.lines:
            l.setup(self)
        
        # This is separate because all of the lines in 
        # an atom need to be initialised first
        for l in self.lines:
            l.setup_xrd_wavelength()

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

@dataclass
class AtomicLevel:
    E: float
    g: float
    label: str
    stage: int
    noPol: bool = False
    atom: AtomicModel = field(init=False)

    def setup(self, atom):
        self.atom = atom
        try: 
            self.qNo = determinate(self)
        except Exception as e:
            print(self.atom.name)
            self.noPol = True
            print('Unable to treat level with polarisation: %s' % e)
            

    @property
    def E_SI(self):
        return self.E * Const.HC / Const.CM_TO_M

    def __repr__(self):
        s = 'AtomicLevel(E=%f, g=%f, label="%s", stage=%d)' % (self.E, self.g, self.label, self.stage)
        return s


# Since we're always going to assume a moving atmosphere, the question of symmetry is unnecessary.
# class LineSymmetry(Enum):
#     Asymmetric = 0
#     Symmetric = auto()

# In my line lists, Gauss is only ever commented out, and composite is extremely rare
# class LineType(Enum):
#     Gauss = 0
#     Voigt = auto()
#     PRD = auto()
#     Composit = auto()

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

@dataclass
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

        deltaR = (Const.E_RYDBERG / (cont.E_SI - line.jLevel.E_SI))**2 \
                 - (Const.E_RYDBERG / (cont.E_SI - line.iLevel.E_SI))**2
        fourPiEps0 = 4.0 * np.pi * Const.EPSILON_0
        C625 = (2.5 * Const.Q_ELECTRON**2 / fourPiEps0 * Const.ABARH / fourPiEps0 \
                   * 2 * np.pi * (Z * Const.RBOHR)**2 / Const.HPLANCK * deltaR)**0.4

        name = line.atom.name

        vRel35He = (8.0 * Const.KBOLTZMANN / (np.pi*Const.AMU * table[name].weight)\
                    * (1.0 + table[name].weight / table['He'].weight))**0.3
        vRel35H = (8.0 * Const.KBOLTZMANN / (np.pi*Const.AMU * table[name].weight)\
                    * (1.0 + table[name].weight / table['H'].weight))**0.3

        heAbund = table['He'].abundance
        self.cross = 8.08 * (self.vals[0] * vRel35H \
                             + self.vals[1] * heAbund * vRel35He) * C625

    def broaden(self, temperature, nHGround, broad):
        broad[:] = self.cross * temperature**0.3 * nHGround 


@dataclass
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

@dataclass
class VdwBarklem(VdwApprox):
    # NOTE(cmo): Since Helium is treated as per Unsold, only 3 vals are used
    def setup(self, line: 'AtomicLine', table: AtomicTable):
        self.line = line
        # TODO(cmo): Need to process the provided values with Barklem.py
        if len(self.vals) != 2:
            # TODO(cmo): I think it actually only expects 2, but related to handling in Barklem.py
            raise ValueError('VdwBarklem expects 2 coefficients (%s)' % (repr(line)))
        self.barklem = Barklem(table)
        try:
            newVals = self.barklem.getActiveCrossSection(line.atom, line)
        except:
            print("Unable to treat line %d->%d of atom %s with Barklem broadening, using Unsold." % (line.j, line.i, line.atom.name))
            # NOTE(cmo): This should all work fine, but mypy typing doesn't like it.
            # Shhh no tears, only dreams now
            self.broaden = lambda *args: VdwUnsold.broaden(self, *args) # type: ignore
            VdwUnsold.setup(self, line, table) # type: ignore
            return
        
        self.vals = newVals

        Z = line.jLevel.stage + 1
        j = line.j
        ic = j + 1
        while line.atom.levels[ic].stage < Z:
            ic += 1
        cont = line.atom.levels[ic]

        deltaR = (Const.E_RYDBERG / (cont.E_SI - line.jLevel.E_SI))**2 \
                 - (Const.E_RYDBERG / (cont.E_SI - line.iLevel.E_SI))**2
        fourPiEps0 = 4.0 * np.pi * Const.EPSILON_0
        C625 = (2.5 * Const.Q_ELECTRON**2 / fourPiEps0 * Const.ABARH / fourPiEps0 \
                   * 2 * np.pi * (Z * Const.RBOHR)**2 / Const.HPLANCK * deltaR)**0.4

        name = line.atom.name

        vRel35He = (8.0 * Const.KBOLTZMANN / (np.pi*Const.AMU * table[name].weight)\
                    * (1.0 + table[name].weight / table['He'].weight))**0.3

        heAbund = table['He'].abundance
        self.cross = 8.08 * self.vals[2] * heAbund * vRel35He * C625

    def broaden(self, temperature, nHGround, broad):
        broad[:] = self.vals[0] * temperature**(0.5*(1.0-self.vals[1])) \
                    + self.cross * temperature**0.3
        broad *= nHGround
        
    
@dataclass
class AtomicLine:
    j: int
    i: int
    f: float
    type: LineType
    Nlambda: int
    qCore: float
    qWing: float
    vdw: VdwApprox
    gRad: float
    stark: float
    gLandeEff: float = 0.0
    atom: AtomicModel = field(init=False)
    jLevel: AtomicLevel = field(init=False)
    iLevel: AtomicLevel = field(init=False)
    wavelength: np.ndarray = field(init=False)

    def __repr__(self):
        s = 'AtomicLine(j=%d, i=%d, f=%e, type=%s, Nlambda=%d, qCore=%f, qWing=%f, vdw=%s, gRad=%e, stark=%f, gLandeEff=%f)' % (
                self.j, self.i, self.f, repr(self.type), self.Nlambda, self.qCore, self.qWing, repr(self.vdw), 
                self.gRad, self.stark, self.gLandeEff)
        return s

    def __hash__(self):
        return hash(repr(self))

@dataclass
class VoigtLine(AtomicLine):
    def setup(self, atom):
        if self.j < self.i:
            self.i, self.j = self.j, self.i

        self.atom: AtomicModel = atom
        self.jLevel: AtomicLevel = self.atom.levels[self.j]
        self.iLevel: AtomicLevel = self.atom.levels[self.i]

        # Might need to init the vdw here
        # we do.

        self.vdw.setup(self, atom.atomicTable)

    def __repr__(self):
        s = 'VoigtLine(j=%d, i=%d, f=%e, type=%s, Nlambda=%d, qCore=%f, qWing=%f, vdw=%s, gRad=%e, stark=%f, gLandeEff=%f)' % (
                self.j, self.i, self.f, repr(self.type), self.Nlambda, self.qCore, self.qWing, repr(self.vdw), 
                self.gRad, self.stark, self.gLandeEff)
        return s

    def stark_broaden(self, atmos):
        weight = self.atom.atomicTable[self.atom.name].weight
        C = 8.0 * Const.KBOLTZMANN / (np.pi * Const.AMU * weight)
        Cm = (1.0 + weight / (Const.M_ELECTRON / Const.AMU))**(1.0/6.0)
        # NOTE(cmo): 28.0 is average atomic weight
        Cm += (1.0 + weight / (28.0))**(1.0/6.0)

        Z = self.iLevel.stage + 1
        ic = self.i + 1
        while self.atom.levels[ic].stage < Z and ic < len(self.atom.levels):
            ic += 1
        if self.atom.levels[ic].stage == self.iLevel.stage:
            raise ValueError('Cant find overlying cont: %s' % repr(self))

        E_Ryd = Const.E_RYDBERG / (1.0 + Const.M_ELECTRON / (weight * Const.AMU))
        neff_l = Z * np.sqrt(E_Ryd / (self.atom.levels[ic].E_SI - self.iLevel.E_SI))
        neff_u = Z * np.sqrt(E_Ryd / (self.atom.levels[ic].E_SI - self.jLevel.E_SI))

        C4 = Const.Q_ELECTRON**2 / (4.0 * np.pi * Const.EPSILON_0) \
             * Const.RBOHR \
             * (2.0 * np.pi * Const.RBOHR**2 / Const.HPLANCK) / (18.0 * Z**4) \
             * ((neff_u * (5.0 * neff_u**2 + 1.0))**2 \
                 - (neff_l * (5.0 * neff_l**2 + 1.0))**2)
        cStark23 = 11.37 * (self.stark * C4)**(2.0/3.0)

        vRel = (C * atmos.temperature)**(1.0/6.0) * Cm
        return cStark23 * vRel * atmos.ne

    def setup_xrd_wavelength(self):
        self.xrd = []

        # This is just a hack to keep the search happy
        self.wavelength = np.zeros(1)
        # NOTE(cmo): This is currently disabled
        if self.type == LineType.PRD and False:

            thisIdx = self.atom.lines.index(self)

            for i, l in enumerate(self.atom.lines):
                if l.type == LineType.PRD and l.j == self.j and l.i != self.i:
                    self.xrd.append(l)

                    if i > thisIdx:
                        waveratio = self.xrd[-1].lambda0 / self.lambda0
                        self.xrd[-1].qWing = waveratio * self.qWing

            if len(self.xrd) > 0:
                print("Found %d subordinate PRD lines, for line %d->%d of atom %s" % \
                    (len(self.xrd), self.j, self.i, self.atom.name))

        # Compute default lambda grid
        # We're going to assume that a line profile is always asymmetric -- should get rid of keyword
        Nlambda = self.Nlambda // 2 if self.Nlambda % 2 == 0 else (self.Nlambda + 1) // 2

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

        qToLambda = self.lambda0 * (Const.VMICRO_CHAR / Const.CLIGHT)
        self.Nlambda = 2 * Nlambda - 1
        line = np.zeros(self.Nlambda)
        Nmid = Nlambda - 1

        line[Nmid] = self.lambda0
        line[:Nmid][::-1] = self.lambda0 - qToLambda * self.q[1:]
        line[Nmid+1:] = self.lambda0 + qToLambda * self.q[1:]
        self.wavelength = line

    def polarized_wavelength(self, bChar: Optional[float]=None) -> Sequence[float]:
        ## NOTE(cmo): bChar in TESLA
        if any([self.iLevel.noPol, self.jLevel.noPol]) or \
           (self.gLandeEff == 0.0) and (self.jLevel.qNo.J - self.iLevel.qNo.J > 1.0):
            print("Can't treat line %d->%d with polarization" % (self.j, self.i))
            return self.wavelength

        if bChar is None:
            bChar = Const.B_CHAR
        # /* --- When magnetic field is present account for denser
        #        wavelength spacing in the unshifted \pi component, and the
        #        red and blue-shifted circularly polarized components.
        #        First, get characteristic Zeeman splitting -- ------------ */
        gLandeEff = effective_lande(self)
        qBChar = gLandeEff * (Const.Q_ELECTRON / (4.0 * np.pi * Const.M_ELECTRON)) * \
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
        qToLambda = self.lambda0 * (Const.VMICRO_CHAR / Const.CLIGHT)
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
        C: float = 2 * np.pi * (Const.Q_ELECTRON / Const.EPSILON_0) \
           * (Const.Q_ELECTRON / Const.M_ELECTRON) / Const.CLIGHT
        return C / self.lambda0_m**2 * gRatio * self.f

    @property
    def Bji(self) -> float:
        return self.lambda0_m**3 / (2.0 * Const.HC) * self.Aji

    @property
    def Bij(self) -> float:
        return self.jLevel.g / self.iLevel.g * self.Bji

    def damping(self, atmos, vBroad):
        aDamp = np.zeros(atmos.Nspace)
        Qelast = np.zeros(atmos.Nspace)

        self.vdw.broaden(atmos.temperature, atmos.hydrogenPops[0, :], aDamp)
        Qelast += aDamp

        Qelast += self.stark_broaden(atmos)

        cDop = self.lambda0_m / (4.0 * np.pi)
        aDamp = (self.gRad + Qelast) * cDop / vBroad
        return aDamp


def get_oribital_number(orbit: str) -> int:
    orbits = ['S', 'P', 'D', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'Q', 'R', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    return orbits.index(orbit)
    

def determinate(level: AtomicLevel) -> PrincipalQuantum:
    endIdx = [level.label.upper().rfind(x) for x in ['E', 'O']]
    maxIdx = max(endIdx)
    if maxIdx == -1:
        raise ValueError("Unable to determine parity of level %s" % (repr(level)))
    label = level.label[:maxIdx+1].upper()
    words: List[str] = label.split()

    # _, multiplicity, orbit = parse('{}{:d}{!s}', words[-1])
    match = re.match('[\S-]*(\d)(\S)[EO]$', words[-1])
    if match is None:
        raise ValueError('Unable to parse level label: %s' % level.label)
    else:
        multiplicity = int(match.group(1))
        orbit = match.group(2)
    S = (multiplicity - 1) / 2.0
    L = get_oribital_number(orbit)
    J = (level.g - 1.0) / 2.0

    if J > L + S:
        raise CompositeLevelError('J (%f) > L (%d) + S (%f): %s' %(J, L, S, repr(level)))

    return PrincipalQuantum(S=S, J=J, L=L)

def lande(qNo: PrincipalQuantum):
    if qNo.J == 0.0:
        return 0.0
    return 1.5 + (qNo.S * (qNo.S + 1.0) - qNo.L * (qNo.L + 1)) / (2.0 * qNo.J * (qNo.J + 1.0))

def effective_lande(line: AtomicLine):
    if line.gLandeEff != 0.0:
        return line.gLandeEff

    qL = line.iLevel.qNo
    qU = line.jLevel.qNo
    gL = lande(qL)
    gU = lande(qU)

    return 0.5 * (gU + gL) + \
           0.25 * (gU - gL) * (qU.J * (qU.J + 1.0) - qL.J * (qL.J + 1.0))


@dataclass
class AtomicContinuum:
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

@dataclass
class ExplicitContinuum(AtomicContinuum):
    alphaGrid: Sequence[Sequence[float]]
    Nlambda: int = field(init=False)

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
        self.Nlambda = self.wavelength.shape[0]

    def __repr__(self):
        s = 'ExplicitContinuum(j=%d, i=%d, alphaGrid=%s)' % (self.j, self.i, repr(self.alphaGrid))
        return s

    def compute_alpha(self, wavelength) -> np.ndarray:
        return interp1d(self.wavelength, self.alpha)(wavelength)

    @property
    def lambda0(self) -> float:
        return self.lambda0_m / Const.NM_TO_M

    @property
    def lambda0_m(self) -> float:
        deltaE = self.jLevel.E_SI - self.iLevel.E_SI
        return Const.HC / deltaE

@dataclass 
class HydrogenicContinuum(AtomicContinuum):
    alpha0: float
    minLambda: float
    Nlambda: int

    def __repr__(self): 
        s = 'HydrogenicContinuum(j=%d, i=%d, alpha0=%e, minLambda=%f, Nlambda=%d)' % (self.j, self.i, self.alpha0, self.minLambda, self.Nlambda)
        return s

    def setup(self, atom):
        if self.j < self.i:
            self.i, self.j = self.j, self.i
        self.atom = atom
        self.jLevel: AtomicLevel = atom.levels[self.j]
        self.iLevel: AtomicLevel = atom.levels[self.i]
        if self.minLambda >= self.lambda0:
            raise ValueError('Minimum wavelength is larger than continuum edge at %f [nm] in continuum %s' % (self.lambda0, repr(self)))
        self.wavelength = np.linspace(self.minLambda, self.lambda0, self.Nlambda)

        self.alpha = self.compute_alpha(self.wavelength)

    def compute_alpha(self, wavelength) -> np.ndarray:
        Z = self.jLevel.stage
        nEff = Z * np.sqrt(Const.E_RYDBERG / (self.jLevel.E_SI - self.iLevel.E_SI))
        gbf0 = gaunt_bf(self.lambda0, nEff, Z)
        gbf = gaunt_bf(wavelength, nEff, Z)
        alpha = self.alpha0 * gbf / gbf0 * (wavelength / self.lambda0)**3
        return alpha

    @property
    def lambda0(self) -> float:
        return self.lambda0_m / Const.NM_TO_M

    @property
    def lambda0_m(self) -> float:
        deltaE = self.jLevel.E_SI - self.iLevel.E_SI
        return Const.HC / deltaE


@dataclass
class CollisionalRates:
    j: int
    i: int
    # Make sure to swap if wrong order in setup
    temperature: Sequence[float]
    rates: Sequence[float]

    def __repr__(self):
        s = 'CollisionalRates(j=%d, i=%d, temperature=%s, rates=%s)' % (self.j, self.i, repr(self.temperature), repr(self.rates))
        return s

    def setup(self, atom):
        pass

    def compute_rates(self, atmos, nstar, Cmat):
        pass

@dataclass
class Omega(CollisionalRates):
    def __repr__(self):
        s = 'Omega(j=%d, i=%d, temperature=%s, rates=%s)' % (self.j, self.i, repr(self.temperature), repr(self.rates))
        return s

    def setup(self, atom):
        i, j = self.i, self.j
        self.i = min(i, j)
        self.j = max(i, j)
        self.atom = atom
        self.jLevel = atom.levels[self.j]
        self.iLevel = atom.levels[self.i]
        self.C0 = Const.E_RYDBERG / np.sqrt(Const.M_ELECTRON) * np.pi * Const.RBOHR**2 * np.sqrt(8.0 / (np.pi * Const.KBOLTZMANN))

    def compute_rates(self, atmos, nstar, Cmat):
        # TODO(cmo): Remove the nstar argument -- replace with g_ij exp(-hv/kbT)
        # NOTE(cmo): This is only linear
        C = interp1d(self.temperature, self.rates, kind=3)(atmos.temperature)
        Cdown = self.C0 * atmos.ne * C / (self.jLevel.g * np.sqrt(atmos.temperature))
        Cmat[self.i, self.j, :] += Cdown
        Cmat[self.j, self.i, :] += Cdown * nstar[self.j] / nstar[self.i]

@dataclass
class CI(CollisionalRates):
    def __repr__(self):
        s = 'CI(j=%d, i=%d, temperature=%s, rates=%s)' % (self.j, self.i, repr(self.temperature), repr(self.rates))
        return s

    def setup(self, atom):
        i, j = self.i, self.j
        self.i = min(i, j)
        self.j = max(i, j)
        self.atom = atom
        self.jLevel = atom.levels[self.j]
        self.iLevel = atom.levels[self.i]
        self.dE = self.jLevel.E_SI - self.iLevel.E_SI

    def compute_rates(self, atmos, nstar, Cmat):
        C = interp1d(self.temperature, self.rates, kind=3)(atmos.temperature)
        Cup = C * atmos.ne * np.exp(-self.dE / (Const.KBOLTZMANN * atmos.temperature)) * np.sqrt(atmos.temperature)
        Cmat[self.j, self.i, :] += Cup
        Cmat[self.i, self.j, :] += Cup * nstar[self.i] / nstar[self.j]


