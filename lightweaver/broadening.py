from dataclasses import dataclass, field
from typing import Sequence, List, Optional, cast, Any, TYPE_CHECKING
import lightweaver.constants as Const
import numpy as np
from .atomic_table import PeriodicTable
from .barklem import Barklem

if TYPE_CHECKING:
    from .atomic_model import AtomicLine
    from .atmosphere import Atmosphere
    from .atomic_set import SpeciesStateTable

@dataclass
class LineBroadeningResult:
    '''
    Result expected from instances of `LineBroadening.broaden`.
    '''
    natural: np.ndarray
    Qelast: np.ndarray
    other: Optional[List] = None


@dataclass
class LineBroadener:
    '''
    Base class for broadening implementations. To be used if your broadener
    does something special and can't just return an array.
    '''
    def __repr__(self):
        raise NotImplementedError

    def setup(self, line: 'AtomicLine'):
        pass

    def broaden(self, atmos: 'Atmosphere', eqPops: 'SpeciesStateTable') -> Any:
        raise NotImplementedError

@dataclass
class StandardLineBroadener(LineBroadener):
    '''
    Standard base class for broadening implementations. Unless you need to do
    something weird, inherit from this one.
    '''
    def __repr__(self):
        raise NotImplementedError

    def setup(self, line: 'AtomicLine'):
        pass

    def broaden(self, atmos: 'Atmosphere', eqPops: 'SpeciesStateTable') -> np.ndarray:
        raise NotImplementedError


@dataclass
class LineBroadening:
    '''
    Standard component of AtomicLine to compute the broadening parameters in
    a flexible way.

    For most Voigt-like situations, this should be usable without
    modifications, but this class can be inherited from to make substantial
    modifications.

    Parameters
    ----------
    natural : list of StandardLineBroadener
        List of broadening terms that are not elastic collisions (separated for PRD).
    elastic : list of StandardLineBroadener
        List of elastic broadening terms.
    other : list of LineBroadener, optional
        List of other broadening terms, not used by the VoigtLine by default,
        but existing to provide _options_ (default: None)
    '''
    natural: List[StandardLineBroadener]
    elastic: List[StandardLineBroadener]
    other: Optional[List[LineBroadener]] = None

    def __repr__(self):
        otherStr = '' if self.other is None else ', other=%s' % repr(self.other)
        s = 'LineBroadening(natural=%s, elastic=%s%s)' % (repr(self.natural), repr(self.elastic), otherStr)
        return s

    def __post_init__(self):
        if len(self.natural) == 0 and len(self.elastic) == 0:
            raise ValueError('No standard broadening terms provided to LineBroadening')

    def setup(self, line: 'AtomicLine'):
        b: LineBroadener
        for b in self.natural:
            b.setup(line)

        for b in self.elastic:
            b.setup(line)

        if self.other is not None:
            for b in self.other:
                b.setup(line)

    @staticmethod
    def sum_broadening_list(broadeners: List[StandardLineBroadener], atmos: 'Atmosphere',
                            eqPops: 'SpeciesStateTable') -> Optional[np.ndarray]:
        '''
        Sums a list of StandardLineBroadeners.
        '''
        if len(broadeners) == 0:
            return None

        result = broadeners[0].broaden(atmos, eqPops)
        for b in broadeners[1:]:
            result += b.broaden(atmos, eqPops)
        return result

    @staticmethod
    def compute_other_broadening(broadeners: Optional[List[LineBroadener]],
                                 atmos: 'Atmosphere',
                                 eqPops: 'SpeciesStateTable') -> Optional[List]:
        '''
        Returns a list of the computed broadening terms.
        '''

        if broadeners is None:
            return None
        if len(broadeners) == 0:
            return None

        result = [b.broaden(atmos, eqPops) for b in broadeners]
        return result

    def broaden(self, atmos: 'Atmosphere', eqPops: 'SpeciesStateTable') -> LineBroadeningResult:
        '''
        Computes the broadening, this function is called by the AtomicLine object.
        '''
        natural = self.sum_broadening_list(self.natural, atmos, eqPops)
        Qelast = self.sum_broadening_list(self.elastic, atmos, eqPops)

        others = self.compute_other_broadening(self.other, atmos, eqPops)

        if natural is None:
            natural = np.zeros_like(Qelast)
        elif Qelast is None:
            Qelast = np.zeros_like(natural)

        return LineBroadeningResult(natural=natural,
                                    Qelast=Qelast, other=others)


@dataclass(eq=False)
class VdwApprox(StandardLineBroadener):
    '''
    Base class for van der Waals approximation using a list of coefficients
    (vals).
    '''
    vals: Sequence[float]
    line: 'AtomicLine' = field(init=False)

    def setup(self, line: 'AtomicLine'):
        self.line = line

    def __repr__(self):
        s = '%s(vals=%s)' % (type(self).__name__, repr(self.vals))
        return s

    def __eq__(self, other):
        if type(self) is not type(other):
            return False

        if self.vals != other.vals:
            return False

        try:
            if self.line != other.line:
                return False
        except:
            pass

        return True


@dataclass(eq=False, repr=False)
class VdwUnsold(VdwApprox):
    '''
    Implementation of the Unsold method for van der Waals broadening.
    '''
    def setup(self, line: 'AtomicLine'):
        self.line = line
        if len(self.vals) != 2:
            raise ValueError('VdwUnsold expects 2 coefficients (%s)' % repr(line))

        Z = line.jLevel.stage + 1
        cont = line.overlyingContinuumLevel

        deltaR = (Const.ERydberg / (cont.E_SI - line.jLevel.E_SI))**2 \
                 - (Const.ERydberg / (cont.E_SI - line.iLevel.E_SI))**2
        fourPiEps0 = 4.0 * np.pi * Const.Epsilon0
        self.C625 = (2.5 * Const.QElectron**2 / fourPiEps0 * Const.ABarH / fourPiEps0 \
                     * 2 * np.pi * (Z * Const.RBohr)**2 / Const.HPlanck * deltaR)**0.4

        element = line.atom.element

        self.vRel35He = (8.0 * Const.KBoltzmann / (np.pi*Const.Amu * element.mass)\
                         * (1.0 + element.mass / PeriodicTable[2].mass))**0.3
        self.vRel35H = (8.0 * Const.KBoltzmann / (np.pi*Const.Amu * element.mass)\
                         * (1.0 + element.mass / PeriodicTable[1].mass))**0.3


    def broaden(self, atmos: 'Atmosphere', eqPops: 'SpeciesStateTable') -> np.ndarray:
        '''
        The function that is called by LineBroadening.

        Parameters
        ----------
        atmos : Atmosphere
            The atmosphere in which to compute the broadening.
        eqPops : SpeciesStateTable
            The populations to use for computing the broadening.

        Returns
        -------
        broad : np.ndarray
            An array detailing the broadening at each location in the
            atmosphere [Nspace].
        '''
        heAbund = eqPops.abundance[PeriodicTable[2]]
        cross = 8.08 * (self.vals[0] * self.vRel35H \
                             + self.vals[1] * heAbund * self.vRel35He) * self.C625
        nHGround = eqPops['H'][0, :]
        broad = cross * atmos.temperature**0.3 * nHGround
        return broad


@dataclass(eq=False, repr=False)
class VdwBarklem(VdwApprox):
    '''
    Implementation of the Barklem method for van der Waals broadening.
    '''
    def setup(self, line: 'AtomicLine'):
        self.line = line
        if len(self.vals) != 2:
            raise ValueError('VdwBarklem expects 2 coefficients (%s)' % (repr(line)))
        newVals = Barklem.get_active_cross_section(line.atom, line, self.vals)

        self.barklemVals = newVals

        Z = line.jLevel.stage + 1
        cont = line.overlyingContinuumLevel

        deltaR = (Const.ERydberg / (cont.E_SI - line.jLevel.E_SI))**2 \
                 - (Const.ERydberg / (cont.E_SI - line.iLevel.E_SI))**2
        fourPiEps0 = 4.0 * np.pi * Const.Epsilon0
        self.C625 = (2.5 * Const.QElectron**2 / fourPiEps0 * Const.ABarH / fourPiEps0 \
                     * 2 * np.pi * (Z * Const.RBohr)**2 / Const.HPlanck * deltaR)**0.4

        element = line.atom.element

        self.vRel35He = (8.0 * Const.KBoltzmann / (np.pi*Const.Amu * element.mass)\
                         * (1.0 + element.mass / PeriodicTable[2].mass))**0.3


    def broaden(self, atmos: 'Atmosphere', eqPops: 'SpeciesStateTable') -> np.ndarray:
        '''
        The function that is called by LineBroadening.

        Parameters
        ----------
        atmos : Atmosphere
            The atmosphere in which to compute the broadening.
        eqPops : SpeciesStateTable
            The populations to use for computing the broadening.

        Returns
        -------
        broad : np.ndarray
            An array detailing the broadening at each location in the
            atmosphere [Nspace].
        '''
        heAbund = eqPops.abundance[PeriodicTable[2]]
        nHGround = eqPops['H'][0, :]
        cross = 8.08 * self.barklemVals[2] * heAbund * self.vRel35He * self.C625

        broad = self.barklemVals[0] * atmos.temperature**(0.5*(1.0-self.barklemVals[1])) \
                 + cross * atmos.temperature**0.3
        broad *= nHGround
        return broad


@dataclass(eq=False)
class RadiativeBroadening(StandardLineBroadener):
    '''
    Simple constant radiative broadening with coefficient gamma.
    '''
    gamma: float
    line: 'AtomicLine' = field(init=False)

    def setup(self, line: 'AtomicLine'):
        self.line = line

    def __repr__(self):
        s = '%s(gamma=%g)' % (type(self).__name__, self.gamma)
        return s

    def __eq__(self, other):
        if type(self) is not type(other):
            return False

        if self.gamma != other.gamma:
            return False

        try:
            if self.line != other.line:
                return False
        except:
            pass

        return True

    def broaden(self, atmos: 'Atmosphere', eqPops: 'SpeciesStateTable') -> np.ndarray:
        '''
        The function that is called by LineBroadening.

        Parameters
        ----------
        atmos : Atmosphere
            The atmosphere in which to compute the broadening.
        eqPops : SpeciesStateTable
            The populations to use for computing the broadening.

        Returns
        -------
        broad : np.ndarray
            An array detailing the broadening at each location in the
            atmosphere [Nspace].
        '''
        return np.ones_like(atmos.temperature) * self.gamma

@dataclass
class QuadraticStarkBroadening(StandardLineBroadener):
    '''
    Lindholm theory result for Quadratic Stark broadening by electrons and
    singly ionised particles.
    Follows HM2014 pp. 238-239, uses C4 from Traving 1960 via RH.
    '''
    coeff: float
    line: 'AtomicLine' = field(init=False)

    def __repr__(self):
        s = '%s(coeff=%g)' % (type(self).__name__, self.coeff)
        return s

    def __eq__(self, other):
        if type(self) is not type(other):
            return False

        if self.coeff != other.coeff:
            return False

        try:
            if self.line != other.line:
                return False
        except:
            pass

        return True

    def setup(self, line: 'AtomicLine'):
        self.line = line
        weight = line.atom.element.mass
        C = 8.0 * Const.KBoltzmann / (np.pi * Const.Amu * weight)
        Cm = (1.0 + weight / (Const.MElectron / Const.Amu))**(1.0/6.0)
        # NOTE(cmo): 28.0 is average atomic weight
        Cm += (1.0 + weight / (28.0))**(1.0/6.0)
        self.C = C
        self.Cm = Cm

        Z = line.iLevel.stage + 1
        cont = line.overlyingContinuumLevel

        E_Ryd = Const.ERydberg / (1.0 + Const.MElectron / (weight * Const.Amu))
        neff_l = Z * np.sqrt(E_Ryd / (cont.E_SI - line.iLevel.E_SI))
        neff_u = Z * np.sqrt(E_Ryd / (cont.E_SI - line.jLevel.E_SI))

        C4 = Const.QElectron**2 / (4.0 * np.pi * Const.Epsilon0) \
            * Const.RBohr \
            * (2.0 * np.pi * Const.RBohr**2 / Const.HPlanck) / (18.0 * Z**4) \
            * ((neff_u * (5.0 * neff_u**2 + 1.0))**2 \
                - (neff_l * (5.0 * neff_l**2 + 1.0))**2)
        self.cStark23 = 11.37 * (self.coeff * C4)**(2.0/3.0)

    def broaden(self, atmos: 'Atmosphere', eqPops: 'SpeciesStateTable') -> np.ndarray:
        '''
        The function that is called by LineBroadening.

        Parameters
        ----------
        atmos : Atmosphere
            The atmosphere in which to compute the broadening.
        eqPops : SpeciesStateTable
            The populations to use for computing the broadening.

        Returns
        -------
        broad : np.ndarray
            An array detailing the broadening at each location in the
            atmosphere [Nspace].
        '''
        vRel = (self.C * atmos.temperature)**(1.0/6.0) * self.Cm
        stark = self.cStark23 * vRel * atmos.ne
        return stark

@dataclass
class MultiplicativeStarkBroadening(StandardLineBroadener):
    '''
    Simple expression for multiplicative Stark broadening, assumes that this
    can be expresed as a constant * ne.
    '''
    coeff: float

    def __repr__(self):
        s = '%s(coeff=%g)' % (type(self).__name__, self.coeff)
        return s

    def __eq__(self, other):
        if type(self) is not type(other):
            return False

        if self.coeff != other.coeff:
            return False

        return True

    def broaden(self, atmos: 'Atmosphere', eqPops: 'SpeciesStateTable') -> np.ndarray:
        '''
        The function that is called by LineBroadening.

        Parameters
        ----------
        atmos : Atmosphere
            The atmosphere in which to compute the broadening.
        eqPops : SpeciesStateTable
            The populations to use for computing the broadening.

        Returns
        -------
        broad : np.ndarray
            An array detailing the broadening at each location in the
            atmosphere [Nspace].
        '''
        return self.coeff * atmos.ne # type: ignore

@dataclass
class HydrogenLinearStarkBroadening(StandardLineBroadener):
    '''
    Linear Stark broadening for the case of Hydrogen from Sutton 1978 (like
    RH).
    '''
    line: 'AtomicLine' = field(init=False)

    def __repr__(self):
        s = '%s()' % type(self).__name__
        return s

    def __eq__(self, other):
        if type(self) is not type(other):
            return False

        try:
            if self.line != other.line:
                return False
        except:
            pass

        return True

    def setup(self, line: 'AtomicLine'):
        self.line = line

        if line.atom.element.Z != 1:
            raise ValueError('HydrogenicLinearStarkBroadening applied to non-Hydrogen line')

    def broaden(self, atmos: 'Atmosphere', eqPops: 'SpeciesStateTable') -> np.ndarray:
        '''
        The function that is called by LineBroadening.

        Parameters
        ----------
        atmos : Atmosphere
            The atmosphere in which to compute the broadening.
        eqPops : SpeciesStateTable
            The populations to use for computing the broadening.

        Returns
        -------
        broad : np.ndarray
            An array detailing the broadening at each location in the
            atmosphere [Nspace].
        '''
        nUpper = int(np.round(np.sqrt(0.5*self.line.jLevel.g)))
        nLower = int(np.round(np.sqrt(0.5*self.line.iLevel.g)))

        a1 = 0.642 if nUpper - nLower == 1 else 1.0
        C = a1 * 0.6 * (nUpper**2 - nLower**2) * Const.CM_TO_M**2
        GStark = C * atmos.ne**(2.0/3.0)
        return GStark
