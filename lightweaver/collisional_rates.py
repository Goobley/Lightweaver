from .utils import sequence_repr
import lightweaver.constants as Const
from dataclasses import dataclass, field
import numpy as np
from typing import Sequence, cast, TYPE_CHECKING
from scipy.special import exp1
from numba import njit
from weno4 import weno4

if TYPE_CHECKING:
    from .atomic_model import AtomicModel
    from .atmosphere import Atmosphere
    from .atomic_set import SpeciesStateTable

@dataclass
class CollisionalRates:
    '''
    Base class for all CollisionalRates.
    '''
    j: int
    i: int
    atom: 'AtomicModel' = field(init=False)

    def __repr__(self):
        s = 'CollisionalRates(j=%d, i=%d)' % (self.j, self.i)
        return s

    def setup(self, atom):
        pass

    def compute_rates(self, atmos: 'Atmosphere', eqPops: 'SpeciesStateTable', Cmat: np.ndarray):
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        if other is self:
            return True

        return repr(self) == repr(other)


@dataclass(eq=False)
class TemperatureInterpolationRates(CollisionalRates):
    '''
    Base class for rates defined by interpolating a coefficient on a
    temperature grid.
    '''
    temperature: Sequence[float]
    rates: Sequence[float]
    def __repr__(self):
        s = '%s(j=%d, i=%d, temperature=%s, rates=%s)' % (type(self).__name__,
                                                          self.j, self.i,
                                                          sequence_repr(self.temperature), sequence_repr(self.rates))
        return s

    def setup(self, atom):
        i, j = self.i, self.j
        self.i = min(i, j)
        self.j = max(i, j)
        self.atom = atom
        self.jLevel = atom.levels[self.j]
        self.iLevel = atom.levels[self.i]
        self.temperature = np.asarray(self.temperature)
        self.rates = np.asarray(self.rates)

@dataclass(eq=False, repr=False)
class Omega(TemperatureInterpolationRates):
    '''
    Collisional (de-)excitation of ions by electrons (dimensionless).
    Omega as in Seaton's collision strength.
    Rate scales as 1/(sqrt(T)) exp(DeltaE).
    '''
    def setup(self, atom):
        super().setup(atom)
        self.C0 = Const.ERydberg / np.sqrt(Const.MElectron) * np.pi * Const.RBohr**2 * np.sqrt(8.0 / (np.pi * Const.KBoltzmann))

    def compute_rates(self, atmos: 'Atmosphere', eqPops: 'SpeciesStateTable', Cmat: np.ndarray):
        C = weno4(atmos.temperature, self.temperature, self.rates)
        C[C < 0.0] = 0.0
        nstar = eqPops.atomicPops[self.atom.element].nStar
        Cdown = self.C0 * atmos.ne * C / (self.jLevel.g * np.sqrt(atmos.temperature))
        Cmat[self.i, self.j, :] += Cdown
        Cmat[self.j, self.i, :] += Cdown * nstar[self.j] / nstar[self.i]

@dataclass(eq=False, repr=False)
class CI(TemperatureInterpolationRates):
    '''
    Collisional ionisation by electrons.
    Units: s^-1 K^-1/2 m^3
    Rate scales as sqrt(T) exp(DeltaE)
    '''
    def setup(self, atom):
        super().setup(atom)
        self.dE = self.jLevel.E_SI - self.iLevel.E_SI

    def compute_rates(self, atmos: 'Atmosphere', eqPops: 'SpeciesStateTable', Cmat: np.ndarray):
        C = weno4(atmos.temperature, self.temperature, self.rates)
        C[C < 0.0] = 0.0
        nstar = eqPops.atomicPops[self.atom.element].nStar
        Cup = C * atmos.ne * np.exp(-self.dE / (Const.KBoltzmann * atmos.temperature)) * np.sqrt(atmos.temperature)
        Cmat[self.j, self.i, :] += Cup
        Cmat[self.i, self.j, :] += Cup * nstar[self.i] / nstar[self.j]


@dataclass(eq=False, repr=False)
class CE(TemperatureInterpolationRates):
    '''
    Collisional (de-)excitation of neutrals by electrons.
    Units: s^-1 K^-1/2 m^3
    Rate scales as sqrt(T) exp(DeltaE)
    '''
    def setup(self, atom):
        super().setup(atom)
        self.gij = self.iLevel.g / self.jLevel.g

    def compute_rates(self, atmos: 'Atmosphere', eqPops: 'SpeciesStateTable', Cmat: np.ndarray):
        C = weno4(atmos.temperature, self.temperature, self.rates)
        C[C < 0.0] = 0.0
        nstar = eqPops.atomicPops[self.atom.element].nStar
        Cdown = C * atmos.ne * self.gij * np.sqrt(atmos.temperature)
        Cmat[self.i, self.j, :] += Cdown
        Cmat[self.j, self.i, :] += Cdown * nstar[self.j] / nstar[self.i]

@dataclass(eq=False, repr=False)
class CP(TemperatureInterpolationRates):
    '''
    Collisional (de-)excitation by protons.
    Units: s^-1 m^3
    '''
    def compute_rates(self, atmos: 'Atmosphere', eqPops: 'SpeciesStateTable', Cmat: np.ndarray):
        C = weno4(atmos.temperature, self.temperature, self.rates)
        C[C < 0.0] = 0.0
        nProton = eqPops['H'][-1, :]
        Cdown = C * nProton
        nstar = eqPops.atomicPops[self.atom.element].nStar
        Cmat[self.i, self.j, :] += Cdown
        Cmat[self.j, self.i, :] += Cdown * nstar[self.j] / nstar[self.i]

@dataclass(eq=False, repr=False)
class CH(TemperatureInterpolationRates):
    '''
    Collisions with neutral hydrogen.
    Units: s^-1 m^3
    '''
    def compute_rates(self, atmos: 'Atmosphere', eqPops: 'SpeciesStateTable', Cmat: np.ndarray):
        C = weno4(atmos.temperature, self.temperature, self.rates)
        C[C < 0.0] = 0.0
        nh0 = eqPops['H'][0, :]
        Cup = C * nh0
        nstar = eqPops.atomicPops[self.atom.element].nStar
        Cmat[self.j, self.i, :] += Cup
        Cmat[self.i, self.j, :] += Cup * nstar[self.i] / nstar[self.j]

@dataclass(eq=False, repr=False)
class ChargeExchangeNeutralH(TemperatureInterpolationRates):
    '''
    Charge exchange with neutral hydrogen.
    Units: s^-1 m^3
    Note: downward rate only.
    '''
    def compute_rates(self, atmos: 'Atmosphere', eqPops: 'SpeciesStateTable', Cmat: np.ndarray):
        C = weno4(atmos.temperature, self.temperature, self.rates)
        C[C < 0.0] = 0.0
        nh0 = eqPops['H'][0, :]
        Cdown = C * nh0
        nstar = eqPops.atomicPops[self.atom.element].nStar
        Cmat[self.i, self.j, :] += Cdown

@dataclass(eq=False, repr=False)
class ChargeExchangeProton(TemperatureInterpolationRates):
    '''
    Charge exchange with protons.
    Units: s^-1 m^3
    Note: upward rate only.
    '''
    def compute_rates(self, atmos: 'Atmosphere', eqPops: 'SpeciesStateTable', Cmat: np.ndarray):
        C = weno4(atmos.temperature, self.temperature, self.rates)
        C[C < 0.0] = 0.0
        nProton = eqPops['H'][-1, :]
        Cup = C * nProton
        nstar = eqPops.atomicPops[self.atom.element].nStar
        Cmat[self.j, self.i, :] += Cup

def fone(x):
    # return np.where(x <= 50.0, np.exp(x) * exp1(x), 1.0/x)
    return np.where(x <= 50.0, np.exp(x) * exp1(x), (1.0 - 1.0 / x + 2.0 / x**2) / x)

@njit(cache=True)
def ftwo(x):
    p = np.array((1.0000e+00, 2.1658e+02, 2.0336e+04, 1.0911e+06, 3.7114e+07,
                  8.3963e+08, 1.2889e+10, 1.3449e+11, 9.4002e+11, 4.2571e+12,
                  1.1743e+13, 1.7549e+13, 1.0806e+13, 4.9776e+11, 0.0000))
    q = np.array((1.0000e+00, 2.1958e+02, 2.0984e+04, 1.1517e+06, 4.0349e+07,
                  9.4900e+08, 1.5345e+10, 1.7182e+11, 1.3249e+12, 6.9071e+12,
                  2.3531e+13, 4.9432e+13, 5.7760e+13, 3.0225e+13, 3.3641e+12))

    def ftwo_impl(x):
        if x > 4.0:
            px = p[0]
            xFact = 1.0
            for i in range(1, 15):
                xFact /= x
                px += p[i] * xFact

            qx = q[0]
            xFact = 1.0
            for i in range(1, 15):
                xFact /= x
                qx += q[i] * xFact

            return px / (qx * x**2)

        else:
            gamma = 0.5772156649
            f0x = np.pi**2 / 12.0
            term = 1.0
            count = 0.0
            fact = 1.0
            xFact = 1.0

            while abs(term / f0x) > 1e-8:
                count += 1.0
                fact *= count
                xFact *= -x
                term = xFact / (count**2 * fact)
                f0x += term

                if count > 100.0:
                    raise ValueError('ftwo too slow to converge')

            y = np.exp(x) * ((np.log(x) + gamma)**2 * 0.5 + f0x)
            return y

    y = np.empty_like(x)
    for i in range(x.shape[0]):
        y[i] = ftwo_impl(x[i])

    return y

@dataclass
class Ar85Cdi(CollisionalRates):
    '''
    Collisional ionisation rates based on Arnaud & Rothenflug (1985, ApJS 60).
    Units remain in CGS as per paper.
    '''
    cdi: Sequence[Sequence[float]]

    def __repr__(self):
        s = 'Ar85Cdi(j=%d, i=%d, cdi=%s)' % (self.j, self.i, sequence_repr(self.cdi))
        return s

    def setup(self, atom):
        i, j = self.i, self.j
        self.i = min(i, j)
        self.j = max(i, j)
        self.atom = atom
        self.iLevel = atom.levels[self.i]
        self.jLevel = atom.levels[self.j]
        self.cdi = np.array(self.cdi)

    def compute_rates(self, atmos: 'Atmosphere', eqPops: 'SpeciesStateTable', Cmat: np.ndarray):
        nstar = eqPops.atomicPops[self.atom.element].nStar
        Cup = np.zeros(atmos.Nspace)
        cdi = cast(np.ndarray, self.cdi)
        for m in range(cdi.shape[0]):
            xj = cdi[m, 0] * Const.EV / (Const.KBoltzmann * atmos.temperature)
            fac = np.exp(-xj) * np.sqrt(xj)
            fxj = cdi[m, 1] + cdi[m, 2] * (1.0 + xj) + (cdi[m, 3] - xj * (cdi[m, 1] + cdi[m, 2] * (2.0 + xj))) * fone(xj) + cdi[m, 4] * xj * ftwo(xj)

            fxj *= fac
            fac = 6.69e-7 / cdi[m, 0]**1.5
            Cup += fac * fxj * Const.CM_TO_M**3
        Cup[Cup < 0] = 0.0

        Cup *= atmos.ne
        Cdown = Cup * nstar[self.i] / nstar[self.j]
        Cmat[self.i, self.j, :] += Cdown
        Cmat[self.j, self.i, :] += Cup


@dataclass
class Burgess(CollisionalRates):
    '''
    Collisional ionisation from excited states from Burgess & Chidichimo
    (1983, MNRAS 203, 1269).
    Fudge parameter is dimensionless.
    '''
    fudge: float = 1.0

    def __repr__(self):
        s = 'Burgess(j=%d, i=%d, fudge=%g)' % (self.j, self.i, self.fudge)
        return s

    def setup(self, atom):
        i, j = self.i, self.j
        self.i = min(i, j)
        self.j = max(i, j)
        self.atom = atom
        self.iLevel = atom.levels[self.i]
        self.jLevel = atom.levels[self.j]

    def compute_rates(self, atmos: 'Atmosphere', eqPops: 'SpeciesStateTable', Cmat: np.ndarray):
        nstar = eqPops.atomicPops[self.atom.element].nStar
        dE = (self.jLevel.E_SI - self.iLevel.E_SI) / Const.EV
        zz = self.iLevel.stage
        betaB = 0.25 * (np.sqrt((100.0 * zz + 91.0) / (4.0 * zz + 3.0)) - 5.0)
        cbar = 2.3

        dEkT = dE * Const.EV / (Const.KBoltzmann * atmos.temperature)
        dEkT = np.minimum(dEkT, 500)
        invdEkT = 1.0 / dEkT
        wlog = np.log(1.0 + invdEkT)
        wb = wlog**(betaB / (1.0 + invdEkT))
        Cup = 2.1715e-8 * cbar * (13.6/dE)**1.5 * np.sqrt(dEkT) * exp1(dEkT) * wb * atmos.ne * Const.CM_TO_M**3

        Cup *= self.fudge
        Cdown = Cup * nstar[self.i, :] / nstar[self.j, :]

        Cmat[self.j, self.i, :] += Cup
        Cmat[self.i, self.j, :] += Cdown



# NOTE(cmo): It's probably better to write an AR85-CEA function per atomic model, and skip all the series checking stuff
# For Helium, after checking gencol and rh it seems to return 0.0 for helium, so it can probably just be removed from the model
