from typing import Dict, Optional, List, Union, Tuple, TYPE_CHECKING
from copy import copy, deepcopy
from collections import OrderedDict
from xdrlib import Unpacker
from dataclasses import dataclass, field
import numpy as np
import pickle
import lightweaver.constants as Const
from .utils import get_data_path

if TYPE_CHECKING:
    from .atmosphere import Atmosphere

class Element:
    def __init__(self, Z: int):
        self.Z = Z

    def __hash__(self):
        return hash(self.Z)

    def __eq__(self, other):
        return self.Z == other.Z

    def __lt__(self, other):
        if type(other) is Isotope:
            return NotImplemented
        return self.Z < other.Z

    def __repr__(self):
        return 'Element(Z=%d)' % self.Z

    def __str__(self):
        return '%s(Z=%d)' % (self.name, self.Z)

    @property
    def mass(self):
        return PeriodicTable.massData[self.Z]

    @property
    def name(self):
        return PeriodicTable.nameMapping[self.Z]

class Isotope(Element):
    def __init__(self, N, Z):
        super().__init__(Z)
        self.N = N

    def __hash__(self):
        return hash((self.N, self.Z))

    def __eq__(self, other):
        if type(other) is Isotope:
            return self.N == other.N and self.Z == other.Z
        return False

    def __lt__(self, other):
        if type(other) is Isotope:
            return (self.Z, self.N) < (other.Z, other.N)
        elif type(other) is Element:
            return self.Z < other.Z
        return False

    def __repr__(self):
        return 'Isotope(N=%d, Z=%d)' % (self.N, self.Z)

    def __str__(self):
        return '%s(N=%d, Z=%d)' % (self.element_name, self.N, self.Z)

    @property
    def mass(self):
        return PeriodicTable.massData[(self.N, self.Z)]

    @property
    def name(self):
        # NOTE(cmo): Handle H isotopes
        if self.Z == 1 and self.N != 1:
            return PeriodicTable.nameMapping[(self.N, self.Z)]

        baseName = self.element_name
        return '^%d_%s' % (self.N, baseName)

    @property
    def element(self):
        return PeriodicTable[self.Z]

    @property
    def element_mass(self):
        return super().mass

    @property
    def element_name(self):
        return super().name

def load_periodic_table_data():
    path = get_data_path() + 'AtomicMassesNames.pickle'
    with open(path, 'rb') as pkl:
        massData, nameMapping = pickle.load(pkl)

    # NOTE(cmo): Manually change names to D and T for H isotopes
    nameMapping[(2, 1)] = 'D'
    nameMapping['D'] = (2, 1)
    nameMapping[(3, 1)] = 'T'
    nameMapping['T'] = (3, 1)

    # NOTE(cmo): Compute list of isotopes for each eleemnt.
    # This can definitely be done more efficiently, but it probably doesn't
    # matter. (~5 ms, done once, on startup, from timeit)
    isotopes = {}
    elements = {}
    for target in massData:
        if isinstance(target, tuple):
            continue

        element = Element(Z=target)
        elements[target] = element
        isotopes[element] = []
        for key in massData:
            if isinstance(key, int):
                continue

            if key[1] == target:
                N, Z = key[0], element.Z
                iso = Isotope(N=N, Z=Z)
                isotopes[element].append(iso)
                elements[(N, Z)] = iso
    return massData, nameMapping, isotopes, elements

def normalise_atom_name(n: str) -> str:
    strlen = len(n)
    if strlen > 2 or strlen == 0:
        raise ValueError('%s does not represent valid Element name' % n)
    elif strlen == 1:
        return n[0].upper()
    else:
        return n[0].upper() + n[1].lower()

class PeriodicTableData:
    massData, nameMapping, isos, elems = load_periodic_table_data()

    def __getitem__(self, x: Union[str, int, Tuple[int, int], Element]) -> Element:
        if isinstance(x, Element):
            return x

        if isinstance(x, int):
            try:
                return self.elems[x]
            except KeyError:
                raise ValueError('Unable to find Element with Z=%d' % x)

        if isinstance(x, tuple) and all(isinstance(y, int) for y in x):
            try:
                return self.elems[x]
            except KeyError:
                raise ValueError('Unable to find Isotope with (N=%d, Z=%d)' % x)

        if isinstance(x, str):
            x = x.strip()
            if x.startswith('^'):
                terms = x[1:].split('_')
                if len(terms) != 2:
                    raise ValueError('Unable to parse Isotope string %s' % x)
                N = int(terms[0])
                name = normalise_atom_name(terms[1])
                try:
                    Z = self.nameMapping[name]
                    return self.elems[(N, Z)]
                except KeyError:
                    raise ValueError('Unable to find Isotope from string %s' % x)
            else:
                name = normalise_atom_name(x)
                try:
                    Z = self.nameMapping[name]
                    return self.elems[Z]
                except KeyError:
                    raise ValueError('Unable to find Element with name %s' % name)

        raise ValueError('Cannot find element from %s' % repr(x))


    @classmethod
    def get_isotopes(cls, e: Element) -> List[Isotope]:
        if not isinstance(e, Element):
            raise ValueError('Requires Element as first argument, got %s' % repr(e))

        if isinstance(e, Isotope):
            return cls.isos[e.element]

        return cls.isos[e]

    @property
    def elements(self):
        return sorted([e for _, e in self.elems.items() if type(e) is Element])

    @property
    def isotopes(self):
        return sorted([e for _, e in self.elems.items() if type(e) is Isotope])

    @property
    def nuclides(self):
        return self.elements + self.isotopes

PeriodicTable = PeriodicTableData()

class AtomicAbundance:
    def __init__(self, abundanceData: dict=None, abundDex=True, metallicity: float=0.0):
        self.abundance = self.load_default_abundance_data()
        # NOTE(cmo): Default abundances always in dex
        self.dex_to_decimal(self.abundance)

        if abundanceData is not None:
            if abundDex:
                self.dex_to_decimal(abundanceData)
            self.abundance.update(abundanceData)

        self.metallicity = metallicity
        if metallicity != 0.0:
            self.apply_metallicity(self.abundance, metallicity)

        self.isotopeProportions = {iso: v for iso, v in self.abundance.items() if type(iso) is Isotope}

        self.convert_isotopes_to_abundances()
        self.compute_stats()

    def convert_isotopes_to_abundances(self):
        for e in PeriodicTable.elements:
            totalProp = 0.0
            isos = PeriodicTable.get_isotopes(e)
            for iso in isos:
                totalProp += self.abundance[iso]
            for iso in isos:
                if totalProp != 0.0:
                    self.abundance[iso] /= totalProp
                    self.abundance[iso] *= self.abundance[e]

    def compute_stats(self):
        totalAbund = 0.0
        avgMass = 0.0
        for e in PeriodicTable.elements:
            totalAbund += self.abundance[e]
            avgMass += self.abundance[e] * e.mass

        self.totalAbundance = totalAbund
        self.massPerH = avgMass
        self.avgMass = avgMass / totalAbund

    def __getitem__(self, x: Union[str, int, Tuple[int, int], Element]) -> float:
        return self.abundance[PeriodicTable[x]]

    def get_primary_isotope(self, x: Element) -> Isotope:
        isos = PeriodicTable.get_isotopes(x)
        maxIso = isos[0]
        maxAbund = self[maxIso]
        for iso in isos[1:]:
            if (abund := self[iso]) > maxAbund:
                maxAbund = abund
                maxIso = iso
        return iso

    @staticmethod
    def dex_to_decimal(abunds):
        for e, v in abunds.items():
            if type(e) is Element:
                abunds[e] = 10**(v - 12.0)

    @staticmethod
    def apply_metallicity(abunds, metallicity):
        m = 10**metallicity
        for e, v in abunds.items():
            if type(e) is Element and e.Z > 2:
                abunds[e] *= m

    @staticmethod
    def load_default_abundance_data() -> dict:
        """
        Load the default abundances and convert to required format for
        AtomicAbundance class (i.e. dict where dict[Element] = abundance, and
        dict[Isotope] = isotope fraction).
        """
        with open(get_data_path() + 'AbundancesAsplund09.pickle', 'rb') as pkl:
            abundances = pickle.load(pkl)

        lwAbundances = {}
        for ele in abundances:
            Z = ele['elem']['elem']['Z']
            abund = ele['elem']['abundance']
            lwAbundances[PeriodicTable[Z]] = abund

            for iso in ele['isotopes']:
                N = iso['N']
                prop = iso['proportion']
                lwAbundances[PeriodicTable[(N, Z)]] = prop

        for e in PeriodicTable.nuclides:
            if e not in lwAbundances:
                lwAbundances[e] = 0.0

        return lwAbundances

DefaultAtomicAbundance = AtomicAbundance()

@dataclass
class KuruczPf:
    element: Element
    abundance: float
    Tpf: np.ndarray
    pf: np.ndarray
    ionPot: np.ndarray

    def lte_ionisation(self, atmos: 'Atmosphere') -> np.ndarray:
        Nstage = self.ionPot.shape[0]
        Nspace = atmos.depthScale.shape[0]

        C1 = (Const.HPlanck / (2.0 * np.pi * Const.MElectron)) * Const.HPlanck / Const.KBoltzmann

        CtNe = 2.0 * (C1/atmos.temperature)**(-1.5) / atmos.ne
        total = np.ones(Nspace)
        pops = np.zeros((Nstage, Nspace))
        pops[0, :] = 1.0

        Uk = np.interp(atmos.temperature, self.Tpf, self.pf[0, :])

        for i in range(1, Nstage):
            Ukp1 = np.interp(atmos.temperature, self.Tpf, self.pf[i, :])

            pops[i, :] = pops[i-1, :] * CtNe * np.exp(Ukp1 - Uk - self.ionPot[i-1] \
                            / (Const.KBoltzmann * atmos.temperature))
            total += pops[i]

            Ukp1, Uk = Uk, Ukp1

        pops[0, :] = self.abundance * atmos.nHTot / total
        pops[1:,:] *= pops[0, :]

        return pops

    def fjk(self, atmos: 'Atmosphere', k: int) -> Tuple[np.ndarray, np.ndarray]:
        Nspace: int = atmos.depthScale.shape[0]
        T: float = atmos.temperature[k]
        ne: float = atmos.ne[k]

        C1 = (Const.HPlanck / (2.0 * np.pi * Const.MElectron)) * Const.HPlanck / Const.KBoltzmann

        CtNe = 2.0 * (C1/T)**(-1.5) / ne
        Nstage: int = self.ionPot.shape[0]
        fjk = np.zeros(Nstage)
        fjk[0] = 1.0
        dfjk = np.zeros(Nstage)

        # fjk: fractional population of stage j, at atmospheric index k
        # The first stage starts with a "population" of 1, then via Saha we compute the relative populations of the other stages, before dividing by the sum across these

        Uk: float = np.interp(T, self.Tpf, self.pf[0, :])

        for j in range(1, Nstage):
            Ukp1: float = np.interp(T, self.Tpf, self.pf[j, :])

            fjk[j] = fjk[j-1] * CtNe * np.exp(Ukp1 - Uk - self.ionPot[j-1] / (Const.KBoltzmann * T))
            dfjk[j] = -j * fjk[j] / ne

            Uk = Ukp1

        sumF = np.sum(fjk)
        sumDf = np.sum(dfjk)
        fjk /= sumF
        dfjk = (dfjk - fjk * sumDf) / sumF
        return fjk, dfjk

    def fj(self, atmos: 'Atmosphere') -> Tuple[np.ndarray, np.ndarray]:
        Nspace: int = atmos.depthScale.shape[0]
        T = atmos.temperature
        ne = atmos.ne

        C1 = (Const.HPlanck / (2.0 * np.pi * Const.MElectron)) * Const.HPlanck / Const.KBoltzmann

        CtNe = 2.0 * (C1/T)**(-1.5) / ne
        Nstage: int = self.ionPot.shape[0]
        fj = np.zeros((Nstage, Nspace))
        fj[0, :] = 1.0
        dfj = np.zeros((Nstage, Nspace))

        # fjk: fractional population of stage j, at atmospheric index k
        # The first stage starts with a "population" of 1, then via Saha we compute the relative populations of the other stages, before dividing by the sum across these

        Uk = np.interp(T, self.Tpf, self.pf[0, :])

        for j in range(1, Nstage):
            Ukp1 = np.interp(T, self.Tpf, self.pf[j, 0])

            fj[j] = fj[j-1] * CtNe * np.exp(Ukp1 - Uk - self.ionPot[j-1] / (Const.KBoltzmann * T))
            dfj[j] = -j * fj[j] / ne

            Uk[:] = Ukp1

        sumF = np.sum(fj, axis=0)
        sumDf = np.sum(dfj, axis=0)
        fj /= sumF
        dfj = (dfj - fj * sumDf) / sumF
        return fj, dfj


class KuruczPfTable:
    def __init__(self, atomicAbundance: AtomicAbundance=None, kuruczPfPath: str=None):
        if atomicAbundance is None:
            atomicAbundance = DefaultAtomicAbundance
        self.atomicAbundance = atomicAbundance
        # TODO(cmo): replace this with a proper default path:
        kuruczPfPath = get_data_path() + 'pf_Kurucz.input' if kuruczPfPath is None else kuruczPfPath
        with open(kuruczPfPath, 'rb') as f:
            s = f.read()
        u = Unpacker(s)

        # NOTE(cmo): Each of these terms is simply in flat lists indexed by Atomic Number Z-1
        self.Tpf = np.array(u.unpack_array(u.unpack_double))
        stages = []
        pf = []
        ionpot = []
        for i in range(99):
            stages.append(u.unpack_int())
            pf.append(np.array(u.unpack_farray(stages[-1] * self.Tpf.shape[0], u.unpack_double)).reshape(stages[-1], self.Tpf.shape[0]))
            ionpot.append(np.array(u.unpack_farray(stages[-1], u.unpack_double)))

        ionpot = [i * Const.HC / Const.CM_TO_M for i in ionpot]
        pf = [np.log(p) for p in pf]
        self.pf = pf
        self.ionpot = ionpot

    def __getitem__(self, x: Element) -> KuruczPf:
        if type(x) is Isotope:
            raise ValueError('Isotopes not supported by KuruczPf')

        zm = x.Z - 1
        return KuruczPf(element=x, abundance=self.atomicAbundance[x], Tpf=self.Tpf, pf=self.pf[zm], ionPot=self.ionpot[zm])