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
    '''
    A simple value comparable description of an element (just proton number
    Z), that can be quickly and easily compared, whilst allowing access to
    things from the periodic table.
    '''
    def __init__(self, Z: int):
        self.Z = Z

    def __hash__(self):
        return hash(self.Z)

    def __eq__(self, other):
        return self.Z == other.Z

    def __lt__(self, other):
        if isinstance(other, Isotope):
            return self.Z <= other.Z
        return self.Z < other.Z

    def __repr__(self):
        return 'Element(Z=%d)' % self.Z

    def __str__(self):
        return '%s(Z=%d)' % (self.name, self.Z)

    @property
    def mass(self):
        '''
        Returns the mass of the element in AMU.
        '''
        return PeriodicTable.massData[self.Z]

    @property
    def name(self):
        '''
        Returns the name of the element as a string.
        '''
        return PeriodicTable.nameMapping[self.Z]

class Isotope(Element):
    '''
    A simple value comparable isotope description, inheriting from Element.
    '''
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
        if isinstance(other, Isotope):
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
        '''
        Returns the mass of the isotope in AMU.
        '''
        return PeriodicTable.massData[(self.N, self.Z)]

    @property
    def name(self):
        '''
        Returns the name of the isotope as a string.
        '''
        # NOTE(cmo): Handle H isotopes
        if self.Z == 1 and self.N != 1:
            return PeriodicTable.nameMapping[(self.N, self.Z)]

        baseName = self.element_name
        return '^%d_%s' % (self.N, baseName)

    @property
    def element(self):
        '''
        Returns the underlying Element of which this isotope is a family
        member.
        '''
        return PeriodicTable[self.Z]

    @property
    def element_mass(self):
        '''
        Returns the average mass of the element.
        '''
        return super().mass

    @property
    def element_name(self):
        '''
        Returns the name of the Element as a string.
        '''
        return super().name

def load_periodic_table_data():
    '''
    Internal use function to load data from the AtomicMassesNames.pickle data
    file.
    '''
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
    '''
    Normalises Element names to be two characters long with an uppercase
    first letter, and lower case second, or a space in the case of single
    character names.

    Parameters
    ----------
    n : str
        The name to normalise.

    Returns
    -------
    result : str
        The normalised name.
    '''
    strlen = len(n)
    if strlen > 2 or strlen == 0:
        raise ValueError('%s does not represent valid Element name' % n)
    elif strlen == 1:
        return n[0].upper()
    else:
        return n[0].upper() + n[1].lower()

class PeriodicTableData:
    '''
    Container and accessor for the periodic table data. Not intended to be
    instantiated by users, instead use the pre-instantiated PeriodicTable
    instance.
    '''
    massData, nameMapping, isos, elems = load_periodic_table_data()

    def __getitem__(self, x: Union[str, int, Tuple[int, int], Element]) -> Element:
        '''
        Allows access to the associated Element or Isotope via a variety of means.
        If input is
            - an Element or Isotope, then this function returns it.
            - an int, then this function returns the element with associated
              proton number.
            - a tuple of ints, then this function returns the element with
              associated (Z, N)
            - a str starting with '^' then the str is parsed as an isotope in
              the form ^N_AtomName' and the associated Isotope is returned.
            - any other str, then the str is parsed as a one or two character
              atom identifier (e.g. H or Ca) and the associated Element is
              returned.
        '''
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
        '''
        Get all isotopes associated with a certain Element.
        '''
        if not isinstance(e, Element):
            raise ValueError('Requires Element as first argument, got %s' % repr(e))

        if isinstance(e, Isotope):
            return cls.isos[e.element]

        return cls.isos[e]

    @property
    def elements(self):
        '''
        Return a sorted list of Elements by proton number Z.
        '''
        return sorted([e for _, e in self.elems.items() if type(e) is Element])

    @property
    def isotopes(self):
        '''
        Return a sorted list of Isotopes by proton number Z.
        '''
        return sorted([e for _, e in self.elems.items() if type(e) is Isotope])

    @property
    def nuclides(self):
        '''
        Return a list of all nuclides (Elements and Isotopes).
        '''
        return self.elements + self.isotopes

PeriodicTable = PeriodicTableData()

class AtomicAbundance:
    '''
    Container and accessor for atomic abundance data. This can be
    instantiated with a subset of atomic abundances, which will be used in
    conjunction with the the default values for non-specified elements.

    Parameters
    ----------
    abundanceData : dict, optional
        Contains the abundance data to override. For elements this should be
        dict[Element] = abundance, and for isotopes dict[Isotope] = isotope
        fraction.
    abundDex : bool, optional
        Whether the supplied abundance is in dex (with Hydrogen abundance of
        12.0) or in relative Hydrogen abundance (default: True i.e. in dex).
    metallicity : float, optional
        Enhance the metallic abundance by a factor of 10**metallicity,
        (default: 0.0).
    '''
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
        '''
        Converts the isotope fractions to relative Hydrogen abundance.
        '''
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
        '''
        Compute the total abundance (totalAbundance), mass per H atom
        (massPerH), and average mass per Element (avgMass).
        '''
        totalAbund = 0.0
        avgMass = 0.0
        for e in PeriodicTable.elements:
            totalAbund += self.abundance[e]
            avgMass += self.abundance[e] * e.mass

        self.totalAbundance = totalAbund
        self.massPerH = avgMass
        self.avgMass = avgMass / totalAbund

    def __getitem__(self, x: Union[str, int, Tuple[int, int], Element]) -> float:
        '''
        Returns the abundance of the requested Element or Isotope. All forms
        of describing these are accepted as the PeriodicTable is invoked.
        '''
        return self.abundance[PeriodicTable[x]]

    def get_primary_isotope(self, x: Element) -> Isotope:
        '''
        Returns the Isotope with the highest abundance of a particular
        Element.
        '''
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
        '''
        Used to convert from absolute abundance in dex to relative fractional
        abundance.
        '''
        for e, v in abunds.items():
            if type(e) is Element:
                abunds[e] = 10**(v - 12.0)

    @staticmethod
    def apply_metallicity(abunds, metallicity):
        '''
        Used to adjust the metallicity of the abundances, by a fraction
        10**metallicity.
        '''
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
    '''
    Storage and functions relating to Bob Kurucz's partition functions.
    Based on the data used in RH.

    Attributes
    ----------
    element : Element
        The element associated with this partition funciton.
    abundance : float
        The abundance of this element.
    Tpf : np.ndarray
        The temperature grid on which the partition function is defined.
    pf : np.ndarray
        The partition function data.
    ionPot : np.ndarray
        The ionisation potential of each level.
    '''
    element: Element
    abundance: float
    Tpf: np.ndarray
    pf: np.ndarray
    ionPot: np.ndarray

    def lte_ionisation(self, atmos: 'Atmosphere') -> np.ndarray:
        '''
        Compute the population of the species in each ionisation
        stage in a given atmosphere.

        Parameters
        ----------
        atmos : Atmosphere
            The atmosphere in which to compute the populations.

        Returns
        -------
        pops : np.ndarray
            The LTE ionisation populations [Nstage x Nspace].
        '''
        Nstage = self.ionPot.shape[0]
        Nspace = atmos.Nspace

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
        '''
        Compute the fractional population of the species in each ionisation
        stage and partial derivative wrt n_e at one point in a given
        atmosphere.

        Parameters
        ----------
        atmos : Atmosphere
            The atmosphere in which to compute the populations.
        k : int
            The spatial index at which to compute the populations

        Returns
        -------
        fj : np.ndarray
            The fractional populations [Nstage].
        dfj : np.ndarray
            The derivatives of the fractional populations [Nstage].
        '''
        Nspace: int = atmos.Nspace
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
        '''
        Compute the fractional population of the species in each ionisation
        stage and partial derivative wrt n_e for each location in a given
        atmosphere.

        Parameters
        ----------
        atmos : Atmosphere
            The atmosphere in which to compute the populations.

        Returns
        -------
        fj : np.ndarray
            The fractional populations [Nstage x Nspace].
        dfj : np.ndarray
            The derivatives of the fractional populations [Nstage x Nspace].
        '''
        Nspace: int = atmos.Nspace
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
    '''
    Container for all of the Kurucz partition function data, allowing
    different paths and AtomicAbundances to be used. Serves to construct the
    KuruczPf objects if used.

    Parameters
    ----------
    atomicAbundance : AtomicAbundance, optional
        The abundance data to use, if non-standard.
    kuruczPfPath : str
        The path to the Kurucz parition function data in RH's XDR format, if
        non-standard.
    '''
    def __init__(self, atomicAbundance: AtomicAbundance=None, kuruczPfPath: str=None):
        if atomicAbundance is None:
            atomicAbundance = DefaultAtomicAbundance
        self.atomicAbundance = atomicAbundance
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
            z = u.unpack_int()
            stages.append(u.unpack_int())
            pf.append(np.array(u.unpack_farray(stages[-1] * self.Tpf.shape[0], u.unpack_double)).reshape(stages[-1], self.Tpf.shape[0]))
            ionpot.append(np.array(u.unpack_farray(stages[-1], u.unpack_double)))

        ionpot = [i * Const.HC / Const.CM_TO_M for i in ionpot]
        pf = [np.log(p) for p in pf]
        self.pf = pf
        self.ionpot = ionpot

    def __getitem__(self, x: Element) -> KuruczPf:
        '''
        Used to construct the partition function object for the requested
        element.
        '''
        if type(x) is Isotope:
            raise ValueError('Isotopes not supported by KuruczPf')

        zm = x.Z - 1
        return KuruczPf(element=x, abundance=self.atomicAbundance[x], Tpf=self.Tpf, pf=self.pf[zm], ionPot=self.ionpot[zm])