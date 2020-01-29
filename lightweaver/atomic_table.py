from typing import Dict, Optional, List, TYPE_CHECKING
from copy import copy, deepcopy
from collections import OrderedDict
from xdrlib import Unpacker
from dataclasses import dataclass, field
import numpy as np
import lightweaver.constants as Const
from scipy.interpolate import interp1d
from numba import jit
from .utils import ConvergenceError, get_data_path

if TYPE_CHECKING:
    from .atomic_model import AtomicModel
    from .atmosphere import Atmosphere

AtomicWeights = OrderedDict([
    ("H ", 1.008),   ("HE", 4.003),   ("LI", 6.939),   ("BE", 9.013),
    ("B ", 10.810),  ("C ", 12.010),  ("N ", 14.010),  ("O ", 16.000),
    ("F ", 19.000),  ("NE", 20.180),  ("NA", 22.990),  ("MG", 24.310),
    ("AL", 26.980),  ("SI", 28.090),  ("P ", 30.980),  ("S ", 32.070),
    ("CL", 35.450),  ("AR", 39.950),  ("K ", 39.100),  ("CA", 40.080),
    ("SC", 44.960),  ("TI", 47.900),  ("V ", 50.940),  ("CR", 52.000),
    ("MN", 54.940),  ("FE", 55.850),  ("CO", 58.940),  ("NI", 58.710),
    ("CU", 63.550),  ("ZN", 65.370),  ("GA", 69.720),  ("GE", 72.600),
    ("AS", 74.920),  ("SE", 78.960),  ("BR", 79.910),  ("KR", 83.800),
    ("RB", 85.480),  ("SR", 87.630),  ("Y ", 88.910),  ("ZR", 91.220),
    ("NB", 92.910),  ("MO", 95.950),  ("TC", 99.000),  ("RU", 101.100),
    ("RH", 102.900), ("PD", 106.400), ("AG", 107.900), ("CD", 112.400),
    ("IN", 114.800), ("SN", 118.700), ("SB", 121.800), ("TE", 127.600),
    ("I ", 126.900), ("XE", 131.300), ("CS", 132.900), ("BA", 137.400),
    ("LA", 138.900), ("CE", 140.100), ("PR", 140.900), ("ND", 144.300),
    ("PM", 147.000), ("SM", 150.400), ("EU", 152.000), ("GD", 157.300),
    ("TB", 158.900), ("DY", 162.500), ("HO", 164.900), ("ER", 167.300),
    ("TM", 168.900), ("YB", 173.000), ("LU", 175.000), ("HF", 178.500),
    ("TA", 181.000), ("W ", 183.900), ("RE", 186.300), ("OS", 190.200),
    ("IR", 192.200), ("PT", 195.100), ("AU", 197.000), ("HG", 200.600),
    ("TL", 204.400), ("PB", 207.200), ("BI", 209.000), ("PO", 210.000),
    ("AT", 211.000), ("RN", 222.000), ("FR", 223.000), ("RA", 226.100),
    ("AC", 227.100), ("TH", 232.000), ("PA", 231.000), ("U ", 238.000),
    ("NP", 237.000), ("PU", 244.000), ("AM", 243.000), ("CM", 247.000),
    ("BK", 247.000), ("CF", 251.000), ("ES", 254.000)])

def atomic_weight_sort(atom):
    name = atom.name.upper().ljust(2)
    return AtomicWeights[name]

# def weight_amu(name: str):
#    name = name.upper()
#    if len(name) == 1:
#       name += ' '
#    return AtomicWeights[name]

## NA   6.73  ## Value needed for small Na I atom
# O   8.93  # original
# C   8.60  # original
# FE   7.67  #original
AtomicAbundances = \
OrderedDict([
("H ", 12.00),
("HE", 10.99),
("LI",  1.16),
("BE",  1.15),
("B ",  2.60),
("C ",  8.39),  ## Asplund et al 2005, A&A, 431, 693-705
("N ",  8.00),
("O ",  8.66),  ## Asplund et al 2004, A&A, 417, 751
("F ",  4.40),  ## for Kurucz model T4250, vt=0.5
("NE",  8.09),
("NA",  6.33),
("MG",  7.58),
("AL",  6.47),
("SI",  7.55),
("P ",  5.45),
("S ",  7.21),
("CL",  5.50),
("AR",  6.56),
("K ",  5.12),
("CA",  6.36),
("SC",  3.10),
("TI",  4.99),
("V ",  4.00),
("CR",  5.67),
("MN",  5.39),
("FE",  7.44),  ## Asplund et al 2000, A&A, 359, 743
("CO",  4.92),
("NI",  6.25),
("CU",  4.21),
("ZN",  4.60),
("GA",  2.88),
("GE",  3.41),
("AS",  2.37),
("SE",  3.35),
("BR",  2.63),
("KR",  3.23),
("RB",  2.60),
("SR",  2.90),
("Y ",  2.24),
("ZR",  2.60),
("NB",  1.42),
("MO",  1.92),
("TC", -7.96),
("RU",  1.84),
("RH",  1.12),
("PD",  1.69),
("AG",  0.94),
("CD",  1.86),
("IN",  1.66),
("SN",  2.00),
("SB",  1.00),
("TE",  2.24),
("I ",  1.51),
("XE",  2.23),
("CS",  1.12),
("BA",  2.13),
("LA",  1.22),
("CE",  1.55),
("PR",  0.71),
("ND",  1.50),
("PM", -7.96),
("SM",  1.00),
("EU",  0.51),
("GD",  1.12),
("TB", -0.10),
("DY",  1.10),
("HO",  0.26),
("ER",  0.93),
("TM",  0.00),
("YB",  1.08),
("LU",  0.76),
("HF",  0.88),
("TA",  0.13),
("W ",  1.11),
("RE",  0.27),
("OS",  1.45),
("IR",  1.35),
("PT",  1.80),
("AU",  1.01),
("HG",  1.09),
("TL",  0.90),
("PB",  1.85),
("BI",  0.71),
("PO", -7.96),
("AT", -7.96),
("RN", -7.96),
("FR", -7.96),
("RA", -7.96),
("AC", -7.96),
("TH",  0.12),
("PA", -7.96),
("U ", -0.47),
("NP", -7.96),
("PU", -7.96),
("AM", -7.96),
("CM", -7.96),
("BK", -7.96),
("CF", -7.96),
("ES", -7.96)])

def read_pf(path):
    with open(path, 'rb') as f:
        s = f.read()
    u = Unpacker(s)
    Tpf = u.unpack_array(u.unpack_double)
    ptis = []
    stages = []
    pf = []
    ionpot = []
    for i in range(len(AtomicWeights)):
        ptis.append(u.unpack_int())
        stages.append(u.unpack_int())
        pf.append(u.unpack_farray(stages[-1] * len(Tpf), u.unpack_double))
        ionpot.append(u.unpack_farray(stages[-1], u.unpack_double))

    return {'Tpf': Tpf, 'stages': stages, 'pf': pf, 'ionpot': ionpot}

@dataclass
class Element:
    name: str
    weight: float
    abundance: float
    ionpot: np.ndarray
    Tpf: np.ndarray
    pf: np.ndarray

    def __post_init__(self):
        Nstage = self.ionpot.shape[0]

    def __hash__(self):
        return hash(repr(self))

    def __cmp__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        return 'Element(name=\'%s\', weight=%e, abundance=%e)' % (self.name, self.weight, self.abundance)

    def lte_populations(self, atmos: 'Atmosphere') -> np.ndarray:
        Nstage = self.ionpot.shape[0]
        Nspace = atmos.depthScale.shape[0]

        C1 = (Const.HPlanck / (2.0 * np.pi * Const.MElectron)) * Const.HPlanck / Const.KBoltzmann
        
        CtNe = 2.0 * (C1/atmos.temperature)**(-1.5) / atmos.ne
        total = np.ones(Nspace)
        pops = np.zeros((Nstage, Nspace))
        pops[0, :] = 1.0

        Uk = np.interp(atmos.temperature, self.Tpf, self.pf[0, :])

        for i in range(1, Nstage):
            Ukp1 = np.interp(atmos.temperature, self.Tpf, self.pf[i, :])

            pops[i, :] = pops[i-1, :] * CtNe * np.exp(Ukp1 - Uk - self.ionpot[i-1] \
                            / (Const.KBoltzmann * atmos.temperature))
            total += pops[i]

            Ukp1, Uk = Uk, Ukp1

        pops[0, :] = self.abundance * atmos.nHTot / total
        pops[1:,:] *= pops[0, :]

        return pops

   # @jit
    def fjk(self, atmos, k):
        Nspace: int = atmos.depthScale.shape[0]
        T: float = atmos.temperature[k]
        ne: float = atmos.ne[k]

        C1 = (Const.HPlanck / (2.0 * np.pi * Const.MElectron)) * Const.HPlanck / Const.KBoltzmann
        
        CtNe = 2.0 * (C1/T)**(-1.5) / ne
        Nstage: int = self.ionpot.shape[0]
        fjk = np.zeros(Nstage)
        fjk[0] = 1.0
        dfjk = np.zeros(Nstage)

        # fjk: fractional population of stage j, at atmospheric index k
        # The first stage starts with a "population" of 1, then via Saha we compute the relative populations of the other stages, before dividing by the sum across these

        Uk: float = np.interp(T, self.Tpf, self.pf[0, :])

        for j in range(1, Nstage):
            Ukp1: float = np.interp(T, self.Tpf, self.pf[j, :])

            fjk[j] = fjk[j-1] * CtNe * np.exp(Ukp1 - Uk - self.ionpot[j-1] / (Const.KBoltzmann * T))
            dfjk[j] = -j * fjk[j] / ne

            Uk = Ukp1

        sumF = np.sum(fjk)
        sumDf = np.sum(dfjk)
        fjk /= sumF
        dfjk = (dfjk - fjk * sumDf) / sumF
        return fjk, dfjk

    def fj(self, atmos):
        Nspace: int = atmos.depthScale.shape[0]
        T = atmos.temperature
        ne = atmos.ne

        C1 = (Const.HPlanck / (2.0 * np.pi * Const.MElectron)) * Const.HPlanck / Const.KBoltzmann
        
        CtNe = 2.0 * (C1/T)**(-1.5) / ne
        Nstage: int = self.ionpot.shape[0]
        fj = np.zeros((Nstage, Nspace))
        fj[0, :] = 1.0
        dfj = np.zeros((Nstage, Nspace))

        # fjk: fractional population of stage j, at atmospheric index k
        # The first stage starts with a "population" of 1, then via Saha we compute the relative populations of the other stages, before dividing by the sum across these

        Uk = np.interp(T, self.Tpf, self.pf[0, :])

        for j in range(1, Nstage):
            Ukp1 = np.interp(T, self.Tpf, self.pf[j, 0])

            fj[j] = fj[j-1] * CtNe * np.exp(Ukp1 - Uk - self.ionpot[j-1] / (Const.KBoltzmann * T))
            dfj[j] = -j * fj[j] / ne

            Uk[:] = Ukp1

        sumF = np.sum(fj, axis=0)
        sumDf = np.sum(dfj, axis=0)
        fj /= sumF
        dfj = (dfj - fj * sumDf) / sumF
        return fj, dfj


@dataclass
class LtePopulations:
    atomicTable: 'AtomicTable'
    atmosphere: 'Atmosphere'
    populations: List[np.ndarray]

    def __getitem__(self, name: str) -> np.ndarray:
        name = name.upper()
        if len(name) == 1:
            name += ' '
        return self.populations[self.atomicTable.indices[name]]
    

class AtomicTable:
    def __init__(self, kuruczPfPath: Optional[str]=None, metallicity: float=0.0, 
                        abundances: Dict=None, abundDex: bool=True):
        if set(AtomicWeights.keys()) != set(AtomicAbundances.keys()):
            raise ValueError('AtomicWeights and AtomicAbundances keys differ (Problem keys: %s)' % repr(set(AtomicWeights.keys()) - set(AtomicAbundances.keys())))

        self.indices = OrderedDict(zip(AtomicWeights.keys(), range(len(AtomicWeights))))

        # Convert abundances and overwrite any provided secondary abundances
        self.abund = deepcopy(AtomicAbundances)
        if self.abund['H '] == 12.0:
            for k, v in self.abund.items():
                self.abund[k] = 10**(v - 12.0)

        if abundances is not None:
            if abundDex:
                for k, v in abundances.items():
                    abundances[k] = 10**(v - 12.0)
            for k, v in abundances.items():
                self.abund[k] = v

        metallicity = 10**metallicity
        for k, v in self.abund.items():
            if k != 'H ':
                self.abund[k] = v*metallicity


        # TODO(cmo): replace this with a proper default path
        kuruczPfPath = get_data_path() + 'pf_Kurucz.input' if kuruczPfPath is None else kuruczPfPath
        with open(kuruczPfPath, 'rb') as f:
            s = f.read()
        u = Unpacker(s)

        self.Tpf = np.array(u.unpack_array(u.unpack_double))
        ptIndex = [] # Index in the periodic table (fortran based, so +1) -- could be used for validation
        stages = []
        pf = []
        ionpot = []
        for i in range(len(AtomicWeights)):
            ptIndex.append(u.unpack_int())
            stages.append(u.unpack_int())
            pf.append(np.array(u.unpack_farray(stages[-1] * self.Tpf.shape[0], u.unpack_double)).reshape(stages[-1], self.Tpf.shape[0]))
            ionpot.append(np.array(u.unpack_farray(stages[-1], u.unpack_double)))

        ionpot = [i * Const.HC / Const.CM_TO_M for i in ionpot]
        pf = [np.log(p) for p in pf]

        totalAbund = 0.0
        avgWeight = 0.0
        self.elements: List[Element] = []
        for k, v in AtomicWeights.items():
            i = self.indices[k]
            ele = Element(k, v, self.abund[k], ionpot[i], self.Tpf, pf[i])
            self.elements.append(ele)
            totalAbund += ele.abundance
            avgWeight += ele.abundance * ele.weight

        self.totalAbundance = totalAbund
        self.weightPerH = avgWeight
        self.avgMolWeight = avgWeight / totalAbund

    def __getitem__(self, name: str) -> Element:
        name = name.upper()
        if len(name) == 1:
            name += ' '
        return self.elements[self.indices[name]]

    def lte_populations(self, atmos: 'Atmosphere') -> LtePopulations:
        pops = []
        for ele in self.elements:
            pops.append(ele.lte_populations(atmos))

        return LtePopulations(self, atmos, pops)

    def compute_ne_k(self, atmos, k):
        # Can add a "fromScratch" where we start with ne=nHii
        neOld = atmos.ne[k]

        C1 = (Const.HPlanck / (2.0 * np.pi * Const.MElectron)) * (Const.HPlanck / Const.KBoltzmann)

        nIter = 1
        NmaxIter = 100
        MaxError = 1e-2
        while nIter < NmaxIter:
            # Electrons per H
            error = neOld / atmos.nHTot[k]
            total = 0.0

            for i, ele in enumerate(self.elements):
                # Fractional stage pops and derivate wrt ne
                fjk, dfjk = ele.fjk(atmos, k)

                if ele.name.startswith('H'):
                # Add H- effects
                    PhiHmin = 0.25 * (C1 / atmos.temperature[k])**1.5 \
                                * np.exp(Const.E_ION_HMIN / (Const.KBoltzmann * atmos.temperature[k]))
                    error += neOld * fjk[0] * PhiHmin
                    total -= (fjk[0] + neOld * dfjk[0]) * PhiHmin

                for j in range(1, len(ele.ionpot)):
                    electronContribution = ele.abundance * j
                    error -= electronContribution * fjk[j]
                    total += electronContribution * dfjk[j]

            # Quasi-Newton iteration
            # atmos.nHTot * total is the sum of the derivative of all of the atomic populations wrt ne, weighted by the ionisation stage
            # Obviously error tends to 0 as this converges -- we're solving f(x) = 0
            atmos.ne[k] = neOld - atmos.nHTot[k] * error / (1.0 - atmos.nHTot[k] * total)
            dne = np.abs((atmos.ne[k] - neOld) / neOld)
            neOld = atmos.ne[k]

            if dne < MaxError:
                break

            nIter += 1

        if dne > MaxError:
            raise ConvergenceError("Electron iteration did not converge at point %d" % k)
        

    def compute_ne(self, atmos):
        # Can add a "fromScratch" where we start with ne=nHii
        neOld = atmos.ne.copy()

        C1 = (Const.HPlanck / (2.0 * np.pi * Const.MElectron)) * (Const.HPlanck / Const.KBoltzmann)

        nIter = 1
        NmaxIter = 100
        MaxError = 1e-2
        while nIter < NmaxIter:
            # Electrons per H
            error = neOld / atmos.nHTot
            total = np.zeros_like(error)

            for i, ele in enumerate(self.elements):
                # Fractional stage pops and derivate wrt ne
                fj, dfj = ele.fj(atmos)

                if ele.name.startswith('H'):
                    # Add H- effects
                    PhiHmin = 0.25 * (C1 / atmos.temperature)**1.5 \
                                * np.exp(Const.E_ION_HMIN / (Const.KBoltzmann * atmos.temperature))
                    error += neOld * fj[0] * PhiHmin
                    total -= (fj[0] + neOld * dfj[0]) * PhiHmin

                for j in range(1, len(ele.ionpot)):
                    electronContribution = ele.abundance * j
                    error -= electronContribution * fj[j]
                    total += electronContribution * dfj[j]

            # Quasi-Newton iteration
            # atmos.nHTot * total is the sum of the derivative of all of the atomic populations wrt ne, weighted by the ionisation stage
            # Obviously error tends to 0 as this converges -- we're solving f(x) = 0
            # Krylov or somtheing?
            atmos.ne[:] = neOld - atmos.nHTot * error / (1.0 - atmos.nHTot * total)
            dne = np.max(np.abs((atmos.ne - neOld) / neOld))
            neOld[:] = atmos.ne

            if dne < MaxError:
                break

            nIter += 1

        if dne > MaxError:
            raise ConvergenceError("Electron iteration did not converge")

_StandardAtomicTable: AtomicTable
def get_global_atomic_table() -> AtomicTable:
    global _StandardAtomicTable
    try:
        return _StandardAtomicTable
    except NameError:
        _StandardAtomicTable = AtomicTable()
    return _StandardAtomicTable

def set_global_atomic_table(table: AtomicTable):
    global _StandardAtomicTable
    _StandardAtomicTable = table