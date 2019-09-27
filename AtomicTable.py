from typing import Dict, Optional, List, TYPE_CHECKING
from copy import copy, deepcopy
from collections import OrderedDict
from xdrlib import Unpacker
from dataclasses import dataclass, field
import numpy as np
import Constants as Const
from Atmosphere import Atmosphere
from scipy.interpolate import interp1d

if TYPE_CHECKING:
   from ComputationalAtom import ComputationalAtom

AtomicWeights: OrderedDict = \
   OrderedDict([
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


# def weight_amu(name: str):
#    name = name.upper()
#    if len(name) == 1:
#       name += ' '
#    return AtomicWeights[name]

## NA   6.73  ## Value needed for small Na I atom
# O   8.93  # original
# C   8.60  # original
# FE   7.67  #original
AtomicAbundances: OrderedDict = \
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

 # TODO(cmo): Make all of this into a class, so that convert_abundances is called upon construction, so that we could ostensibly have multiple AtomicTables for different contexts
 # This should also make it easier to convert into the C-struct if needed --  though hopefully it won't be

# def convert_abundances():
#    if set(AtomicWeights.keys()) != set(AtomicAbundances.keys()):
#       raise ValueError('AtomicWeghts and AtomicAbundances keys differ (Problem keys: %s)' % repr(set(AtomicWeights.keys()) - set(AtomicAbundances.keys())))
#    global _OriginalAtomicAbundances 
#    _OriginalAtomicAbundances = copy(AtomicAbundances)
#    if AtomicAbundances['H '] == 12.0:
#       for k, v in AtomicAbundances.items():
#          AtomicAbundances[k] = 10**(v - 12.0)

# def override_abundance(element, value, dex=True):
#    if dex:
#       AtomicAbundances[element] = 10**(value - 12.0)
#    else:
#       AtomicAbundances[element] = value

# convert_abundances()

# TODO(cmo): Read the Kurucz pf stuff here too
# TODO(cmo): Try and move everything that depends on the AtomicTable and pf stuff to P/Cython. The sticking point there is the molecule ICE stuff, I think. We might be best off just keeping a version of the old struct around for that. But the problem there is that I think it adjusts the populations of the active atoms due to computing ICE -- it does, all the populations relating to molecules, in fact (duh). I mean, on a theoretical level all of that might be better in Cython, but much work. Alternatively, we might be able to bridge the two sufficiently transparently


# def read_abundance(metallicity):

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
   atom: Optional['ComputationalAtom'] = None

   def __hash__(self):
      return hash(repr(self))

   def __cmp__(self, other):
      return hash(self) == hash(other)

   def __repr__(self):
      return 'Element(name=\'%s\', weight=%e, abundance=%e)' % (self.name, self.weight, self.abundance)

   def lte_populations(self, atmos: Atmosphere) -> np.ndarray:
      Nspace = atmos.depthScale.shape[0]

      C1 = (Const.HPLANCK / (2.0 * np.pi * Const.M_ELECTRON)) * Const.HPLANCK / Const.KBOLTZMANN
      
      CtNe = 2.0 * (C1/atmos.temperature)**(-1.5) / atmos.ne
      total = np.ones_like(CtNe)
      pops = np.zeros((self.ionpot.shape[0], Nspace))
      pops[0, :] = 1.0

      Uk = interp1d(self.Tpf, self.pf[0])(atmos.temperature)

      Nstage = self.ionpot.shape[0]
      for i in range(1, Nstage):
         Ukp1 = interp1d(self.Tpf, self.pf[i])(atmos.temperature)

         pops[i, :] = pops[i-1, :] * CtNe * np.exp(Ukp1 - Uk - self.ionpot[i-1]) \
                        / Const.KBOLTZMANN * atmos.temperature
         total += pops[i]

         Ukp1, Uk = Uk, Ukp1

      pops[0] = self.abundance * atmos.nHTot / total
      for i in range(1, Nstage):
         pops[i] *= pops[0]

      return pops

   def fjk(self, atmos, k) -> np.ndarray:
      Nspace = atmos.depthScale.shape[0]
      T = atmos.temperature[k]
      ne = atmos.ne[k]

      C1 = (Const.HPLANCK / (2.0 * np.pi * Const.M_ELECTRON)) * Const.HPLANCK / Const.KBOLTZMANN
      
      CtNe = 2.0 * (C1/T)**(-1.5) / ne
      Nstage = self.ionpot.shape[0]
      fjk = np.zeros(Nstage)
      fjk[0] = 1.0
      dfjk = np.zeros(Nstage)
      sum1 = 0.0
      sum2 = 0.0

      Uk = interp1d(self.Tpf, self.pf[0])(T)

      for j in range(1, Nstage):
         Ukp1 = interp1d(self.Tpf, self.pf[j])(T)

         fjk[j] = fjk[j-1] * CtNe * np.exp(Ukp1 - Uk - self.ionpot[j-1] / (Const.KBOLTZMANN * T))
         dfjk[j] = -j * fjk[j] / ne

         sum1 += fjk[j]
         sum2 += dfjk[j]
         Uk = Ukp1

      fjk /= sum1
      dfjk = (dfjk - fjk * sum2) / sum1
      return fjk, dfjk


@dataclass
class LtePopulations:
   atomicTable: 'AtomicTable'
   atmosphere: Atmosphere
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
         raise ValueError('AtomicWeghts and AtomicAbundances keys differ (Problem keys: %s)' % repr(set(AtomicWeights.keys()) - set(AtomicAbundances.keys())))

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
      kuruczPfPath = '../Atoms/pf_Kurucz.input' if kuruczPfPath is None else kuruczPfPath
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

   def lte_populations(self, atmos: Atmosphere) -> LtePopulations:
      pops = []
      for ele in self.elements:
         pops.append(ele.lte_populations(atmos))

      return LtePopulations(self, atmos, pops)



      
         





