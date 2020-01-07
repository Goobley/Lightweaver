from parse import parse
import lightweaver.constants as Const
from typing import Tuple, Set, List, TYPE_CHECKING, Optional
from .atomic_table import LtePopulations, AtomicTable, Element, get_global_atomic_table
from .atmosphere import Atmosphere
import numpy as np
from numpy.linalg import solve
from numba import njit
from dataclasses import dataclass
from collections import OrderedDict

def get_next_line(data):
    if len(data) == 0:
        return None
    for i, d in enumerate(data):
        if d.strip().startswith('#') or d.strip() == '':
            # print('Skipping %s' % d)
            continue
        # print('Accepting %s' % d)
        break
    d = data[i]
    if i == len(data) - 1:
        data[:] = []
        return d.strip()
    data[:] = data[i+1:]
    return d.strip()

def get_constituent(name: str) -> Tuple[int, str]:
    res = parse('{:d}{!s}', name)
    if res is None:
        constituent = (1, name)
    else:
        constituent = (res[0], res[1])
    return constituent

def equilibrium_constant_kurucz_70(tempRange, mk, Ediss, eqc):
    minTemp = tempRange[0]
    maxTemp = tempRange[1]
    kB = Const.KBoltzmann
    CM_TO_M = Const.CM_TO_M

    @njit('float64(float64)')
    def kurucz_70(T):
        if T < minTemp or T > maxTemp:
            return 0.0

        kT = kB * T
        eq = eqc[0]
        for i in range(1, eqc.shape[0]):
            eq = eq * T + eqc[i]
        arg = Ediss / kT + eq - 1.5 * mk * np.log(T)
        eq = np.exp(arg)
        return eq * (CM_TO_M**3)**mk
    return kurucz_70

def equilibrium_constant_kurucz_85(tempRange, mk, Ediss, eqc):
    minTemp = tempRange[0]
    maxTemp = tempRange[1]
    kB = Const.KBoltzmann
    CM_TO_M = Const.CM_TO_M

    @njit('float64(float64)')
    def kurucz_85(T):
        if T < minTemp or T > maxTemp:
            return 0.0

        t = T * 1e-4
        kT = kB * T
        eq = eqc[0]
        for i in range(1, eqc.shape[0]):
            eq = eq * t + eqc[i]
        eq = np.exp(Ediss / kT + eq - 1.5 * mk * np.log(T))
        return eq * (CM_TO_M**3)**mk
    return kurucz_85

def equilibrium_constant_sauval_tatum(tempRange, Ediss, eqc):
    minTemp = tempRange[0]
    maxTemp = tempRange[1]
    kB = Const.KBoltzmann
    CM_TO_M = Const.CM_TO_M
    THETA0 = Const.Theta0
    Ediss = Ediss / Const.EV

    @njit('float64(float64)')
    def sauval_tatum(T):
        if T < minTemp or T > maxTemp:
            return 0.0

        theta = THETA0 / T
        t = np.log10(theta)
        kT = kB * T
        
        eq = eqc[0]
        for i in range(1, eqc.shape[0]):
            eq = eq * t + eqc[i]
        eq = 10**(Ediss * theta - eq) * kT
        return eq
    return sauval_tatum


class Molecule:
    def __init__(self, filePath: str, atomicTable: AtomicTable):
        with open(filePath, 'r') as f:
            lines = f.readlines()

        l = get_next_line(lines)
        self.name = l
        l = get_next_line(lines)
        self.charge = int(l)
        if self.charge < 0 or self.charge > 1:
            raise ValueError("Only neutral or singly charged positive molecules are allowed (%s)" % self.name)

        structure = get_next_line(lines)
        constituents = [get_constituent(s.strip()) for s in structure.split(',')]
        self.elements = [atomicTable[c[1]] for c in constituents]
        self.elementCount = [c[0] for c in constituents]
        self.Nnuclei = sum(self.elementCount)

        l = get_next_line(lines)
        self.Ediss = float(l) * Const.EV

        fitStr = get_next_line(lines)
        self.formationTempRange = [float(f) for f in get_next_line(lines).split()]
        if len(self.formationTempRange) != 2:
            raise ValueError("Expected two entries for formation temperature range (%s)" % self.name)

        pfCoeffs = get_next_line(lines).split()
        Npf = int(pfCoeffs[0].strip())
        if len(pfCoeffs) != Npf+1:
            raise ValueError("Unexpected number of partition function fit parameters (%s)" % self.name)
        self.pfCoeffs = np.array([float(f.strip()) for f in pfCoeffs[1:]][::-1])

        eqcCoeffs = get_next_line(lines).split()
        Neqc = int(eqcCoeffs[0].strip())
        if len(eqcCoeffs) != Neqc+1:
            raise ValueError("Unexpected number of equilibrium coefficient fit parameters (%s)" % self.name)
        self.eqcCoeffs = np.array([float(f.strip()) for f in eqcCoeffs[1:]][::-1])

        self.weight = 0.0
        for count, ele in zip(self.elementCount, self.elements):
            self.weight += count * ele.weight

        if fitStr == 'KURUCZ_70':
            self.equilibrium_constant = equilibrium_constant_kurucz_70(self.formationTempRange, 
             self.Nnuclei - 1 - self.charge, 
             self.Ediss, self.eqcCoeffs)
        elif fitStr == 'KURUCZ_85':
            self.equilibrium_constant = equilibrium_constant_kurucz_85(self.formationTempRange, 
            self.Nnuclei - 1 - self.charge, self.Ediss, self.eqcCoeffs)
        elif fitStr == 'SAUVAL_TATUM_84':
            self.equilibrium_constant = equilibrium_constant_sauval_tatum(self.formationTempRange, self.Ediss, self.eqcCoeffs)
        else:
            raise ValueError('Unknown molecular equilibrium constant fit method %s in molecule %s' % (fitStr, self.name))

class MolecularTable:
    def __init__(self, paths: Optional[List[str]]=None, table: Optional[AtomicTable]=None):
        if table is None:
            table = get_global_atomic_table()
        self.molecules: List[Molecule] = []

        if paths is None:
            return

        for path in paths:
            self.molecules.append(Molecule(path, table))

        self.indices = OrderedDict(zip([m.name for m in self.molecules], list(range(len(self.molecules)))))

    def __getitem__(self, name: str) -> Molecule:
        name = name.upper()
        return self.molecules[self.indices[name]]

    def __contains__(self, name: str) -> bool:
        name = name.upper()
        return name in self.indices.keys()

    def __len__(self) -> int:
        return len(self.molecules)

    def __iter__(self) -> 'MolecularTableIterator':
        return MolecularTableIterator(self)

class MolecularTableIterator():
    def __init__(self, table: MolecularTable):
        self.table = table
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.table):
            mol = self.table.molecules[self.index]
            self.index += 1
            return mol

        raise StopIteration

@dataclass
class EquilibriumPopulations:
    atmosphere: Atmosphere
    atomicTable: AtomicTable
    atomicPops: List[np.ndarray]
    molecularTable: MolecularTable
    molecularPops: List[np.ndarray]
    HminPops: np.ndarray

    def __getitem__(self, name: str) -> np.ndarray:
        if name == 'H-':
            return self.HminPops
        else:
            name = name.upper()
            if len(name) == 1:
                name += ' '

            if name in self.molecularTable.indices.keys():
                key = self.molecularTable.indices[name]
                return self.molecularPops[key]
            elif name in self.atomicTable.indices.keys():
                key = self.atomicTable.indices[name]
                return self.atomicPops[key]
            else:
                raise KeyError('Unknown key: %s' % name)

    def __contains__(self, name: str) -> bool:
        if name == 'H-':
            return True
        
        if name in self.molecularTable.indices.keys():
            return True

        if name in self.atomicTable.indices.keys():
            return True

        return False

    def atomic_population(self, name: str) -> np.ndarray:
        name = name.upper()
        if len(name) == 1:
            name += ' '
        key = self.atomicTable.indices[name]
        return self.atomicPops[key]

    def molecular_population(self, name: str) -> np.ndarray:
        name = name.upper()
        key = self.molecularTable.indices[name]
        return self.molecularPops[key]




# TODO(cmo): remove?
def chemical_equilibrium(atmos: Atmosphere, molecules: MolecularTable, atomicPops: LtePopulations, computeNe=True) -> EquilibriumPopulations:
    table = atomicPops.atomicTable
    nucleiSet: Set[Element] = set()
    for mol in molecules:
        nucleiSet |= set(mol.elements)
    nuclei = list(nucleiSet)
    nuclei = sorted(nuclei, key=lambda x: table.indices[x.name])
    Nnuclei = len(nuclei)

    if not nuclei[0].name.startswith('H'):
        raise ValueError('H not list of nuclei -- check H2 molecule')

    # nuclIndex = [[table.indices[ele.name] for ele in mol.elements] for mol in molecules]
    nuclIndex = [[nuclei.index(ele) for ele in mol.elements] for mol in molecules]

    Neqn = Nnuclei + len(molecules)
    f = np.zeros(Neqn)
    n = np.zeros(Neqn)
    df = np.zeros((Neqn, Neqn))
    a = np.zeros(Neqn)

    # Equilibrium constant per molecule
    Phi = np.zeros(len(molecules))
    # Neutral fraction
    fn0 = np.zeros(Nnuclei)

    CI = (Const.HPlanck / (2.0 * np.pi * Const.MElectron)) * (Const.HPlanck / Const.KBoltzmann)
    Nspace = atmos.depthScale.shape[0]
    HminPops = np.zeros(Nspace)
    molPops = [np.zeros(Nspace) for mol in molecules]
    maxIter = 0
    for k in range(Nspace):
        for i in range(Nnuclei):
            a[i] = nuclei[i].abundance * atmos.nHTot[k]

        if not computeNe:
            for i in range(Nnuclei):
                fjk, dfjk = nuclei[i].fjk(atmos, k)
                fn0[i] = fjk[0]

            PhiHmin = 0.25 * (CI / atmos.temperature[k])**1.5 \
                        * np.exp(Const.E_ION_HMIN / (Const.KBoltzmann * atmos.temperature[k]))
            fHmin = atmos.ne[k] * fn0[0] * PhiHmin


        # Eq constant for each molecule at this location
        for i, mol in enumerate(molecules):
            Phi[i] = mol.equilibrium_constant(atmos.temperature[k])

        # Setup initial solution. Everything dissociated
        # n[:Nnuclei] = a[:Nnuclei]
        # n[Nnuclei:] = 0.0
        n[:] = a[:]
        print('a', a[Nnuclei])

        nIter = 1
        NmaxIter = 100
        IterLimit = 1e-3
        prevN = n.copy()
        while nIter < NmaxIter:
            if computeNe:
                table.compute_ne_k(atmos, k)
                for i in range(Nnuclei):
                    fjk, dfjk = nuclei[i].fjk(atmos, k)
                    fn0[i] = fjk[0]

                PhiHmin = 0.25 * (CI / atmos.temperature[k])**1.5 \
                            * np.exp(Const.E_ION_HMIN / (Const.KBoltzmann * atmos.temperature[k]))
                fHmin = atmos.ne[k] * fn0[0] * PhiHmin

            print(k, ',', nIter)
            # Save previous solution
            prevN[:] = n[:]

            # Set up iteration
            f[:] = n - a
            df[:, :] = 0.0
            np.fill_diagonal(df, 1.0)

            # Add nHmin to number conservation for H
            f[0] += fHmin * n[0]
            df[0, 0] += fHmin

            # Fill population vector f and derivative matrix df
            for i, mol in enumerate(molecules):
                saha = Phi[i]
                for j, ele in enumerate(mol.elements):
                    nu = nuclIndex[i][j]
                    saha *= (fn0[nu] * n[nu])**mol.elementCount[j]
                    # Contribution to conservation for each nucleus in this molecule
                    f[nu] += mol.elementCount[j] * n[Nnuclei + i]

                saha /= atmos.ne[k]**mol.charge
                f[Nnuclei + i] -= saha
                # if Nnuclei + i == f.shape[0]-1:
                #     print(i)
                #     print(saha)

                # Compute derivative matrix
                for j, ele in enumerate(mol.elements):
                    nu = nuclIndex[i][j]
                    df[nu, Nnuclei + i] += mol.elementCount[j]
                    df[Nnuclei + i, nu] = -saha * (mol.elementCount[j] / n[nu])

            # print(df)
            # print(f)
            correction = solve(df, f)
            n -= correction

            # correction = dgesvx(df, f.reshape(f.shape[0], 1))[7]
            # n -= correction.squeeze()
            # correction = newton_krylov(lambda x: df@x - f, np.zeros_like(f))
            # n -= correction
            # print(correction)

            dnMax = np.nanmax(np.abs((n - prevN) / n))
            print(dnMax)
            if dnMax <= IterLimit:
                maxIter = max(maxIter, nIter)
                break

            nIter += 1
        if dnMax > IterLimit:
            raise ValueError("ChemEq iteration not converged: T: %e [K], density %e [m^-3], dnmax %e" % (atmos.temperature[k], atmos.nHTot[k], dnMax))

        for i, ele in enumerate(nuclei):
            atomPop = atomicPops[ele.name]
            fraction = n[i] / np.sum(atomPop[:, k])
            atomPop[:, k] *= fraction

        HminPops[k] = atmos.ne[k] * n[0] * PhiHmin

        for i, pop in enumerate(molPops):
            pop[k] = n[Nnuclei + i] 

    result = EquilibriumPopulations(atmos, table, atomicPops.populations, molecules, molPops, HminPops)
    print("Maximum number of iterations taken: %d" % maxIter)
    return result







# def chemical_equilibrium(atmos: Atmosphere, molecules: MolecularTable, atomicPops: LtePopulations, computeNe=True) -> EquilibriumPopulations:
#     table = atomicPops.atomicTable
#     nucleiSet: Set[Element] = set()
#     for mol in molecules:
#         nucleiSet |= set(mol.elements)
#     nuclei = list(nucleiSet)
#     nuclei = sorted(nuclei, key=lambda x: table.indices[x.name])
#     Nnuclei = len(nuclei)

#     if not nuclei[0].name.startswith('H'):
#         raise ValueError('H not list of nuclei -- check H2 molecule')

#     # nuclIndex = [[table.indices[ele.name] for ele in mol.elements] for mol in molecules]
#     nuclIndex = [[nuclei.index(ele) for ele in mol.elements] for mol in molecules]

#     Neqn = Nnuclei + len(molecules)
#     f = np.zeros(Neqn)
#     n = np.zeros(Neqn)
#     df = np.zeros((Neqn, Neqn))
#     a = np.zeros(Neqn)

#     # Equilibrium constant per molecule
#     Phi = np.zeros(len(molecules))
#     # Neutral fraction
#     fn0 = np.zeros(Nnuclei)

#     CI = (Const.HPLANCK / (2.0 * np.pi * Const.M_ELECTRON)) * (Const.HPLANCK / Const.KBOLTZMANN)
#     Nspace = atmos.depthScale.shape[0]
#     HminPops = np.zeros(Nspace)
#     molPops = [np.zeros(Nspace) for mol in molecules]
#     maxIter = 0
#     for k in range(Nspace):
#         for i in range(Nnuclei):
#             fjk, dfjk = nuclei[i].fjk(atmos, k)
#             fn0[i] = fjk[0]
#             a[i] = nuclei[i].abundance * atmos.nHTot[k]

#         PhiHmin = 0.25 * (CI / atmos.temperature[k])**1.5 \
#                     * np.exp(Const.E_ION_HMIN / (Const.KBOLTZMANN * atmos.temperature[k]))
#         fHmin = atmos.ne[k] * fn0[0] * PhiHmin

#         # Eq constant for each molecule at this location
#         for i, mol in enumerate(molecules):
#             Phi[i] = mol.equilibrium_constant(atmos.temperature[k])

#         # Setup initial solution. Everything dissociated
#         # n[:Nnuclei] = a[:Nnuclei]
#         # n[Nnuclei:] = 0.0
#         n[:] = a[:]
#         print('a', a[Nnuclei])

#         nIter = 1
#         NmaxIter = 100
#         IterLimit = 1e-3
#         prevN = n.copy()
#         while nIter < NmaxIter:
#             print(k, ',', nIter)
#             # Save previous solution
#             prevN[:] = n[:]

#             # Set up iteration
#             f[:] = n - a
#             df[:, :] = 0.0
#             np.fill_diagonal(df, 1.0)

#             # Add nHmin to number conservation for H
#             f[0] += fHmin * n[0]
#             df[0, 0] += fHmin

#             # Fill population vector f and derivative matrix df
#             for i, mol in enumerate(molecules):
#                 saha = Phi[i]
#                 for j, ele in enumerate(mol.elements):
#                     nu = nuclIndex[i][j]
#                     saha *= (fn0[nu] * n[nu])**mol.elementCount[j]
#                     # Contribution to conservation for each nucleus in this molecule
#                     f[nu] += mol.elementCount[j] * n[Nnuclei + i]

#                 saha /= atmos.ne[k]**mol.charge
#                 f[Nnuclei + i] -= saha
#                 # if Nnuclei + i == f.shape[0]-1:
#                 #     print(i)
#                 #     print(saha)

#                 # Compute derivative matrix
#                 for j, ele in enumerate(mol.elements):
#                     nu = nuclIndex[i][j]
#                     df[nu, Nnuclei + i] += mol.elementCount[j]
#                     df[Nnuclei + i, nu] = -saha * (mol.elementCount[j] / n[nu])

#             # print(df)
#             # print(f)
#             correction = solve(df, f)
#             n -= correction

#             # correction = dgesvx(df, f.reshape(f.shape[0], 1))[7]
#             # n -= correction.squeeze()
#             # correction = newton_krylov(lambda x: df@x - f, np.zeros_like(f))
#             # n -= correction
#             # print(correction)

#             dnMax = np.nanmax(np.abs((n - prevN) / n))
#             print(dnMax)
#             if dnMax <= IterLimit:
#                 maxIter = max(maxIter, nIter)
#                 break

#             nIter += 1
#         if dnMax > IterLimit:
#             raise ValueError("ChemEq iteration not converged: T: %e [K], density %e [m^-3], dnmax %e" % (atmos.temperature[k], atmos.nHTot[k], dnMax))

#         for i, ele in enumerate(nuclei):
#             atomPop = atomicPops[ele.name]
#             fraction = n[i] / np.sum(atomPop[:, k])
#             atomPop[:, k] *= fraction

#         HminPops[k] = atmos.ne[k] * n[0] * PhiHmin

#         for i, pop in enumerate(molPops):
#             pop[k] = n[Nnuclei + i] 

#     result = EquilibriumPopulations(atmos, table, atomicPops.populations, molecules, molPops, HminPops)
#     print("Maximum number of iterations taken: %d" % maxIter)
#     return result