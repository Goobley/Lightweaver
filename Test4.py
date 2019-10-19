import numpy as np
from Falc80 import Falc80
from AtomicTable import AtomicTable
from Molecule import chemical_equilibrium, MolecularTable
from CAtmosphere import LwContext, InitialSolution
from AtomicSet import RadiativeSet
from AllOfTheAtoms import CaIIatom, MgIIatom, H_6atom
from PyProto import background
import time

from dataclasses import dataclass

@dataclass
class NgOptions:
    Norder: int = 2
    Nperiod: int = 3
    Ndelay: int = 12

atmosZero = Falc80()
# atmosPert = Falc80()
# atmosPert.vlos[40] = 10000
atmos = Falc80()
# vel = np.sin(np.linspace(0, 8*np.pi, atmos.vlos.shape[0])) * 2e4
# atmos.vlos = vel
at = AtomicTable()
atmos.convert_scales(at)
atmos.quadrature(5)
atmosZero.convert_scales(at)
atmosZero.quadrature(5)
# atmosPert.convert_scales(at)
# atmosPert.quadrature(5)
aSet = RadiativeSet([H_6atom(), CaIIatom(), MgIIatom()], set([]))
aSet.set_active('H', 'Ca', 'Mg')
spect = aSet.compute_wavelength_grid(np.linspace(150, 600, 500))
# spect = aSet.compute_wavelength_grid()

# np.seterr(invalid='raise', divide='raise')

bg = background(atmos, spect)
# activeAtoms = [a for a in aSet.activeAtoms]
activeAtoms = aSet.activeAtoms
ltePops = at.lte_populations(atmos)
molPaths = ['../Molecules/' + m + '.molecule' for m in ['H2', 'H2+', 'C2', 'N2', 'O2', 'CH', 'CO', 'CN', 'NH', 'NO', 'OH', 'H2O']]
mols = MolecularTable(molPaths, at)
eqPops = chemical_equilibrium(atmos, mols, ltePops, computeNe=False)
aSet.wavelength = spect.wavelength

a = np.zeros((3, spect.wavelength.shape[0]))
for la in range(spect.wavelength.shape[0]):
    for atom in spect.contributors[la]:
        if atom.name.startswith('H'):
            a[0, la] = 1
        elif atom.name == 'MG':
            a[1, la] = 1
        elif atom.name == 'CA':
            a[2, la] = 1
# TODO(cmo): This needs to take the radiativeSet, rather than simply activeAtoms
ctx = LwContext(atmos, spect, eqPops, aSet, bg, at, ngOptions=NgOptions(0,0,0), initSol=InitialSolution.Lte)
# ctx.gamma_matrices_formal_sol()
# delta = ctx.stat_equil()
# print("delta: %e"%delta)

# input()
start = time.time()
delta = 1.0
dJ = 1.0
it = 0
for it in range(200):
    it += 1
    dJ = ctx.gamma_matrices_formal_sol()
    if dJ < 1e-2:
        break
    delta = ctx.stat_equil()
    print(delta, it)
end = time.time()
print('%.2e'%(end-start))

# # ctx2 = LwContext(atmosZero, spect, activeAtoms, bg, at, ngOptions=NgOptions(), initSol=InitialSolution.EscapeProbability)
# # delta = 1.0
# # dJ = 1.0
# # it = 0
# # for it in range(200):
# #     it += 1
# #     dJ = ctx2.gamma_matrices_formal_sol()
# #     if dJ < 1e-2:
# #         break
# #     delta = ctx2.stat_equil()
# #     print(delta, it)
# # end = time.time()
# # print('%.2e'%(end-start))
# # # ctx.gamma_matrices_formal_sol()
# # # Iplus = gamma_matrices(atmos, spect, activeAtoms, bg)
# # # delta = stat_equil(atmos, activeAtoms)