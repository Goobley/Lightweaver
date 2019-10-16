import numpy as np
from Falc80 import Falc80
from AtomicTable import AtomicTable
from CAtmosphere import LwContext, InitialSolution
from AtomicSet import RadiativeSet
from AllOfTheAtoms import CaIIatom, MgIIatom
from PyProto import background
import time

from dataclasses import dataclass

@dataclass
class NgOptions:
    Norder: int = 2
    Nperiod: int = 3
    Ndelay: int = 12

atmosZero = Falc80()
atmos = Falc80()
vel = np.sin(np.linspace(0, 8*np.pi, atmos.vlos.shape[0])) * 2e4
atmos.vlos = vel
at = AtomicTable()
atmos.convert_scales(at)
atmos.quadrature(5)
atmosZero.convert_scales(at)
atmosZero.quadrature(5)
aSet = RadiativeSet([CaIIatom(), MgIIatom()], [])
aSet.set_active('Ca', 'Mg')
spect = aSet.compute_wavelength_grid(np.linspace(150, 600, 500))
# spect = aSet.compute_wavelength_grid()

np.seterr(invalid='raise', divide='raise')

bg = background(atmos, spect)
activeAtoms = [a for a in aSet.activeAtoms]
start = time.time()
# TODO(cmo): This needs to take the radiativeSet, rather than simply activeAtoms
ctx = LwContext(atmos, spect, activeAtoms, bg, at, ngOptions=NgOptions(), initSol=InitialSolution.EscapeProbability)
# ctx.gamma_matrices_formal_sol()
# delta = ctx.stat_equil()
# print("delta: %e"%delta)

# input()
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

ctx2 = LwContext(atmosZero, spect, activeAtoms, bg, at, ngOptions=NgOptions(), initSol=InitialSolution.EscapeProbability)
delta = 1.0
dJ = 1.0
it = 0
for it in range(200):
    it += 1
    dJ = ctx2.gamma_matrices_formal_sol()
    if dJ < 1e-2:
        break
    delta = ctx2.stat_equil()
    print(delta, it)
end = time.time()
print('%.2e'%(end-start))
# ctx.gamma_matrices_formal_sol()
# Iplus = gamma_matrices(atmos, spect, activeAtoms, bg)
# delta = stat_equil(atmos, activeAtoms)
