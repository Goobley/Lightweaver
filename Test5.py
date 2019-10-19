from Falc80 import Falc80
from AllOfTheAtoms import H_6atom, Catom, Oatom, CaIIatom, Featom, He_9atom, MgIIatom
# TODO(cmo): Find out why the other atoms weren't generated
from AtomicTable import AtomicTable
from AtomicSet import RadiativeSet
from Molecule import MolecularTable
from CAtmosphere import LwContext
from PyProto import InitialSolution
import time
import numpy as np

from dataclasses import dataclass

@dataclass
class NgOptions:
    Norder: int = 2
    Nperiod: int = 3
    Ndelay: int = 12

atmos = Falc80()
at = AtomicTable()
atmos.convert_scales(at)
atmos.quadrature(5)

# NOTE(cmo): adding FeAtom seems to make us explode... Need to fix
aSet = RadiativeSet([H_6atom(), Catom(), Oatom(), CaIIatom(), MgIIatom(), He_9atom()], set([]))
aSet.set_active('Ca', 'Mg', 'H')
spect = aSet.compute_wavelength_grid(np.linspace(150, 600, 500))

molPaths = ['../Molecules/' + m + '.molecule' for m in ['H2', 'H2+', 'C2', 'N2', 'O2', 'CH', 'CO', 'CN', 'NH', 'NO', 'OH', 'H2O']]
start = time.time()
mols = MolecularTable(molPaths, at)

eqPops = aSet.compute_eq_pops(mols, atmos)
ctx = LwContext(atmos, spect, aSet, eqPops, at, ngOptions=NgOptions(0,0,0), initSol=InitialSolution.Lte)
delta = 1.0
dJ = 1.0
it = 0
for it in range(200):
    it += 1
    dJ = ctx.gamma_matrices_formal_sol()
    if dJ < 1e-2:
        break
    delta = ctx.stat_equil()
end = time.time()
print('%e s' % (end-start))