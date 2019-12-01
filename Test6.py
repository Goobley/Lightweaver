from Falc80 import Falc80
from AllOfTheAtoms import H_6_atom, H_6_CRD_atom, C_atom, O_atom, OI_ord_atom, Si_atom, Al_atom, CaII_atom, Fe_atom, FeI_atom, He_9_atom, He_atom, He_large_atom, MgII_atom, N_atom, Na_atom, S_atom
from AtomicSet import RadiativeSet
from AtomicTable import get_global_atomic_table
from Molecule import MolecularTable
from CAtmosphere import LwContext
from PyProto import InitialSolution
from dataclasses import dataclass
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import pickle
import numpy as np

@dataclass
class NgOptions:
    Norder: int = 3
    Nperiod: int = 20
    Ndelay: int = 20

atmos = Falc80()
atmos.convert_scales()
atmos.quadrature(5)
aSet = RadiativeSet([H_6_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe_atom(), He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
aSet.set_active('H', 'Ca')
spect = aSet.compute_wavelength_grid()

molPaths = ['../Molecules/' + m + '.molecule' for m in ['H2']]
mols = MolecularTable(molPaths)

eqPops = aSet.compute_eq_pops(mols, atmos)
ctx = LwContext(atmos, spect, aSet, eqPops, get_global_atomic_table(), ngOptions=NgOptions(0,0,0), initSol=InitialSolution.Lte, hprd=True)

for i in range(4):
    dJ = ctx.formal_sol_gamma_matrices()
    if i >= 3:
        delta = ctx.stat_equil()
        # dRho = ctx.prd_redistribute()

aSet2 = RadiativeSet([H_6_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe_atom(), He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
aSet2.set_active('H', 'Ca')
spect2 = aSet2.compute_wavelength_grid()

sd = deepcopy(ctx.state_dict())
ctx2 = LwContext.construct_from_state_dict_with(sd, spect=spect2, preserveProfiles=True)

ctx3 = pickle.loads(pickle.dumps(ctx))

ctx3.formal_sol_gamma_matrices()
ctx3.stat_equil()
# ctx3.prd_redistribute()
ctx3.formal_sol_gamma_matrices()
ctx3.stat_equil()
# ctx3.prd_redistribute()
    
print('----------')

ctx2.formal_sol_gamma_matrices()
ctx2.stat_equil()
# ctx2.prd_redistribute()
ctx2.formal_sol_gamma_matrices()
ctx2.stat_equil()
# ctx2.prd_redistribute()

print('----------')
ctx.formal_sol_gamma_matrices()
ctx.stat_equil()
# ctx.prd_redistribute()
ctx.formal_sol_gamma_matrices()
ctx.stat_equil()
# ctx.prd_redistribute()