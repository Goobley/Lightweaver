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
from tqdm import tqdm

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
ctx = LwContext(atmos, spect, eqPops, ngOptions=NgOptions(0,0,0), hprd=True)

# for i in range(4):
#     dJ = ctx.formal_sol_gamma_matrices()
#     if i >= 3:
#         delta = ctx.stat_equil()
#         dRho = ctx.prd_redistribute()

def iterate_ctx(ctx):
    for i in range(300):
        dJ = ctx.formal_sol_gamma_matrices()
        if i < 3:
            continue
        delta = ctx.stat_equil()
        dRho = ctx.prd_redistribute()

        if dJ < 3e-3 and delta < 1e-3:
            print(i)
            print('----------')
            return

iterate_ctx(ctx)
ctx2 = LwContext.construct_from_state_dict_with(ctx.state_dict())
# wave = np.linspace(853.9444, 854.9444, 1001)
# wave = np.linspace(392, 398, 10001)
wave = np.linspace(392, 395, 2001)
Iwave = ctx2.compute_rays(wave, [1.0])

# rf = np.load('8542RF.npy')

# cfData = ctx.contrib_fn(aSet['Ca'].lines[-1], wavelengths=wave, mu=[1.0])


pertPlus = np.zeros((2001, atmos.Nspace))
pertMinus = np.zeros((2001, atmos.Nspace))
pertSize = 0

state = deepcopy(ctx.state_dict())
for k in range(81, 75, -1):
    print('==========%.3d==========' % k)
    sd = deepcopy(state)
    atmos2 = deepcopy(sd['arguments']['atmos'])
    atmos2.temperature[k] += 0.5 * pertSize
    ctxPlus = LwContext.construct_from_state_dict_with(sd, atmos=atmos2)
    iterate_ctx(ctxPlus)
    pertPlus[:, k] = ctxPlus.compute_rays(wave, [1.0])[:, 0]

    sd = deepcopy(state)
    atmos2 = deepcopy(sd['arguments']['atmos'])
    atmos2.temperature[k] -= pertSize
    ctxMinus = LwContext.construct_from_state_dict_with(sd, atmos=atmos2)
    iterate_ctx(ctxMinus)
    pertMinus[:, k] = ctxMinus.compute_rays(wave, [1.0])[:, 0]


rf = (pertPlus - pertMinus) / Iwave

# ctx2 = LwContext.construct_from_state_dict_with(sd)
# iterate_ctx(ctx2)

# s = ctx.spect
# s2 = ctx2.spect
# plt.plot(s.wavelength, s.I[:,-1])
# plt.plot(s2.wavelength, s2.I[:,-1], '--')

# ctx3 = pickle.loads(pickle.dumps(ctx))


# ctx3.formal_sol_gamma_matrices()
# ctx3.stat_equil()
# ctx3.prd_redistribute()
# ctx3.formal_sol_gamma_matrices()
# ctx3.stat_equil()
# ctx3.prd_redistribute()
    
# print('----------')

# ctx2.formal_sol_gamma_matrices()
# ctx2.stat_equil()
# ctx2.prd_redistribute()
# ctx2.formal_sol_gamma_matrices()
# ctx2.stat_equil()
# ctx2.prd_redistribute()

# print('----------')
# ctx.formal_sol_gamma_matrices()
# ctx.stat_equil()
# ctx.prd_redistribute()
# ctx.formal_sol_gamma_matrices()
# ctx.stat_equil()
# ctx.prd_redistribute()