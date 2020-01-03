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
from concurrent.futures import ProcessPoolExecutor, wait
from tqdm import tqdm
from Utils import NgOptions
from Multi import MultiMetadata, read_multi_atmos


# meta, atmos = read_multi_atmos('FaultyFal/model1003.atmos')
atmos = Falc80()
atmos.convert_scales()
atmos.quadrature(5)
aSet = RadiativeSet([H_6_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe_atom(), He_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
aSet.set_active('H', 'Ca')
spect = aSet.compute_wavelength_grid()

molPaths = ['../Molecules/' + m + '.molecule' for m in ['H2']]
mols = MolecularTable(molPaths)

eqPops = aSet.compute_eq_pops(mols, atmos)
ctx = LwContext(atmos, spect, eqPops, initSol=InitialSolution.Lte, ngOptions=NgOptions(0,0,0), hprd=False)

def iterate_ctx(ctx, prd=True, Nscatter=3, NmaxIter=500):
    for i in range(NmaxIter):
        dJ = ctx.formal_sol_gamma_matrices()
        if i < Nscatter:
            continue
        delta = ctx.stat_equil()
        if prd:
            dRho = ctx.prd_redistribute(maxIter=5)

        if dJ < 3e-3 and delta < 1e-3:
            print(i)
            print('----------')
            return

pertSize = 30
wave = np.linspace(655, 658, 1001)
pertPlus = np.zeros((spect.wavelength.shape[0], atmos.Nspace))
pertMinus = np.zeros((spect.wavelength.shape[0], atmos.Nspace))

iterate_ctx(ctx)
Iwave = ctx.compute_rays(wave, [1.0])
sd = deepcopy(ctx.state_dict())

def response_fn_temp_k(state, k):
    sd = deepcopy(state)
    atmos2 = deepcopy(sd['arguments']['atmos'])
    atmos2.temperature[k] += 0.5 * pertSize
    # pertSize = 0.0001 * atmos2.ne[k]
    # atmos2.ne[k] += 0.5 * pertSize
    # atmos2.vlos[k] += 0.5 * pertSize
    ctxPlus = LwContext.construct_from_state_dict_with(sd, atmos=atmos2)
    iterate_ctx(ctxPlus)
    res1 = np.copy(ctxPlus.compute_rays(wavelengths=None, mus=[1.0])[:, 0])

    sd = deepcopy(state)
    atmos2 = deepcopy(sd['arguments']['atmos'])
    atmos2.temperature[k] -= 0.5*pertSize
    # atmos2.ne[k] -= 0.5*pertSize
    # atmos2.vlos[k] -= 0.5*pertSize
    ctxMinus = LwContext.construct_from_state_dict_with(sd, atmos=atmos2)
    iterate_ctx(ctxMinus)
    res2 = np.copy(ctxMinus.compute_rays(wavelengths=None, mus=[1.0])[:, 0])

    return (res1, res2)

with ProcessPoolExecutor() as executor:
    futures = [executor.submit(response_fn_temp_k, sd, k) for k in range(atmos.Nspace)]
    wait(futures)
    perturbations = [f.result() for f in futures]
# perturbations = [response_fn_temp_k(sd, k) for k in range(81, 75, -1)]

pos, neg = zip(*perturbations)
for k in range(len(pos)):
    pertPlus[:, k] = pos[k]
    pertMinus[:, k] = neg[k]

rf = (pertPlus - pertMinus) / Iwave
