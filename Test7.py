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

from concurrent.futures import ProcessPoolExecutor, as_completed, wait

@dataclass
class NgOptions:
    Norder: int = 3
    Nperiod: int = 20
    Ndelay: int = 20
    
with open('state.pkl', 'rb') as p:
    ctx = pickle.load(p)

# with open('state2.pkl', 'rb') as p:
#     ctx2 = pickle.load(p)

# with open('state3.pkl', 'rb') as p:
#     ctx3 = pickle.load(p)

def iterate_ctx(ctx):
    for i in range(300):
        dJ = ctx.formal_sol_gamma_matrices()
        if i < 3:
            continue
        delta = ctx.stat_equil()
        dRho = ctx.prd_redistribute()

        if dJ < 7e-3 and delta < 5e-3:
            print(i)
            print('----------')
            return
# wave = np.linspace(853.9444, 854.9444, 1001)
# wave = np.linspace(392, 398, 10001)
iterate_ctx(ctx)
wave = np.linspace(392, 395, 2001)
Iwave = ctx.compute_rays(wave, [1.0])

# rf = np.load('8542RF.npy')


pertSize = 25
pertPlus = np.zeros((wave.shape[0], 82))
pertMinus = np.zeros((wave.shape[0], 82))
sd = deepcopy(ctx.state_dict())

def response_fn_temp_k(state, k):
    sd = deepcopy(state)
    atmos2 = deepcopy(sd['arguments']['atmos'])
    atmos2.temperature[k] += 0.5 * pertSize
    ctxPlus = LwContext.construct_from_state_dict_with(sd, atmos=atmos2)
    iterate_ctx(ctxPlus)
    res1 = np.copy(ctxPlus.compute_rays(wave, [1.0])[:, 0])

    sd = deepcopy(state)
    atmos2 = deepcopy(sd['arguments']['atmos'])
    atmos2.temperature[k] -= 0.5*pertSize
    ctxMinus = LwContext.construct_from_state_dict_with(sd, atmos=atmos2)
    iterate_ctx(ctxMinus)
    res2 = np.copy(ctxMinus.compute_rays(wave, [1.0])[:, 0])

    return (res1, res2)

with ProcessPoolExecutor() as executor:
    futures = [executor.submit(response_fn_temp_k, sd, k) for k in range(82)]
    wait(futures)
    perturbations = [f.result() for f in futures]
# perturbations = [response_fn_temp_k(sd, k) for k in range(81, 75, -1)]

pos, neg = zip(*perturbations)
for k in range(len(pos)):
    pertPlus[:, k] = pos[k]
    pertMinus[:, k] = neg[k]

# for k in range(82):
    # print('==========%.3d==========' % k)
    # atmos2 = deepcopy(sd['arguments']['atmos'])
    # atmos2.temperature[k] += 0.5 * pertSize
    # ctxPlus = LwContext.construct_from_state_dict_with(sd, atmos=atmos2)
    # iterate_ctx(ctxPlus)
    # pertPlus[:, k] = ctxPlus.compute_rays(wave, [1.0])[:, 0]

    # atmos2.temperature[k] -= pertSize
    # ctxMinus = LwContext.construct_from_state_dict_with(sd, atmos=atmos2)
    # iterate_ctx(ctxMinus)
    # pertMinus[:, k] = ctxMinus.compute_rays(wave, [1.0])[:, 0]


rf = (pertPlus - pertMinus) / Iwave
cfData = ctx.contrib_fn(ctx.arguments['spect'].radSet['Ca'].lines[1], wavelengths=wave, mu=[1.0])