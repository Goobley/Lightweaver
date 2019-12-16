from Falc80 import Falc80
from AllOfTheAtoms import H_6_atom, H_6_CRD_atom, H_4_atom, C_atom, O_atom, OI_ord_atom, Si_atom, Al_atom, CaII_atom, Fe_atom, FeI_atom, He_9_atom, He_atom, He_large_atom, MgII_atom, N_atom, Na_atom, S_atom
from Atmosphere import Atmosphere, ScaleType
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
from astropy.io import fits



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

atmos= Falc80()
atmos.convert_scales()
atmos.quadrature(5)
aSet = RadiativeSet([H_4_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe_atom(), He_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
aSet.set_active('H')
spect = aSet.compute_wavelength_grid()

molPaths = ['../Molecules/' + m + '.molecule' for m in ['H2']]
mols = MolecularTable(molPaths)

eqPops = aSet.compute_eq_pops(mols, atmos)
ctx = LwContext(atmos, spect, eqPops, ngOptions=NgOptions(0,0,0), hprd=False, conserveCharge=False)
iterate_ctx(ctx, prd=False)

print('Achieved initial Stat Eq')
print('Waiting')
input()

dt = 0.5
NtStep = 30
NsubStep = 100

prevT = np.copy(atmos.temperature)
for i in range(11, 31):
    di = (i - 20.0) / 3.0
    atmos.temperature[i] *= 1.0 + 2.0 * np.exp(-di**2)
print(atmos.temperature - prevT)

hPops = [np.copy(eqPops['H'])]
for it in range(NtStep):
    eqPops.update_lte_atoms_Hmin_pops(atmos, conserveCharge=True)
    ctx.background.update_background()
    ctx.compute_profiles()

    prevState = None
    for sub in range(NsubStep):
        dJ = ctx.formal_sol_gamma_matrices()
        delta, prevState = ctx.time_dep_update(dt, prevState)

        if delta < 1e-3 and dJ < 3e-3:
            break
    else:
        raise ValueError('No converge')

    ctx.time_dep_conserve_charge(prevState)
    hPops.append(np.copy(eqPops['H']))
    print('Iteration %d (%f s) done after %d sub iterations' % (it, (it+1)*dt, sub))

    # input()

initialAtmos = Falc80()

fig, ax = plt.subplots(2,2, sharex=True)
ax = ax.flatten()
cmass = np.log10(atmos.cmass/1e1)

ax[0].plot(cmass, initialAtmos.temperature, 'k')
ax[0].plot(cmass, atmos.temperature, '--')

for p in hPops[1:]:
    ax[1].plot(cmass, np.log10(p[0,:]/1e6))
    ax[2].plot(cmass, np.log10(p[1,:]/1e6))
    ax[3].plot(cmass, np.log10(p[-1,:]/1e6))

p = hPops[0]
ax[1].plot(cmass, np.log10(p[0,:]/1e6), 'k')
ax[2].plot(cmass, np.log10(p[1,:]/1e6), 'k')
ax[3].plot(cmass, np.log10(p[-1,:]/1e6), 'k')

ax[0].set_xlim(-4.935, -4.931)
ax[0].set_ylim(0, 6e4)
ax[1].set_ylim(6, 11)
ax[2].set_ylim(1, 6)
ax[3].set_ylim(10, 11)

