'''
======================
Time-dependent Example
======================
Simple illustrative example of time-dependent method. Herein we reproduce the
time-dependent population figure present in Judge 2017. Here the complete
Rybicki-Hummer MALI method is used.
Herein we also conserve charge.

Judge (2017): ApJ 851, 5
'''
from lightweaver.fal import Falc82
from lightweaver.rh_atoms import H_6_atom, H_6_CRD_atom, H_4_atom, C_atom, O_atom, OI_ord_atom, Si_atom, Al_atom, CaII_atom, Fe_atom, FeI_atom, He_9_atom, He_atom, He_large_atom, MgII_atom, N_atom, Na_atom, S_atom
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import numpy as np
import lightweaver as lw

def iterate_ctx(ctx, Nscatter=3, NmaxIter=500):
    '''
    Iterate a context to statistical equilbrium convergence.
    '''
    for i in range(NmaxIter):
        dJ = ctx.formal_sol_gamma_matrices()
        if i < Nscatter:
            continue
        delta = ctx.stat_equil()

        if dJ < 3e-3 and delta < 1e-3:
            print(i)
            print('----------')
            return

#%%
# Set up the standard FAL C 82 point initial atmosphere.
atmos = Falc82()
atmos.quadrature(5)
aSet = lw.RadiativeSet([H_4_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(),\
                        Fe_atom(), He_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
aSet.set_active('H')
spect = aSet.compute_wavelength_grid()

eqPops = aSet.iterate_lte_ne_eq_pops(atmos)
ctx = lw.Context(atmos, spect, eqPops, conserveCharge=True, Nthreads=8)

#%%
# Find the initial statistical equilibrium solution,
iterate_ctx(ctx)

print('Achieved initial Stat Eq\n\n')


#%%
# Simulation parameters, timestep, number of steps to run for, and how many
# times to attempt to solve the equations for convergence per step.
start = time.time()
dt = 0.1
NtStep = 30
NsubStep = 100

#%%
# Perturb the atmospheric temperature structure like in the paper.
prevT = np.copy(atmos.temperature)
for i in range(11, 31):
    di = (i - 20.0) / 3.0
    atmos.temperature[i] *= 1.0 + 2.0 * np.exp(-di**2)

#%%
# Solve the problem
hPops = [np.copy(eqPops['H'])]
subIters = []
for it in range(NtStep):
    # Recompute line profiles etc to account for changing electron density and temperature.
    ctx.update_deps()

    prevState = None
    for sub in range(NsubStep):
        dJ = ctx.formal_sol_gamma_matrices()
        # If prevState is None, then the function assumes that this is the
        # subiteration for this step and constructs and returns prevState
        delta, prevState = ctx.time_dep_update(dt, prevState)
        # Update electron density.
        deltaNr = ctx.nr_post_update(timeDependentData={'dt': dt, 'nPrev': prevState})
        delta = max(delta, deltaNr)

        # Check subiteration convergence
        if delta < 1e-3 and dJ < 3e-3:
            subIters.append(sub)
            break
    else:
        raise ValueError('No convergence within required Nsubstep')

    hPops.append(np.copy(eqPops['H']))
    print('Iteration %d (%f s) done after %d sub iterations' % (it, (it+1)*dt, sub))

    # input()
end = time.time()

#%%
# Reproduce Judge plot.

initialAtmos = Falc82()

plt.ion()
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

print('Time taken: %.2f s' % (end-start))
