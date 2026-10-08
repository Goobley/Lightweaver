"""
======================
Time-dependent Example
======================
Simple illustrative example of time-dependent method. Herein we reproduce the
time-dependent population figure present in Judge 2017. Here the complete
Rybicki-Hummer MALI method is used.
Herein we also conserve charge.

Judge (2017): ApJ 851, 5
"""

import time

import matplotlib.pyplot as plt
import numpy as np

import lightweaver as lw
from lightweaver.fal import Falc82
from lightweaver.rh_atoms import (
    Al_atom,
    C_atom,
    CaII_atom,
    Fe_atom,
    H_4_atom,
    He_atom,
    MgII_atom,
    N_atom,
    Na_atom,
    O_atom,
    S_atom,
    Si_atom,
)

# %%
# Set up the standard FAL C 82 point initial atmosphere.
atmos = Falc82()
atmos.quadrature(5)
rad_set = lw.RadiativeSet(
    [
        H_4_atom(),
        C_atom(),
        O_atom(),
        Si_atom(),
        Al_atom(),
        CaII_atom(),
        Fe_atom(),
        He_atom(),
        MgII_atom(),
        N_atom(),
        Na_atom(),
        S_atom(),
    ]
)
rad_set.set_active('H')
spect = rad_set.compute_wavelength_grid()

eq_pops = rad_set.iterate_lte_ne_eq_pops(atmos)
ctx = lw.Context(atmos, spect, eq_pops, conserve_charge=True, Nthreads=1)

# %%
# Find the initial statistical equilibrium solution,
lw.iterate_ctx_se(ctx)

print('Achieved initial Stat Eq\n\n')


# %%
# Simulation parameters, timestep, number of steps to run for, and how many
# times to attempt to solve the equations for convergence per step.
start = time.time()
dt = 0.1
Nt_step = 30
Nsub_step = 100

# %%
# Perturb the atmospheric temperature structure like in the paper.
prev_T = np.copy(atmos.temperature)
for i in range(11, 31):
    di = (i - 20.0) / 3.0
    atmos.temperature[i] *= 1.0 + 2.0 * np.exp(-(di**2))

# %%
# Solve the problem
h_pops = [np.copy(eq_pops['H'])]
sub_iters = []
for it in range(Nt_step):
    # Recompute line profiles etc to account for changing electron density and temperature.
    ctx.update_deps()

    prev_state = None
    for sub in range(Nsub_step):
        J_update = ctx.formal_sol_gamma_matrices()
        # If prev_state is None, then the function assumes that this is the
        # subiteration for this step and constructs and returns prev_state
        pops_update, prev_state = ctx.time_dep_update(dt, prev_state)
        # Update electron density.
        # If conserve_charge is set to True when the context is constructed, then
        # the effects of `time_dep_update` are included in the IterationUpdate
        # returned from `nr_post_update`, as the Context is expecting this to be
        # called immediately after `time_dep_update`.
        nr_update = ctx.nr_post_update(time_dependent_data={'dt': dt, 'n_prev': prev_state})

        # Check subiteration convergence
        if nr_update.dpops_max < 1e-3 and J_update.dJ_max < 3e-3:
            sub_iters.append(sub)
            break
    else:
        raise ValueError('No convergence within required Nsubstep')

    h_pops.append(np.copy(eq_pops['H']))
    print('Iteration %d (%f s) done after %d sub iterations' % (it, (it + 1) * dt, sub))

    # input()
end = time.time()

# %%
# Reproduce Judge plot.

initial_atmos = Falc82()

plt.ion()
fig, ax = plt.subplots(2, 2, sharex=True)
ax = ax.flatten()
cmass = np.log10(atmos.cmass / 1e1)

ax[0].plot(cmass, initial_atmos.temperature, 'k')
ax[0].plot(cmass, atmos.temperature, '--')

for p in h_pops[1:]:
    ax[1].plot(cmass, np.log10(p[0, :] / 1e6))
    ax[2].plot(cmass, np.log10(p[1, :] / 1e6))
    ax[3].plot(cmass, np.log10(p[-1, :] / 1e6))

p = h_pops[0]
ax[1].plot(cmass, np.log10(p[0, :] / 1e6), 'k')
ax[2].plot(cmass, np.log10(p[1, :] / 1e6), 'k')
ax[3].plot(cmass, np.log10(p[-1, :] / 1e6), 'k')

ax[0].set_xlim(-4.935, -4.931)
ax[0].set_ylim(0, 6e4)
ax[1].set_ylim(6, 11)
ax[2].set_ylim(1, 6)
ax[3].set_ylim(10, 11)

print('Time taken: %.2f s' % (end - start))
