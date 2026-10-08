"""
===============================================================
Computing a simple NLTE 8542 line profile in a FAL C atmosphere
===============================================================
"""

# %%
# First, we import everything we need. Lightweaver is typically imported as
# `lw`, but things like the library of model atoms and Fal atmospheres need to
# be imported separately.
import matplotlib.pyplot as plt
import numpy as np

import lightweaver as lw
from lightweaver.fal import Falc82
from lightweaver.rh_atoms import (
    Al_atom,
    C_atom,
    CaII_atom,
    Fe_atom,
    H_6_atom,
    He_9_atom,
    MgII_atom,
    N_atom,
    Na_atom,
    O_atom,
    S_atom,
    Si_atom,
)


# %%
# Now, we define the functions that will be used in our spectral synthesise.
# First `synth_8542` which synthesises and returns the line given by an
# atmosphere.
def synth_8542(atmos, conserve, use_ne, wave):
    """
    Synthesise a spectral line for given atmosphere with different
    conditions.

    Parameters
    ----------
    atmos : lw.Atmosphere
        The atmospheric model in which to synthesise the line.
    conserve : bool
        Whether to start from LTE electron density and conserve charge, or
        simply use from the electron density present in the atomic model.
    use_ne : bool
        Whether to use the electron density present in the model as the
        starting solution, or compute the LTE electron density.
    wave : np.ndarray
        Array of wavelengths over which to resynthesise the final line
        profile for muz=1.

    Returns
    -------
    ctx : lw.Context
        The Context object that was used to compute the equilibrium
        populations.
    I_wave : np.ndarray
        The intensity at muz=1 for each wavelength in `wave`.
    """
    # Configure the atmospheric angular quadrature
    atmos.quadrature(5)
    # Configure the set of atomic models to use.
    rad_set = lw.RadiativeSet(
        [
            H_6_atom(),
            C_atom(),
            O_atom(),
            Si_atom(),
            Al_atom(),
            CaII_atom(),
            Fe_atom(),
            He_9_atom(),
            MgII_atom(),
            N_atom(),
            Na_atom(),
            S_atom(),
        ]
    )
    # Set H and Ca to "active" i.e. NLTE, everything else participates as an
    # LTE background.
    rad_set.set_active('H', 'Ca')
    # Compute the necessary wavelength dependent information (SpectrumConfiguration).
    spect = rad_set.compute_wavelength_grid()

    # Either compute the equilibrium populations at the fixed electron density
    # provided in the model, or iterate an LTE electron density and compute the
    # corresponding equilibrium populations (SpeciesStateTable).
    if use_ne:
        eq_pops = rad_set.compute_eq_pops(atmos)
    else:
        eq_pops = rad_set.iterate_lte_ne_eq_pops(atmos)

    # Configure the Context which holds the state of the simulation for the
    # backend, and provides the python interface to the backend.
    # Feel free to increase Nthreads to increase the number of threads the
    # program will use.
    ctx = lw.Context(atmos, spect, eq_pops, conserve_charge=conserve, Nthreads=1)
    # Iterate the Context to convergence (using the iteration function now
    # provided by Lightweaver)
    lw.iterate_ctx_se(ctx)
    # Update the background populations based on the converged solution and
    # compute the final intensity for mu=1 on the provided wavelength grid.
    eq_pops.update_lte_atoms_hmin_pops(atmos)
    I_wave = ctx.compute_rays(wave, [atmos.muz[-1]], stokes=False)
    return ctx, I_wave


# %%
# The wavelength grid to output the final synthesised line on.
wave = np.linspace(853.9444, 854.9444, 1001)

# %%
# Load an lw.Atmosphere object containing the FAL C atmosphere with 82 points
# in depth, before synthesising the Ca II 8542 \AA line profile using:
#
# - The given electron density.
# - The electron density charge conserved from a starting LTE solution.
# - The LTE electron density.
#
# These results are then plotted.

atmos_ref = Falc82()
ctx_ref, I_wave_ref = synth_8542(atmos_ref, conserve=False, use_ne=True, wave=wave)
atmos_cons = Falc82()
ctx_cons, I_wave_cons = synth_8542(atmos_cons, conserve=True, use_ne=False, wave=wave)
atmos_lte = Falc82()
ctx, I_wave_lte = synth_8542(atmos_lte, conserve=False, use_ne=False, wave=wave)

plt.plot(wave, I_wave_ref, label='Reference FAL')
plt.plot(wave, I_wave_cons, label='Reference Cons')
plt.plot(wave, I_wave_lte, label='Reference LTE n_e')
plt.legend()
plt.show()
