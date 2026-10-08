"""
Contribution functions: store the depth-dependent opacity and emissivity
during a formal solution, and use them to compute optical depths, formation
heights and contribution functions.

Demonstrates `ctx.depthData`, `lw.utils.compute_contribution_fn`,
`lw.utils.compute_tau` and `lw.utils.tau_isosurface`.
"""

import numpy as np
from conftest import copy_ctx
from test_stat_eq import Ca8542

from lightweaver.utils import compute_contribution_fn, compute_tau, tau_isosurface


def test_contribution_fn(falc_se, reference):
    atmos, eqPops, ctx, Niter = falc_se
    ctx = copy_ctx(ctx)

    # Ask the formal solver to store the full depth-dependent chi
    # and eta, then perform a formal solution.
    ctx.depthData.fill = True
    ctx.formal_sol_gamma_matrices()

    # The contribution function for the outgoing ray with the
    # largest mu (the last angle in the quadrature) [Nwave, Nspace].
    mu = -1
    cfn = compute_contribution_fn(ctx, mu=mu)
    assert np.all(np.isfinite(cfn))
    assert np.all(cfn >= 0.0)

    # Integrating the contribution function over height recovers
    # the emergent intensity, up to the accuracy of the simple first-order
    # quadrature on this grid: ~3% typically, ~10% at the 95th percentile, and
    # up to ~20% in the Lyman alpha core, where the source function varies
    # rapidly between grid points.
    z = np.asarray(atmos.height)
    I = np.asarray(ctx.spect.I)[:, mu]
    integrated = -np.trapezoid(cfn, z, axis=1)
    rel = np.abs(integrated - I) / I
    assert np.median(rel) < 0.05
    assert np.percentile(rel, 95) < 0.15
    assert rel.max() < 0.25

    # The 854.2 core forms far above the nearby continuum.
    wavelength = np.asarray(ctx.spect.wavelength)
    core = np.argmin(np.abs(wavelength - Ca8542))
    cont = np.argmin(np.abs(wavelength - 850.0))
    tau = compute_tau(ctx, mu=mu)
    tau1 = tau_isosurface(tau, z)
    assert tau1[core] > tau1[cont] + 500e3
    assert z[np.argmax(cfn[core])] > z[np.argmax(cfn[cont])] + 500e3

    reference.check('contribution_fn/cfn_8542', cfn[core])
    reference.check('contribution_fn/tau1', tau1[[core, cont]])
