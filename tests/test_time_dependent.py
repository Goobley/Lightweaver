"""
Time-dependent populations: perturb the temperature of a converged model and
evolve the populations in time with the implicit kinetic equation solver.

Demonstrates `Context.update_deps` after modifying the atmosphere, and
iterating `Context.time_dep_update` to convergence within each timestep.

Only Ca II is active here (H is treated in LTE as a background species): with
H active, the hydrogen ionisation balance of the chromosphere relaxes over
hundreds to thousands of seconds, which is too slow for a quick test.
"""

import numpy as np
from conftest import Nthreads, copy_ctx

import lightweaver as lw
from lightweaver.fal import Falc82
from lightweaver.rh_atoms import CaII_atom, H_6_atom


def converged_ca_ctx():
    atmos = Falc82()
    atmos.quadrature(3)
    rad_set = lw.RadiativeSet([H_6_atom(), CaII_atom()])
    rad_set.set_active('Ca')
    spect = rad_set.compute_wavelength_grid()
    eq_pops = rad_set.compute_eq_pops(atmos)
    ctx = lw.Context(atmos, spect, eq_pops, Nthreads=Nthreads)
    lw.iterate_ctx_se(ctx, pops_tol=1e-4, quiet=True)
    return ctx


def time_step(ctx, dt, pops_tol=1e-3, max_sub_iter=500):
    """
    Advance the populations of the active atoms by dt [s]. Within a timestep,
    the radiation field and the populations are iterated together until
    consistent. The populations at the start of the step (`prev_time_pops`) are
    returned by the first `time_dep_update` and passed back in subsequently.
    """
    prev_time_pops = None
    for sub in range(max_sub_iter):
        ctx.formal_sol_gamma_matrices()
        update, prev_time_pops = ctx.time_dep_update(dt, prev_time_pops)
        if update.dpops_max < pops_tol:
            return sub
    raise RuntimeError(f'Timestep (dt = {dt} s) did not converge')


def heated_copy(ctx):
    """
    A copy of the converged context with a band of the chromosphere
    (~800-1500 km) heated by 10%, and the dependent quantities (LTE
    populations, line profiles, background opacities) updated accordingly.
    """
    ctx = copy_ctx(ctx)
    atmos = ctx.kwargs['atmos']
    atmos.temperature[40:50] *= 1.1
    ctx.update_deps(temperature=True)
    return ctx


def ca_pops(ctx):
    return np.array(ctx.active_atoms[0].n)


def max_rel_diff(a, b):
    return np.max(np.abs(a - b) / b)


def test_time_dependent(reference):
    ctx = converged_ca_ctx()

    # The statistical equilibrium solution of the heated atmosphere, which
    # the time-dependent populations should relax towards.
    se_ctx = heated_copy(ctx)
    lw.iterate_ctx_se(se_ctx, pops_tol=1e-4, quiet=True)
    heated_se = ca_pops(se_ctx)
    start = ca_pops(heated_copy(ctx))
    assert max_rel_diff(start, heated_se) > 0.1

    # Evolve with a solar-like timestep of 0.1 s: the populations move
    # steadily towards the new equilibrium, conserving the total Ca
    # population.
    evolve_ctx = heated_copy(ctx)
    distance = [max_rel_diff(start, heated_se)]
    for step in range(10):
        time_step(evolve_ctx, 0.1)
        n = ca_pops(evolve_ctx)
        assert np.all(np.isfinite(n))
        np.testing.assert_allclose(
            n.sum(axis=0), np.asarray(evolve_ctx.active_atoms[0].n_total), rtol=1e-6
        )
        distance.append(max_rel_diff(n, heated_se))
        if step == 0:
            first_step = n
    # The fastest-responding populations can overshoot slightly in the
    # first step, after which the approach is monotonic.
    assert np.all(np.diff(distance[1:]) < 0.0)
    assert distance[-1] < distance[0]
    reference.check('time_dependent/CaPops_1s', ca_pops(evolve_ctx))

    # A shorter timestep changes the populations less.
    short_ctx = heated_copy(ctx)
    time_step(short_ctx, 1e-3)
    assert max_rel_diff(ca_pops(short_ctx), start) < max_rel_diff(first_step, start)

    # Repeated 10 s steps relax the populations to the new statistical
    # equilibrium.
    for step in range(10):
        time_step(evolve_ctx, 10.0)
        if max_rel_diff(ca_pops(evolve_ctx), heated_se) < 0.01:
            break
    assert max_rel_diff(ca_pops(evolve_ctx), heated_se) < 0.01
