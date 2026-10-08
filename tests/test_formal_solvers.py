"""
Formal solvers: compute the emergent radiation from a converged model with the
different 1D formal solvers, and with the full Stokes solver in a magnetised
atmosphere.

Demonstrates selecting a formal solver, reusing converged populations in a
new Context (`Context.construct_from_state_dict_with`), and
`compute_rays(stokes=True)`.
"""

import pickle
from copy import deepcopy

import numpy as np
import pytest
from conftest import copy_ctx
from lightweaver.LwCompiled import BasicBackground, FastBackground
from test_stat_eq import Ca8542, line_wavelengths

import lightweaver as lw

SOLVERS_1D = ['piecewise_linear_1d', 'piecewise_besser_1d', 'piecewise_bezier3_1d']


def ctx_with_formal_solver(ctx, formal_solver):
    """
    A copy of a converged Context (with its populations), using a different
    formal solver.
    """
    sd = deepcopy(ctx.state_dict())
    sd['kwargs'] = dict(sd['kwargs'], formal_solver=formal_solver)
    new_ctx = ctx.construct_from_state_dict_with(sd)
    np.asarray(new_ctx.spect.J)[:] = np.asarray(ctx.spect.J)
    return new_ctx


def test_1d_formal_solvers(falc_se, reference):
    atmos, eq_pops, ctx, Niter = falc_se
    wave = np.concatenate([[500.0, 800.0], line_wavelengths(Ca8542, 0.1)])

    profiles = {}
    for solver in SOLVERS_1D:
        solver_ctx = ctx_with_formal_solver(ctx, solver)
        profiles[solver] = solver_ctx.compute_rays(wave, [1.0])
        assert np.all(np.isfinite(profiles[solver]))

    # The solvers differ in their order of accuracy, but should
    # agree closely in the continuum and reasonably in the line core.
    ref = profiles['piecewise_besser_1d']
    for solver in SOLVERS_1D:
        rel = np.abs(profiles[solver] - ref) / ref
        assert rel[:2].max() < 0.02, solver
        assert rel.max() < 0.1, solver
        reference.check(f'formal_solvers/{solver}', profiles[solver])


def test_full_stokes(falc_se, reference):
    atmos, eq_pops, ctx, Niter = falc_se

    # Construct a copy of the atmosphere with a uniform magnetic
    # field (B [T], inclination gamma_B and azimuth chi_B [rad]), and reuse the
    # converged populations with it.
    Nspace = atmos.Nspace
    mag_atmos = lw.Atmosphere.make_1d(
        lw.ScaleType.Geometric,
        depth_scale=np.copy(atmos.z),
        temperature=np.copy(atmos.temperature),
        vlos=np.copy(atmos.vz),
        vturb=np.copy(atmos.vturb),
        ne=np.copy(atmos.ne),
        nh_tot=np.copy(atmos.nh_tot),
        B=np.full(Nspace, 0.1),
        gamma_B=np.full(Nspace, 0.6),
        chi_B=np.full(Nspace, 0.3),
    )
    mag_atmos.quadrature(3)
    mag_ctx = copy_ctx(ctx, atmos=mag_atmos)

    # Continuum points away from any polarisable line, one on either
    # side of the line.
    continuum = np.array([500.0, 900.0])
    wave = np.concatenate([continuum, line_wavelengths(Ca8542, 0.1)])
    iquv = mag_ctx.compute_rays(wave, [1.0], stokes=True)
    unpolarised = mag_ctx.compute_rays(wave, [1.0])
    assert np.all(np.isfinite(iquv))

    # Stokes I is close to the unpolarised intensity.
    np.testing.assert_allclose(iquv[0], unpolarised, rtol=0.05)

    # Negligible polarisation in the continuum (only the far wings of
    # Zeeman-split lines contribute there).
    assert np.all(np.abs(iquv[1:, :2]) < 1e-6 * iquv[0, :2])

    # Stokes V is non-zero in the line, and approximately
    # antisymmetric about the line core.
    V = iquv[3, 2:]
    assert np.abs(V).max() > 1e-3 * iquv[0, 2:].max()
    assert np.abs(V + V[::-1]).max() < 0.2 * np.abs(V).max()

    reference.check('formal_solvers/stokes_8542', iquv)


@pytest.mark.parametrize('provider', [BasicBackground, FastBackground])
def test_background_provider_pickle(falc_se, provider):
    # The background providers' pickled state must round-trip.
    atmos, eq_pops, ctx, _ = falc_se
    bg_ctx = lw.Context(atmos, ctx.kwargs['spect'], eq_pops, background_provider=provider)
    restored = pickle.loads(pickle.dumps(bg_ctx))
    for name in ('chi', 'eta', 'sca'):
        np.testing.assert_array_equal(
            np.asarray(getattr(restored.background, name)),
            np.asarray(getattr(bg_ctx.background, name)),
        )
