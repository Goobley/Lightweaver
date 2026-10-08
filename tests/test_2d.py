"""
2D: solve statistical equilibrium in a horizontally homogeneous 2D slab built
from FAL-C, with periodic horizontal boundaries, and compare against the
equivalent 1D solution.

Demonstrates `lw.Atmosphere.make_2d`, horizontal boundary conditions, and
reshaping the flattened spatial axis of 2D results to [z, x].
"""

import numpy as np
from conftest import Nthreads
from test_stat_eq import Ca8542

import lightweaver as lw
from lightweaver.fal import Falc82
from lightweaver.rh_atoms import CaII_atom, H_6_atom

Nx = 8


def converged_ca(atmos):
    """
    Ca II active (H in LTE as a background species), iterated to statistical
    equilibrium.
    """
    aSet = lw.RadiativeSet([H_6_atom(), CaII_atom()])
    aSet.set_active('Ca')
    spect = aSet.compute_wavelength_grid()
    eqPops = aSet.compute_eq_pops(atmos)
    ctx = lw.Context(atmos, spect, eqPops, Nthreads=Nthreads)
    lw.iterate_ctx_se(ctx, popsTol=1e-3, quiet=True)
    return eqPops, ctx


def test_2d_homogeneous_slab(reference):
    fal = Falc82()

    # 2D atmospheres are specified on a geometric height grid, with
    # all quantities as [z, x] arrays. Here every column is the FAL-C model.
    def tile(a):
        return np.ascontiguousarray(np.repeat(np.asarray(a)[:, None], Nx, axis=1))

    x = np.arange(Nx) * 50e3
    zero = np.zeros((fal.Nspace, Nx))
    atmos2d = lw.Atmosphere.make_2d(
        height=np.copy(fal.z),
        x=x,
        temperature=tile(fal.temperature),
        vx=zero,
        vz=zero,
        vturb=tile(fal.vturb),
        ne=tile(fal.ne),
        nHTot=tile(fal.nHTot),
        xLowerBc=lw.PeriodicRadiation(),
        xUpperBc=lw.PeriodicRadiation(),
    )
    atmos2d.quadrature(6)
    eqPops2d, ctx2d = converged_ca(atmos2d)

    # The spatial axis is flattened in [z, x] order.
    Nz = atmos2d.Nz
    ca2d = np.asarray(eqPops2d['Ca']).reshape(-1, Nz, Nx)
    assert np.all(np.isfinite(ca2d))

    # The slab is horizontally homogeneous with periodic boundaries,
    # so every column should have the same solution, up to small (~0.3%)
    # differences from the treatment of rays crossing the periodic boundary.
    colRef = ca2d[:, :, Nx // 2 : Nx // 2 + 1]
    np.testing.assert_allclose(ca2d, np.broadcast_to(colRef, ca2d.shape), rtol=5e-3)

    # ... which is close to the 1D plane-parallel solution. The angular
    # quadratures differ, which matters most at the top of the transition
    # region (up to ~6% there, ~1% typically).
    atmos1d = Falc82()
    atmos1d.quadrature(3)
    eqPops1d, ctx1d = converged_ca(atmos1d)
    rel = np.abs(colRef[..., 0] - eqPops1d['Ca']) / eqPops1d['Ca']
    assert np.median(rel) < 0.02
    assert rel.max() < 0.1

    wavelength = np.asarray(ctx2d.spect.wavelength)
    core = np.argmin(np.abs(wavelength - Ca8542))
    J = np.asarray(ctx2d.spect.J).reshape(-1, Nz, Nx)
    reference.check('2d/CaPops_column', colRef[..., 0])
    reference.check('2d/J_8542_column', J[core, :, Nx // 2])
