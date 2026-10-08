"""
Statistical equilibrium: iterate a FAL-C model with H and Ca II active to
convergence, then synthesise line profiles.

Demonstrates `lw.iterate_ctx_se`, `Context.compute_rays`, PRD
(`prd=True`) and charge conservation (`conserveCharge=True`).
"""

import numpy as np
from conftest import converged_falc_ctx

NmaxIter = 2000


def line_wavelengths(lambda0, halfWidth, N=101):
    """
    A uniform wavelength grid [nm] around a line core.
    """
    return np.linspace(lambda0 - halfWidth, lambda0 + halfWidth, N)


# Vacuum line centres [nm] from the rh_atoms models.
CaK = 393.478
MgK = 279.635
MgH = 280.353
Ca8542 = 854.444
Halpha = 656.469


def check_pops(eqPops, element):
    """
    Populations should be finite, positive and sum to the total population of
    the species.
    """
    atom = eqPops.atomicPops[element]
    assert np.all(np.isfinite(atom.n))
    assert np.all(atom.n > 0.0)
    np.testing.assert_allclose(atom.n.sum(axis=0), atom.nTotal, rtol=1e-10)


def test_crd_h_ca(falc_se, reference):
    atmos, eqPops, ctx, Niter = falc_se
    assert Niter < NmaxIter - 1

    check_pops(eqPops, 'H')
    check_pops(eqPops, 'Ca')

    # Synthesise the emergent profiles at disk centre (mu = 1) on a
    # wavelength grid of our choice.
    profiles = {}
    for name, lambda0, halfWidth in [
        ('CaK', CaK, 0.1),
        ('Ca8542', Ca8542, 0.1),
        ('Halpha', Halpha, 0.2),
    ]:
        wave = line_wavelengths(lambda0, halfWidth)
        profiles[name] = ctx.compute_rays(wave, [1.0])
        assert np.all(np.isfinite(profiles[name]))
        assert np.all(profiles[name] > 0.0)
        reference.check(f'stat_eq/{name}', profiles[name])

    # 854.2 is a strong absorption line: the core should be much
    # darker than the far wing.
    wing = ctx.compute_rays(np.array([Ca8542 + 1.5]), [1.0])
    coreRatio = profiles['Ca8542'][50] / wing
    assert 0.1 < coreRatio < 0.5

    reference.check('stat_eq/CaPops', eqPops['Ca'])
    reference.check('stat_eq/HPops', eqPops['H'])


def test_prd_charge_conservation(reference):
    # Mg II h & k, Ca II H & K, and Lyman alpha & beta are PRD lines in these
    # models; `prd=True` iterates the PRD redistribution alongside the
    # populations, and `conserveCharge=True` updates the electron density
    # self-consistently.
    atmos, eqPops, ctx, Niter = converged_falc_ctx(conserveCharge=True, prd=True, includeMg=True)
    assert Niter < NmaxIter - 1

    check_pops(eqPops, 'H')
    check_pops(eqPops, 'Ca')
    check_pops(eqPops, 'Mg')
    ne = np.asarray(atmos.ne)
    assert np.all(np.isfinite(ne))
    assert np.all(ne > 0.0)

    for name, lambda0 in [('CaK', CaK), ('MgK', MgK), ('MgH', MgH)]:
        wave = line_wavelengths(lambda0, 0.1)
        profile = ctx.compute_rays(wave, [1.0])
        assert np.all(np.isfinite(profile))
        reference.check(f'stat_eq_prd/{name}', profile)

    # Mg II h & k are self-reversed: the core (k3) is darker than the
    # emission peaks (k2) either side of it.
    wave = line_wavelengths(MgK, 0.03, N=61)
    k = ctx.compute_rays(wave, [1.0])
    assert k[30] < k[:30].max() and k[30] < k[31:].max()

    reference.check('stat_eq_prd/ne', ne)
