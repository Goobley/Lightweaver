"""
Far-wing treatment of lines in `Context.compute_rays`.

Lines contribute over their original wavelength quadrature (as in the NLTE
solution), and are then smoothly tapered to zero beyond its edges (see
`lightweaver.atomic_set.line_wing_taper`), rather than their Voigt wings
contributing across the entire spectrum.
"""

import numpy as np

import lightweaver as lw
from lightweaver.atomic_set import line_wing_taper

Ca8542 = 854.444


def test_line_wing_taper():
    lambda0 = 500.0
    lambda_blue, lambda_red = 499.0, 502.0  # asymmetric: half-widths 1 and 2
    f = 0.5
    wav = np.linspace(497.0, 505.0, 8001)
    taper, blue, red = line_wing_taper(wav, lambda_blue, lambda_red, lambda0, f)
    sub = wav[blue:red]

    # Support is the quadrature plus f * half-width on each side, with the
    # taper 1 inside the quadrature and 0.5 halfway through each roll-off.
    np.testing.assert_allclose(sub[[0, -1]], [lambda_blue - f * 1.0, lambda_red + f * 2.0])
    assert np.all(taper[(sub >= lambda_blue) & (sub <= lambda_red)] == 1.0)
    np.testing.assert_allclose(np.interp(lambda_blue - 0.25, sub, taper), 0.5, atol=1e-6)
    np.testing.assert_allclose(np.interp(lambda_red + 0.5, sub, taper), 0.5, atol=1e-6)

    # A single point within the support gets a neighbour (with zero taper),
    # so the line always has at least two wavelengths.
    taper, blue, red = line_wing_taper(
        np.array([490.0, 500.0, 510.0]), lambda_blue, lambda_red, lambda0, f
    )
    assert (blue, red) == (1, 3)
    np.testing.assert_array_equal(taper, [1.0, 0.0])


def test_full_range_continuum(falc_se):
    _, _, ctx, _ = falc_se
    spect = ctx.kwargs['spect']
    sub = spect.subset_configuration(spect.wavelength)
    far = np.ones(spect.wavelength.shape, dtype=bool)
    for k in sub.far_wing_tapers:
        far &= ~sub.active_wavelengths[k]
    assert far.sum() > 50

    I_taper = ctx.compute_rays(None, [1.0])
    I_hard = ctx.compute_rays(None, [1.0], wing_taper_fraction=0.0)
    I_legacy = ctx.compute_rays(None, [1.0], wing_taper_fraction=None)

    # Away from all lines, the taper adds nothing: identical to a hard cut at
    # the edge of the quadrature (i.e. the opacity seen by the NLTE solution).
    np.testing.assert_array_equal(I_taper[far], I_hard[far])
    # Whereas the untapered far wings substantially alter the UV continuum.
    uv = far & (spect.wavelength > 100.0) & (spect.wavelength < 300.0)
    rel = np.abs(I_legacy[uv] - I_taper[uv]) / I_taper[uv]
    assert np.median(rel) > 0.1


def test_taper_only_chunk(falc_se):
    # A chunk lying only in a line's taper region still includes the line, so
    # evaluating a spectrum in chunks matches evaluating it in one go.
    _, _, ctx, _ = falc_se
    spect = ctx.kwargs['spect']
    ca = [m for m in spect.models if m.element == lw.PeriodicTable['Ca']][0]
    line = [l for l in ca.lines if abs(l.lambda0 - Ca8542) < 0.1][0]
    lambda_red = line.wavelength()[-1]
    taper_width = 0.5 * (lambda_red - line.lambda0)
    taper_only = np.linspace(lambda_red + 0.1 * taper_width, lambda_red + 0.9 * taper_width, 7)
    assert spect.subset_configuration(taper_only).active_trans[line.trans_id]

    combined = np.sort(np.concatenate([spect.wavelength, taper_only]))
    idx = np.searchsorted(combined, taper_only)
    I_combined = ctx.compute_rays(combined, [1.0])
    I_taper_only = ctx.compute_rays(taper_only, [1.0])
    np.testing.assert_allclose(I_taper_only, I_combined[idx], rtol=1e-12)
