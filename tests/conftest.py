"""
Shared fixtures for the Lightweaver smoke tests.

The converged FAL-C H + Ca II statistical equilibrium solution is computed
once per session and shared between the tests that only need to read from it.
Tests that modify a simulation should take a copy first (see `copy_ctx`).
"""

import os
from copy import deepcopy

import numpy as np
import pytest
from references import ReferenceStore

import lightweaver as lw
from lightweaver.fal import Falc82
from lightweaver.rh_atoms import CaII_atom, H_6_atom, MgII_atom

Nthreads = 2

# Optionally force a particular iteration scheme (scalar, SSE2, AVX2FMA, AVX512).
if 'LW_TEST_SIMD' in os.environ:
    lw.ConfigDict['SimdImpl'] = os.environ['LW_TEST_SIMD']


def converged_falc_ctx(conserveCharge=False, prd=False, includeMg=False):
    """
    Set up and converge a FAL-C simulation with H and Ca II (and optionally
    Mg II) active. This is the standard Lightweaver workflow:
      - construct an atmosphere and its angular quadrature,
      - construct a RadiativeSet of the model atoms, and choose which are active,
      - compute the wavelength grid and the LTE (starting) populations,
      - construct the Context and iterate it to statistical equilibrium.
    """
    atmos = Falc82()
    atmos.quadrature(3)
    atoms = [H_6_atom(), CaII_atom()]
    active = ['H', 'Ca']
    if includeMg:
        atoms.append(MgII_atom())
        active.append('Mg')
    aSet = lw.RadiativeSet(atoms)
    aSet.set_active(*active)
    spect = aSet.compute_wavelength_grid()
    eqPops = aSet.compute_eq_pops(atmos)
    ctx = lw.Context(atmos, spect, eqPops, Nthreads=Nthreads, conserveCharge=conserveCharge)
    Niter = lw.iterate_ctx_se(ctx, prd=prd, popsTol=1e-3, quiet=True)
    return atmos, eqPops, ctx, Niter


def copy_ctx(ctx, **kwargs):
    """
    Construct an independent copy of a Context (including its atmosphere and
    populations), optionally replacing parts of it, see
    `Context.construct_from_state_dict_with`.
    """
    newCtx = ctx.construct_from_state_dict_with(deepcopy(ctx.state_dict()), **kwargs)
    # J is only carried over when the wavelength grid changes, so copy
    # it across to keep the radiation field consistent with the populations.
    if 'spect' not in kwargs:
        np.asarray(newCtx.spect.J)[:] = np.asarray(ctx.spect.J)
    return newCtx


@pytest.fixture(scope='session')
def falc_se():
    """
    Converged CRD FAL-C solution with H_6 and Ca II active. Do not modify.
    """
    return converged_falc_ctx()


@pytest.fixture(scope='session')
def reference():
    store = ReferenceStore()
    yield store
    store.save()


def pytest_report_header(config):
    return f'lightweaver {lw.__version__}, SimdImpl: {lw.ConfigDict["SimdImpl"]}'
