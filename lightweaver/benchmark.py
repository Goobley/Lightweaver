import time

import numpy as np
from tqdm import tqdm
from weno4 import weno4

from .atmosphere import Atmosphere, ScaleType
from .atomic_set import RadiativeSet
from .config import get_home_config_path, update_config_file
from .config import params as rcParams
from .deprecation import accepts_old_kwargs
from .fal import Falc82
from .LwCompiled import LwContext
from .rh_atoms import CaII_atom, H_6_atom
from .simd_management import get_available_simd_suffixes

__all__ = ['benchmark']


@accepts_old_kwargs
def configure_context(Nspace=500, fs_iter_scheme=None):
    """
    Configure a FALC context (with more or fewer depth points), 1 thread and a
    particular iteration scheme. For use in benchmarking.

    Parameters
    ----------
    Nspace : int, optional
        Number of spatial points to interpolate the atmosphere to. (Default: 500)
    fs_iter_scheme : str, optional
        The fs_iter_scheme to use in the Context. (Default: None, i.e. read from
        the user's config)
    """
    fal = Falc82()

    def interp(x):
        return weno4(np.linspace(0, 1, Nspace), np.linspace(0, 1, fal.Nspace), x)

    atmos = Atmosphere.make_1d(
        ScaleType.Geometric,
        interp(fal.height),
        temperature=interp(fal.temperature),
        vlos=interp(fal.vlos),
        vturb=interp(fal.vturb),
        ne=interp(fal.ne),
        nh_tot=interp(fal.nh_tot),
    )
    atmos.quadrature(5)
    aSet = RadiativeSet([H_6_atom(), CaII_atom()])
    aSet.set_active('H', 'Ca')
    eq_pops = aSet.compute_eq_pops(atmos)
    spect = aSet.compute_wavelength_grid()
    ctx = LwContext(atmos, spect, eq_pops, fs_iter_scheme=fs_iter_scheme)
    return ctx


@accepts_old_kwargs
def benchmark(Niter=50, Nrep=3, verbose=True, write_config=True, warm_up=True):
    """
    Benchmark the various SIMD implementations for Lightweaver's formal solver
    and iteration functions.

    Parameters
    ----------
    Niter : int, optional
        The number of iterations to use for each scheme. (Default: 50)
    Nrep : int, optional
        The number of repetitions to average for each scheme. (Default: 3)
    verbose : bool, optional
        Whether to print information as the function runs. (Default: True)
    write_config : bool, optional
        Whether to writ the optimal method to the user's config file. (Default:
        True)
    warm_up : bool, optional
        Whether to run a Context first (discarded) to ensure that all numba jit
        code is jitted and warm. (Default: True)
    """
    timer = time.perf_counter

    if verbose:
        print('This will take a couple of minutes...')

    suffixes = get_available_simd_suffixes()
    suffixes = ['scalar'] + suffixes
    methods = [f'mali_full_precond_{suffix}' for suffix in suffixes]

    if warm_up:
        ctx = configure_context(fs_iter_scheme=methods[0])
        for _ in range(max(Niter // 5, 10)):
            ctx.formal_sol_gamma_matrices()

    timings = [0.0] * len(suffixes)
    it = tqdm(methods * Nrep) if verbose else methods * Nrep
    for idx, method in enumerate(it):
        ctx = configure_context(fs_iter_scheme=method)
        start = timer()
        for _ in range(Niter):
            ctx.formal_sol_gamma_matrices()
        end = timer()
        duration = end - start
        timings[idx % len(methods)] += duration

    timings = [t / Nrep for t in timings]
    if verbose:
        for idx, method in enumerate(methods):
            print(
                f'Timing for method "{method}": {timings[idx]:.3f} s '
                f'({Niter} iterations, {Nrep} repetitions)'
            )

    if write_config:
        minTiming = min(timings)
        minIdx = timings.index(minTiming)
        if verbose:
            print(f'Selecting method: {methods[minIdx]}')

        impl = suffixes[minIdx]
        rcParams['SimdImpl'] = impl

        path = get_home_config_path()
        if verbose:
            print(f"Writing config to '{path}'...")
        update_config_file(path)

    if verbose:
        print('Benchmark complete.')
