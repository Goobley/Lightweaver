import time

import numpy as np
from tqdm import tqdm
from weno4 import weno4

from lightweaver.config import update_config_file
from .atmosphere import Atmosphere, ScaleType
from .atomic_set import RadiativeSet
from .config import params as rcParams
from .config import get_home_config_path, update_config_file
from .fal import Falc82
from .rh_atoms import CaII_atom, H_6_atom
from .simd_management import get_available_simd_suffixes
from .LwCompiled import LwContext

__all__ = ['benchmark']

def configure_context(Nspace=500, fsIterScheme=None):
    '''
    Configure a FALC context (with more or fewer depth points), 1 thread and a
    particular iteration scheme. For use in benchmarking.

    Parameters
    ----------
    Nspace : int, optional
        Number of spatial points to interpolate the atmosphere to. (Default: 500)
    fsIterScheme : str, optional
        The fsIterScheme to use in the Context. (Default: None, i.e. read from
        the user's config)
    '''
    fal = Falc82()
    interp = lambda x: weno4(np.linspace(0,1,Nspace), np.linspace(0,1,fal.Nspace), x)

    atmos = Atmosphere.make_1d(ScaleType.Geometric, interp(fal.height),
                               temperature=interp(fal.temperature), vlos=interp(fal.vlos),\
                               vturb=interp(fal.vturb), ne=interp(fal.ne),
                               nHTot=interp(fal.nHTot))
    atmos.quadrature(5)
    aSet = RadiativeSet([H_6_atom(), CaII_atom()])
    aSet.set_active('H', 'Ca')
    eqPops = aSet.compute_eq_pops(atmos)
    spect = aSet.compute_wavelength_grid()
    ctx = LwContext(atmos, spect, eqPops, fsIterScheme=fsIterScheme)
    return ctx

def benchmark(Niter=50, Nrep=3, verbose=True, writeConfig=True, warmUp=True):
    '''
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
    writeConfig : bool, optional
        Whether to writ the optimal method to the user's config file. (Default:
        True)
    warmUp : bool, optional
        Whether to run a Context first (discarded) to ensure that all numba jit
        code is jitted and warm. (Default: True)
    '''
    timer = time.perf_counter

    if verbose:
        print('This will take a couple of minutes...')

    suffixes = get_available_simd_suffixes()
    suffixes = ['scalar'] + suffixes
    methods = [f'mali_full_precond_{suffix}' for suffix in suffixes]

    if warmUp:
        ctx = configure_context(fsIterScheme=methods[0])
        for _ in range(max(Niter // 5, 10)):
            ctx.formal_sol_gamma_matrices(printUpdate=False)

    timings = [0.0] * len(suffixes)
    it = tqdm(methods * Nrep) if verbose else methods * Nrep
    for idx, method in enumerate(it):
        ctx = configure_context(fsIterScheme=method)
        start = timer()
        for _ in range(Niter):
            ctx.formal_sol_gamma_matrices(printUpdate=False)
        end = timer()
        duration = (end - start)
        timings[idx % len(methods)] += duration

    timings = [t / Nrep for t in timings]
    if verbose:
        for idx, method in enumerate(methods):
            print(f'Timing for method "{method}": {timings[idx]:.3f} s '
                  f'({Niter} iterations, {Nrep} repetitions)')

    if writeConfig:
        minTiming = min(timings)
        minIdx = timings.index(minTiming)
        if verbose:
            print(f'Selecting method: {methods[minIdx]}')

        impl = suffixes[minIdx]
        rcParams['SimdImpl'] = impl

        path = get_home_config_path()
        if verbose:
            print(f'Writing config to \'{path}\'...')
        update_config_file(path)

    if verbose:
        print('Benchmark complete.')
