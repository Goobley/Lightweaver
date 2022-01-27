from lightweaver.simd_management import get_available_simd_suffixes, LwSimdImplsAndFlags
from copy import copy
import astropy.config as conf
from typing import Optional
import os.path as path
import yaml
import warnings

Defaults = {
    'FormalSolver1d': 'piecewise_bezier3_1d',
    'FormalSolver2d': 'piecewise_besser_2d',
    'IterationScheme': 'mali_full_precond',
    'SimdImpl': 'Scalar',
}

params = copy(Defaults)

def get_home_config_path() -> str:
    '''
    Return the location where the user's configuration data *should* be stored,
    whether it is currently present or not.
    '''
    confDir = conf.get_config_dir('lightweaver')
    homePath = path.join(confDir, 'lightweaverrc')
    return homePath


def get_config_path() -> Optional[str]:
    '''
    Returns the path to the `lightweaverrc` configuration file, or None if one
    cannot be found.
    '''
    localPath = 'lightweaverrc'
    if path.isfile(localPath):
        return localPath
    homePath = get_home_config_path()
    if path.isfile(homePath):
        return homePath

    return None


def set_most_advanced_simd_impl():
    '''
    Picks the most advanced SIMD extensions (as detected by NumPy), that can be
    used on this system. This does not guarantee fastest (see the note in
    `update_config_dict`).
    '''
    availableImpls = get_available_simd_suffixes()

    def check_add_impl(simdType):
        if simdType in availableImpls:
            params['SimdImpl'] = simdType

    for impl in LwSimdImplsAndFlags:
        check_add_impl(impl)


def update_config_dict(configPath: Optional[str]):
    '''
    Updates the configuration dict (`lightweaver.ConfigDict`), from the config
    file. If there is no config file, the defaults are used, and the most
    advanced instruction set is chosen for the SimdImpl.
    '''
    if configPath is None:
        warnings.warn('No config file found, using defaults. For optimised vectorised code, please run `lightweaver.benchmark()`, otherwise the most advanced instruction set supported by your machine will be picked, which may not be the fastest (due to e.g. aggressive AVX offsets).')
        set_most_advanced_simd_impl()
        return

    with open(configPath, 'r') as f:
        confDict = yaml.safe_load(f)
        params.update(confDict)


def update_config_file(configPath: str):
    '''
    Updates the config file to the current values of the config dict.
    '''
    with open(configPath, 'w') as f:
        yaml.safe_dump(params, f)


update_config_dict(get_config_path())