import warnings
from copy import copy
from os import path
from typing import List, Optional

import astropy.config as conf
import yaml

from lightweaver.simd_management import LwSimdImplsAndFlags, get_available_simd_suffixes

from .deprecation import accepts_old_kwargs

Defaults = {
    'FormalSolver1d': 'piecewise_bezier3_1d',
    'FormalSolver2d': 'piecewise_besser_2d',
    'IterationScheme': 'mali_full_precond',
    'SimdImpl': 'scalar',
}

params = copy(Defaults)


def get_home_config_path() -> str:
    """
    Return the location where the user's configuration data *should* be stored,
    whether it is currently present or not.
    """
    confDir = conf.get_config_dir('lightweaver')
    homePath = path.join(confDir, 'lightweaverrc')
    return homePath


def get_config_path() -> Optional[str]:
    """
    Returns the path to the `lightweaverrc` configuration file, or None if one
    cannot be found.
    """
    localPath = 'lightweaverrc'
    if path.isfile(localPath):
        return localPath
    homePath = get_home_config_path()
    if path.isfile(homePath):
        return homePath

    return None


def set_most_advanced_simd_impl():
    """
    Picks the most advanced SIMD extensions (as detected by NumPy), that can be
    used on this system. This does not guarantee fastest (see the note in
    `update_config_dict`).
    """
    availableImpls = get_available_simd_suffixes()

    def check_add_impl(simd_type):
        if simd_type in availableImpls:
            params['SimdImpl'] = simd_type

    for impl in LwSimdImplsAndFlags:
        check_add_impl(impl)


@accepts_old_kwargs
def update_config_dict(config_path: Optional[str]):
    """
    Updates the configuration dict (`lightweaver.ConfigDict`), from the config
    file. If there is no config file, the defaults are used, and the most
    advanced instruction set is chosen for the SimdImpl. If the SimdImpl in the
    config file is too advanced for the current CPU, the maximum available is
    chosen.

    Parameters
    ----------
    config_path :  str, optional
        The path to the config file, or None.
    """
    if config_path is None:
        warnings.warn(
            'No config file found, using defaults. For optimised vectorised code,'
            ' please run `lightweaver.benchmark()`, otherwise the most advanced'
            ' instruction set supported by your machine will be picked, which may'
            ' not be the fastest (due to e.g. aggressive AVX offsets).'
        )
        set_most_advanced_simd_impl()
        return

    with open(config_path, 'r') as f:
        confDict = yaml.safe_load(f)
    params.update(confDict)

    availableSimd: List[str] = get_available_simd_suffixes()
    if params['SimdImpl'] not in ['scalar'] + availableSimd:
        set_most_advanced_simd_impl()
        warnings.warn(
            'SimdImpl was set to an overly advanced instruction set for the '
            'current CPU, setting to the maximum supported by your CPU.'
        )


@accepts_old_kwargs
def update_config_file(config_path: str):
    """
    Updates the config file to the current values of the config dict.

    Parameters
    ----------
    config_path : str
        The path to the config file.
    """
    with open(config_path, 'w') as f:
        yaml.safe_dump(params, f)


update_config_dict(get_config_path())
