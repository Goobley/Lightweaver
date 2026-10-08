"""
Tests for the deprecated (pre-1.0 camelCase) names: every renamed name has a
shim, the shims forward and warn correctly, and they can be silenced.
"""

import dataclasses
import importlib
import inspect
import pkgutil
import warnings
from fractions import Fraction

import numpy as np
import pytest
from conftest import copy_ctx

import lightweaver as lw
import lightweaver.rh_atoms as rh_atoms
from lightweaver._renames import INTERNAL_NO_ALIAS, NR_DICT_KEY_RENAMES, RENAMES
from lightweaver.deprecation import (
    LightweaverDeprecationWarning,
    accepts_old_kwargs,
    deprecated_alias,
    remap_old_keys,
)
from lightweaver.migrate import build_kwarg_registry


def this_file():
    # The path warnings report for code in this module (__file__ can differ when
    # pytest reuses a cached compiled module from another checkout path).
    return inspect.currentframe().f_code.co_filename


def lightweaver_classes():
    for mod in pkgutil.iter_modules(lw.__path__):
        try:
            module = importlib.import_module(f'lightweaver.{mod.name}')
        except ImportError:
            continue
        for obj in vars(module).values():
            if inspect.isclass(obj) and obj.__module__.startswith('lightweaver'):
                yield obj
    yield lw.Context


def test_every_rename_has_a_shim():
    registry = build_kwarg_registry()
    kwarg_names = set().union(*registry.values())
    alias_names = {
        name
        for cls in lightweaver_classes()
        for name, value in vars(cls).items()
        if isinstance(value, deprecated_alias)
    }
    missing = set(RENAMES) - INTERNAL_NO_ALIAS - kwarg_names - alias_names
    assert not missing


def test_dataclass_constructors_accept_old_names():
    # Each dataclass generates its own __init__, so every one with a renamed
    # field (including inherited fields) needs its __init__ wrapped.
    new_to_old = {}
    for old, new in RENAMES.items():
        new_to_old.setdefault(new, set()).add(old)
    for cls in lightweaver_classes():
        if not dataclasses.is_dataclass(cls):
            continue
        renamed = {f.name for f in dataclasses.fields(cls) if f.name in new_to_old and f.init}
        if not renamed:
            continue
        accepted = getattr(cls.__init__, '__lw_renamed_kwargs__', {})
        for new in renamed:
            assert new_to_old[new] <= accepted.keys(), (cls, new)


def test_old_kwarg_forwards_and_warns():
    with pytest.warns(LightweaverDeprecationWarning, match='`qCore` is deprecated, use `q_core`'):
        q = lw.atomic_model.LinearCoreExpWings(qCore=2.0, q_wing=30.0, Nlambda=15)
    assert q.q_core == 2.0


def test_old_and_new_kwarg_raises():
    with pytest.raises(TypeError, match='both `qCore` and its replacement `q_core`'):
        lw.atomic_model.LinearCoreExpWings(qCore=2.0, q_core=3.0, q_wing=30.0, Nlambda=15)


def test_old_attribute_forwards_and_warns():
    ng = lw.NgOptions()
    with pytest.warns(LightweaverDeprecationWarning, match='`lowerThreshold`'):
        assert ng.lowerThreshold == ng.lower_threshold
    with pytest.warns(LightweaverDeprecationWarning, match='`lowerThreshold`'):
        ng.lowerThreshold = 1e-3
    assert ng.lower_threshold == 1e-3


def test_warning_points_at_caller():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        lw.NgOptions().lowerThreshold  # noqa: B018
    assert w[0].filename == this_file()


def test_warns_once_per_call_site():
    ng = lw.NgOptions()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('default')
        for _ in range(3):
            ng.lowerThreshold  # noqa: B018
    assert len(w) == 1


def test_silence_deprecations():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        lw.silence_deprecations()
        lw.NgOptions().lowerThreshold  # noqa: B018
    assert not w


def test_accepts_old_kwargs_merges_old_names():
    # Several old names can map to one new name (NmaxIter and maxIter).
    @accepts_old_kwargs
    def f(max_iter=1):
        return max_iter

    for old in ('maxIter', 'NmaxIter'):
        with pytest.warns(LightweaverDeprecationWarning):
            assert f(**{old: 5}) == 5


def test_remap_old_keys_old_value_wins():
    with pytest.warns(LightweaverDeprecationWarning, match='`formalSolver`'):
        d = remap_old_keys({'formal_solver': 'a', 'formalSolver': 'b', 'atmos': 1})
    assert d == {'formal_solver': 'b', 'atmos': 1}


def test_remap_old_keys_with_table():
    with pytest.warns(LightweaverDeprecationWarning, match='`nPrev` is deprecated, use `n_prev`'):
        d = remap_old_keys({'dt': 1.0, 'nPrev': [1]}, renames=NR_DICT_KEY_RENAMES)
    assert d == {'dt': 1.0, 'n_prev': [1]}


def test_nr_post_update_old_dict_key(falc_se):
    # The old `nPrev` key of time_dependent_data is still accepted, and gives the
    # same result as `n_prev`.
    _, _, ctx, _ = falc_se
    results = []
    for key in ('n_prev', 'nPrev'):
        step_ctx = copy_ctx(ctx)
        step_ctx.formal_sol_gamma_matrices()
        _, prev_state = step_ctx.time_dep_update(1.0, None)
        if key == 'nPrev':
            with pytest.warns(LightweaverDeprecationWarning, match='`nPrev`') as w:
                step_ctx.nr_post_update(time_dependent_data={'dt': 1.0, key: prev_state})
            assert w[0].filename == this_file()
        else:
            step_ctx.nr_post_update(time_dependent_data={'dt': 1.0, key: prev_state})
        results.append(np.copy(step_ctx.active_atoms[0].n))
    np.testing.assert_array_equal(results[0], results[1])


def test_context_old_names(falc_se):
    _, _, ctx, _ = falc_se
    with pytest.warns(LightweaverDeprecationWarning, match='`activeAtoms`'):
        assert ctx.activeAtoms[0] is ctx.active_atoms[0]
    with pytest.warns(LightweaverDeprecationWarning, match='`nHTot`'):
        np.testing.assert_array_equal(ctx.atmos.nHTot, ctx.atmos.nh_tot)
    with pytest.warns(LightweaverDeprecationWarning, match='`nStar`'):
        np.testing.assert_array_equal(ctx.active_atoms[0].nStar, ctx.active_atoms[0].n_star)
    with pytest.warns(LightweaverDeprecationWarning, match='`upOnly`'):
        ctx.formal_sol(upOnly=True)


def test_context_constructor_old_names(falc_se):
    atmos, eq_pops, ctx, _ = falc_se
    with pytest.warns(LightweaverDeprecationWarning, match='`conserveCharge`'):
        new_ctx = lw.Context(atmos, ctx.kwargs['spect'], eq_pops, conserveCharge=True)
    assert new_ctx.conserve_charge


def test_repr_round_trip():
    # The atomic model reprs generate source (e.g. rh_atoms.py), so they must
    # use the new names.
    namespace = dict(vars(rh_atoms), Fraction=Fraction)
    model = rh_atoms.H_6_atom()
    with warnings.catch_warnings():
        warnings.simplefilter('error', LightweaverDeprecationWarning)
        rebuilt = eval(repr(model), namespace)
    assert repr(rebuilt) == repr(model)
