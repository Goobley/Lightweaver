"""
Tests for `python -m lightweaver.migrate`.
"""

import pytest

from lightweaver.migrate import build_kwarg_registry, main, migrate_source

OLD = """\
import lightweaver as lw

def synth(eqPops, conserveCharge=False):  # User function: untouched.
    return eqPops

eqPops = aSet.compute_eq_pops(atmos)  # Bare names are untouched.
ctx = lw.Context(atmos, spect, eqPops,
                 conserveCharge=True, Nthreads=2)  # Keyword in a Lightweaver call.
synth(eqPops, conserveCharge=True)
lw.iterate_ctx_se(ctx, popsTol=1e-3, NmaxIter=100)
print("λ", ctx.activeAtoms[0].nStar)  # Non-ASCII before attributes on the line.
ctx.atmos.nHTot[:] = 1.0
sd = ctx.state_dict()
sd['kwargs']['eqPops'] = eqPops
newCtx = lw.Context.construct_from_state_dict_with(dict(sd['kwargs'], formalSolver='x'))
ctx.nr_post_update(timeDependentData={'dt': dt, 'nPrev': prevState})
"""

NEW = """\
import lightweaver as lw

def synth(eqPops, conserveCharge=False):  # User function: untouched.
    return eqPops

eqPops = aSet.compute_eq_pops(atmos)  # Bare names are untouched.
ctx = lw.Context(atmos, spect, eqPops,
                 conserve_charge=True, Nthreads=2)  # Keyword in a Lightweaver call.
synth(eqPops, conserveCharge=True)
lw.iterate_ctx_se(ctx, pops_tol=1e-3, max_iter=100)
print("λ", ctx.active_atoms[0].n_star)  # Non-ASCII before attributes on the line.
ctx.atmos.nh_tot[:] = 1.0
sd = ctx.state_dict()
sd['kwargs']['eq_pops'] = eqPops
newCtx = lw.Context.construct_from_state_dict_with(dict(sd['kwargs'], formal_solver='x'))
ctx.nr_post_update(time_dependent_data={'dt': dt, 'n_prev': prevState})
"""


@pytest.fixture(scope='module')
def registry():
    return build_kwarg_registry()


def test_migrate_source(registry):
    new, kw_edits, attr_edits = migrate_source(OLD, registry)
    assert new == NEW
    assert len(attr_edits) == 3


def test_migrate_is_idempotent(registry):
    assert migrate_source(NEW, registry)[0] == NEW


def test_main_dry_run_and_write(tmp_path, capsys):
    path = tmp_path / 'script.py'
    path.write_text(OLD, encoding='utf-8')
    assert main([str(tmp_path)]) == 0
    assert path.read_text(encoding='utf-8') == OLD
    assert '+                 conserve_charge=True' in capsys.readouterr().out
    assert main(['--inplace', str(path)]) == 0
    assert path.read_text(encoding='utf-8') == NEW
