from .atmosphere import Atmosphere, ScaleType, BoundaryCondition
from .atomic_set import SpectrumConfiguration, RadiativeSet, lte_pops, hminus_pops
from .atomic_table import atomic_weight_sort, AtomicTable, get_global_atomic_table, set_global_atomic_table
from .constants import *
from .molecule import MolecularTable
from .multi import read_multi_atmos
from .utils import NgOptions, InitialSolution, voigt_H, planck, gaunt_bf, ConvergenceError, ExplodingMatrixError, get_data_path, get_default_molecule_path, vac_to_air, air_to_vac, convert_specific_intensity, CrswIterator, UnityCrswIterator
from .nr_update import nr_post_update, F
from .LwCompiled import LwContext

def Context(*args, **kwargs):
    import types
    ctx = LwContext(*args, **kwargs)
    ctx.nr_post_update = types.MethodType(nr_post_update, ctx)
    ctx.F = types.MethodType(F, ctx)
    return ctx