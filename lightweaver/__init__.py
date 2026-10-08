import inspect

from .atmosphere import (
    Atmosphere,
    BoundaryCondition,
    Layout,
    NoBc,
    PeriodicRadiation,
    ScaleType,
    Stratifications,
    ThermalisedRadiation,
    ZeroRadiation,
)
from .atomic_model import reconfigure_atom
from .atomic_set import RadiativeSet, SpectrumConfiguration, hminus_pops, lte_pops
from .atomic_table import (
    AtomicAbundance,
    DefaultAtomicAbundance,
    Element,
    Isotope,
    KuruczPfTable,
    PeriodicTable,
)
from .benchmark import benchmark
from .config import params as ConfigDict
from .constants import *
from .deprecation import (
    LightweaverDeprecationWarning,
    accepts_old_kwargs,
    silence_deprecations,
)
from .iterate_ctx import ConvergenceCriteria, DefaultConvergenceCriteria, iterate_ctx_se
from .iteration_update import IterationUpdate
from .LwCompiled import LwContext
from .molecule import MolecularTable
from .multi import read_multi_atmos
from .nr_update import nr_post_update
from .utils import (
    ConvergenceError,
    CrswIterator,
    ExplodingMatrixError,
    InitialSolution,
    NgOptions,
    UnityCrswIterator,
    air_to_vac,
    compute_contribution_fn,
    compute_height_edges,
    compute_radiative_losses,
    compute_tau,
    compute_wavelength_edges,
    convert_specific_intensity,
    gaunt_bf,
    get_data_path,
    get_default_molecule_path,
    integrate_line_losses,
    planck,
    tau_isosurface,
    vac_to_air,
    voigt_H,
)
from .version import version as __version__


# NOTE(cmo): This is here to make it easier to retroactively monkeypatch
class Context(LwContext):
    # NOTE: Cython ignores Python decorators on a cdef class's __init__ and doesn't
    # allow them on cpdef methods, so the deprecated keyword names for those are
    # accepted here.
    @accepts_old_kwargs
    def __init__(
        self,
        atmos,
        spect,
        eq_pops,
        ng_options=None,
        init_sol=None,
        conserve_charge=False,
        nr_h_only=False,
        detailed_atom_prd=True,
        hprd=False,
        crsw_callback=None,
        Nthreads=1,
        background_provider=None,
        formal_solver=None,
        interp_fn=None,
        fs_iter_scheme=None,
    ):
        super().__init__(
            atmos,
            spect,
            eq_pops,
            ng_options=ng_options,
            init_sol=init_sol,
            conserve_charge=conserve_charge,
            nr_h_only=nr_h_only,
            detailed_atom_prd=detailed_atom_prd,
            hprd=hprd,
            crsw_callback=crsw_callback,
            Nthreads=Nthreads,
            background_provider=background_provider,
            formal_solver=formal_solver,
            interp_fn=interp_fn,
            fs_iter_scheme=fs_iter_scheme,
        )


def _forward_with_old_kwargs(name):
    base = getattr(LwContext, name)

    def method(self, *args, **kwargs):
        return base(self, *args, **kwargs)

    method.__name__ = name
    method.__qualname__ = f'Context.{name}'
    method.__doc__ = base.__doc__
    method.__signature__ = inspect.signature(base)
    return accepts_old_kwargs(method)


for _name in (
    'formal_sol_gamma_matrices',
    'formal_sol',
    'time_dep_update',
    'time_dep_restore_prev_pops',
    'stat_equil',
    'single_stokes_fs',
    'prd_redistribute',
):
    setattr(Context, _name, _forward_with_old_kwargs(_name))

Context.construct_from_state_dict_with = staticmethod(
    accepts_old_kwargs(LwContext.construct_from_state_dict_with)
)
setattr(Context, 'nr_post_update', nr_post_update)
