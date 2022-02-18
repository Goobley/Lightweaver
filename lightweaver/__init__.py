from .atmosphere import (Atmosphere, BoundaryCondition, Layout, NoBc,
                         PeriodicRadiation, ScaleType, Stratifications,
                         ThermalisedRadiation, ZeroRadiation)
from .atomic_model import reconfigure_atom
from .atomic_set import (RadiativeSet, SpectrumConfiguration, hminus_pops,
                         lte_pops)
from .atomic_table import (AtomicAbundance, DefaultAtomicAbundance, Element,
                           Isotope, KuruczPfTable, PeriodicTable)
from .benchmark import benchmark
from .config import params as ConfigDict
from .constants import *
from .iterate_ctx import (ConvergenceCriteria, DefaultConvergenceCriteria,
                          iterate_ctx_se)
from .iteration_update import IterationUpdate
from .LwCompiled import LwContext
from .molecule import MolecularTable
from .multi import read_multi_atmos
from .nr_update import nr_post_update
from .utils import (ConvergenceError, CrswIterator, ExplodingMatrixError,
                    InitialSolution, NgOptions, UnityCrswIterator, air_to_vac,
                    compute_contribution_fn, compute_height_edges,
                    compute_radiative_losses, compute_wavelength_edges,
                    convert_specific_intensity, gaunt_bf, get_data_path,
                    get_default_molecule_path, integrate_line_losses, planck,
                    vac_to_air, voigt_H)
from .version import version as __version__


# NOTE(cmo): This is here to make it easier to retroactively monkeypatch
class Context(LwContext):
    pass

setattr(Context, 'nr_post_update', nr_post_update)
