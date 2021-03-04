from .atmosphere import Atmosphere, ScaleType, BoundaryCondition, NoBc, ZeroRadiation, \
                        ThermalisedRadiation, PeriodicRadiation, Stratifications, Layout
from .atomic_set import SpectrumConfiguration, RadiativeSet, lte_pops, hminus_pops
from .atomic_table import PeriodicTable, AtomicAbundance, KuruczPfTable, \
                          DefaultAtomicAbundance, Element, Isotope
from .constants import *
from .molecule import MolecularTable
from .multi import read_multi_atmos
from .utils import NgOptions, InitialSolution, voigt_H, planck, gaunt_bf, ConvergenceError, \
                   ExplodingMatrixError, get_data_path, get_default_molecule_path, vac_to_air, \
                   air_to_vac, convert_specific_intensity, CrswIterator, UnityCrswIterator, \
                   compute_radiative_losses, integrate_line_losses, compute_contribution_fn, \
                   compute_height_edges, compute_wavelength_edges, grotrian_diagram
from .nr_update import nr_post_update
from .LwCompiled import LwContext
from .version import version as __version__

# NOTE(cmo): This is here to make it easier to retroactively monkeypatch
class Context(LwContext):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

setattr(Context, 'nr_post_update', nr_post_update)
