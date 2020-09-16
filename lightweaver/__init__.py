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
                   air_to_vac, convert_specific_intensity, CrswIterator, UnityCrswIterator
from .nr_update import nr_post_update, F, Ftd
from .LwCompiled import LwContext
from .version import version as __version__

class Context(LwContext):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

setattr(Context, 'nr_post_update', nr_post_update)
setattr(Context, 'F', F)
setattr(Context, 'Ftd', Ftd)
