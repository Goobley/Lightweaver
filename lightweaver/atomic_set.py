from copy import copy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union, cast

import astropy.units as u
import numpy as np
from numba import njit
from scipy.linalg import solve
from scipy.optimize import newton_krylov

import lightweaver.constants as Const

from .atmosphere import Atmosphere
from .atomic_model import AtomicModel, LineType, element_sort
from .atomic_table import (
    AtomicAbundance,
    DefaultAtomicAbundance,
    Element,
    KuruczPf,
    KuruczPfTable,
    PeriodicTable,
)
from .deprecation import accepts_old_kwargs, deprecated_names
from .molecule import MolecularTable


@njit(cache=True)
def lte_pops_impl(
    temperature, ne, n_total, stages, energies, gs, n_star=None, debye=True, compute_diff=False
):
    Nlevel = stages.shape[0]
    Nspace = ne.shape[0]
    c1 = (Const.HPlanck / (2.0 * np.pi * Const.MElectron)) * (Const.HPlanck / Const.KBoltzmann)

    c2 = 0.0
    nDebye = np.zeros(Nlevel)
    if debye:
        c2 = (
            np.sqrt(8.0 * np.pi / Const.KBoltzmann)
            * (Const.QElectron**2 / (4.0 * np.pi * Const.Epsilon0)) ** 1.5
        )
        for i in range(1, Nlevel):
            stage = stages[i]
            Z = stage
            for m in range(1, stage - stages[0] + 1):
                nDebye[i] += Z
                Z += 1

    if n_star is None:
        n_star = np.empty((Nlevel, Nspace))

    if compute_diff:
        prev = np.empty(Nlevel)
    # NOTE(cmo): Will remain 0 and be returned as second return value if
    # compute_diff is not set to True
    maxDiff = 0.0

    # NOTE(cmo): For some reason this is consistently faster with these hoisted
    # from below, despite the allocations this causes
    dE = energies - energies[0]
    gi0 = gs / gs[0]
    dZ = stages - stages[0]

    for k in range(Nspace):
        if debye:
            dEion = c2 * np.sqrt(ne[k] / temperature[k])
        else:
            dEion = 0.0
        cNe_T = 0.5 * ne[k] * (c1 / temperature[k]) ** 1.5
        total = 1.0
        if compute_diff:
            for i in range(Nlevel):
                prev[i] = n_star[i, k]
        for i in range(1, Nlevel):
            dE_kT = (dE[i] - nDebye[i] * dEion) / (Const.KBoltzmann * temperature[k])
            neFactor = cNe_T ** dZ[i]

            nst = gi0[i] * np.exp(-dE_kT)
            n_star[i, k] = nst
            n_star[i, k] /= neFactor
            total += n_star[i, k]
        n_star[0, k] = n_total[k] / total

        for i in range(1, Nlevel):
            n_star[i, k] *= n_star[0, k]

        if compute_diff:
            for i in range(Nlevel):
                maxDiff = max(abs((n_star[i, k] - prev[i]) / n_star[i, k]), maxDiff)

    return n_star, maxDiff


@accepts_old_kwargs
def lte_pops(
    atomic_model: AtomicModel,
    temperature: np.ndarray,
    ne: np.ndarray,
    n_total: np.ndarray,
    n_star=None,
    debye: bool = True,
) -> np.ndarray:
    """
    Compute the LTE populations for a given atomic model under given
    thermodynamic conditions.

    Parameters
    ----------
    atomic_model : AtomicModel
        The atomic model to consider.
    temperature : np.ndarray
        The temperature structure in the atmosphere.
    ne : np.ndarray
        The electron density in the atmosphere.
    n_total : np.ndarray
        The total population of the species at each point in the atmosphere.
    n_star : np.ndarray, optional
        An optional array to store the result in.
    debye : bool, optional
        Whether to consider Debye shielding (default: True).1

    Returns
    -------
    ltePops : np.ndarray
        The ltePops for the species.
    """
    stages = np.array([l.stage for l in atomic_model.levels])
    energies = np.array([l.E_SI for l in atomic_model.levels])
    gs = np.array([l.g for l in atomic_model.levels])
    return lte_pops_impl(
        temperature, ne, n_total, stages, energies, gs, n_star=n_star, debye=debye
    )[0]


@accepts_old_kwargs
def update_lte_pops_inplace(
    atomic_model: AtomicModel,
    temperature: np.ndarray,
    ne: np.ndarray,
    n_total: np.ndarray,
    n_star: np.ndarray,
    debye: bool = True,
) -> Tuple[np.ndarray, float]:
    stages = np.array([l.stage for l in atomic_model.levels])
    energies = np.array([l.E_SI for l in atomic_model.levels])
    gs = np.array([l.g for l in atomic_model.levels])
    return lte_pops_impl(
        temperature,
        ne,
        n_total,
        stages,
        energies,
        gs,
        debye=debye,
        n_star=n_star,
        compute_diff=True,
    )


@deprecated_names(attrs=('atomic_pops', 'n_total', 'nh_tot', 'nlte_starting_pops', 'sorted_atoms'))
class LteNeIterator:
    def __init__(
        self,
        atoms: Iterable[AtomicModel],
        temperature: np.ndarray,
        nh_tot: np.ndarray,
        abundance: AtomicAbundance,
        nlte_starting_pops: Dict[Element, np.ndarray],
    ):
        sorted_atoms = sorted(atoms, key=element_sort)
        self.n_total = [abundance[a.element] * nh_tot for a in sorted_atoms]
        self.stages = [np.array([l.stage for l in a.levels]) for a in sorted_atoms]
        self.temperature = temperature
        self.nh_tot = nh_tot
        self.sorted_atoms = sorted_atoms
        self.abundances = [abundance[a.element] for a in sorted_atoms]
        self.nlte_starting_pops = nlte_starting_pops

    def __call__(self, prev_ne_ratio: np.ndarray) -> np.ndarray:
        atomic_pops = []
        ne = np.zeros_like(prev_ne_ratio)
        prevNe = prev_ne_ratio * self.nh_tot

        for i, a in enumerate(self.sorted_atoms):
            n_star = lte_pops(a, self.temperature, prevNe, self.n_total[i], debye=True)
            atomic_pops.append(
                AtomicState(
                    model=a, abundance=self.abundances[i], n_star=n_star, n_total=self.n_total[i]
                )
            )
            # NOTE(cmo): Take into account NLTE pops if provided
            if a.element in self.nlte_starting_pops:
                if self.nlte_starting_pops[a.element].shape != n_star.shape:
                    raise ValueError(
                        ('Starting populations provided for %s do not match model.') % a.element
                    )
                n_star = self.nlte_starting_pops[a.element]

            ne += np.sum(n_star * self.stages[i][:, None], axis=0)

        self.atomic_pops = atomic_pops
        diff = (ne - prevNe) / self.nh_tot
        return diff


@deprecated_names
@dataclass
class SpectrumConfiguration:
    """
    Container for the configuration of common wavelength grid and species
    active at each wavelength.

    Attributes
    ----------
    rad_set : RadiativeSet
        The set of atoms involved in the creation of this simulation.
    wavelength : np.ndarray
        The common wavelength array used for this simulation.
    models : list of AtomicModel
        The models for the active and detailed static atoms present in this
        simulation.
    trans_wavelengths : Dict[(Element, i, j), np.ndarray]
        The local wavelength grid for each transition stored in a dictionary
        by transition ID.
    blue_idx : Dict[(Element, i, j), int]
        The index at which each local grid starts in the global wavelength
        array.
    red_idx : Dict[(Element, i, j), int]
        The index at which each local grid has ended in the global wavelength
        array (exclusive,
        i.e. transWavelength = globalWavelength[blue_idx:red_idx]).
    active_trans : Dict[(Element, i, j), bool]
        Whether this transition is ever active (contributing in either an
        active or detailed static sense) over the range of wavelength.
    active_wavelengths : Dict[(Element, i, j), np.ndarray]
        A mask of the wavelengths at which this transition is active.

    Properties
    ----------
    Nprd_trans : int
        The number of PRD transitions present on the active transitions.
    """

    rad_set: 'RadiativeSet'
    wavelength: np.ndarray
    models: List[AtomicModel]
    trans_wavelengths: Dict[Tuple[Element, int, int], np.ndarray]
    blue_idx: Dict[Tuple[Element, int, int], int]
    red_idx: Dict[Tuple[Element, int, int], int]
    active_trans: Dict[Tuple[Element, int, int], bool]
    active_wavelengths: Dict[Tuple[Element, int, int], np.ndarray]

    def subset_configuration(self, wavelengths) -> 'SpectrumConfiguration':
        """
        Computes a SpectrumConfiguration for a sub-region of the global wavelength array.

        This is typically used for computing a final formal solution on a single
        ray through the atmosphere. In this situation all lines are set to
        contribute throughout the entire grid, to avoid situations where small
        jumps in intensity occur from lines being cut off.

        Parameters
        ----------
        wavelengths : np.ndarray
            The grid on which to produce the new SpectrumConfiguration.

        Returns
        -------
        spectrumConfig : SpectrumConfiguration
            The subset spectrum configuration.
        """
        Nblue = np.searchsorted(self.wavelength, wavelengths[0])
        Nred = min(np.searchsorted(self.wavelength, wavelengths[-1]) + 1, self.wavelength.shape[0])
        Nwavelengths = wavelengths.shape[0]

        active_trans = {k: bool(np.any(v[Nblue:Nred])) for k, v in self.active_wavelengths.items()}
        transGrids = {k: np.copy(wavelengths) for k, active in active_trans.items() if active}
        active_wavelengths = {k: np.ones_like(wavelengths, dtype=bool) for k in transGrids}
        blue_idx = {k: 0 for k in transGrids}
        red_idx = {k: Nwavelengths for k in transGrids}

        def test_atom_active(atom: AtomicModel) -> bool:
            for t in atom.transitions:
                if active_trans[t.trans_id]:
                    return True
            return False

        models = []
        for atom in self.models:
            if test_atom_active(atom):
                models.append(atom)

        return SpectrumConfiguration(
            rad_set=self.rad_set,
            wavelength=wavelengths,
            models=models,
            trans_wavelengths=transGrids,
            blue_idx=blue_idx,
            red_idx=red_idx,
            active_trans=active_trans,
            active_wavelengths=active_wavelengths,
        )

    @property
    def Nprd_trans(self):
        try:
            return self._NprdTrans
        except AttributeError:
            count = 0
            for element in self.rad_set.active_set:
                atom = self.rad_set.atoms[element]
                for l in atom.lines:
                    if l.type == LineType.PRD:
                        count += 1
            self._NprdTrans = count
            return count


@deprecated_names
@dataclass
class AtomicState:
    """
    Container for the state of an atomic model during a simulation.

    This hold both the model, as well as the simulations properties such as
    abundance, populations and radiative rates.

    Attributes
    ----------
    model : AtomicModel
        The python model of the atom.
    abundance : float
        The abundance of the species as a fraction of H abundance.
    n_star : np.ndarray
        The LTE populations of the species.
    n_total : np.ndarray
        The total species population at each point in the atmosphere.
    detailed : bool
        Whether the species has detailed populations.
    pops : np.ndarray, optional
        The NLTE populations for the species, if detailed is True.
    radiative_rates: Dict[(int, int), np.ndarray], optional
        If detailed the radiative rates for the species will be present here,
        stored under (i, j) and (j, i) for each transition.
    """

    model: AtomicModel
    abundance: float
    n_star: np.ndarray
    n_total: np.ndarray
    detailed: bool = False
    pops: Optional[np.ndarray] = None
    radiative_rates: Optional[Dict[Tuple[int, int], np.ndarray]] = None

    def __post_init__(self):
        if self.detailed:
            self.radiative_rates = {}
            ratesShape = self.n_star.shape[1:]
            for t in self.model.transitions:
                self.radiative_rates[(t.i, t.j)] = np.zeros(ratesShape)
                self.radiative_rates[(t.j, t.i)] = np.zeros(ratesShape)

    def __str__(self):
        s = 'AtomicState(%s)' % self.element
        return s

    def __hash__(self):
        # return hash(repr(self))
        raise NotImplementedError

    def dimensioned_view(self, shape):
        """
        Returns a view over the contents of AtomicState reshaped so all data
        has the correct (1/2/3D) dimensionality for the atmospheric model, as
        these are all stored under a flat scheme.

        Parameters
        ----------
        shape : tuple
            The shape to reshape to, this can be obtained from
            Atmosphere.structure.dimensioned_shape

        Returns
        -------
        state : AtomicState
            An instance of self with the arrays reshaped to the appropriate
            dimensionality.
        """
        state = copy(self)
        state.n_star = self.n_star.reshape(-1, *shape)
        state.n_total = self.n_total.reshape(shape)
        if self.pops is not None:
            state.pops = self.pops.reshape(-1, *shape)
            state.radiative_rates = {k: v.reshape(shape) for k, v in self.radiative_rates.items()}
        return state

    def unit_view(self):
        """
        Returns a view over the contents of the AtomicState with the correct
        `astropy.units`.
        """
        state = copy(self)
        m3 = u.m ** (-3)
        state.n_star = self.n_star << m3
        state.n_total = self.n_total << m3
        if self.pops is not None:
            state.pops = self.pops << m3
            state.radiative_rates = {k: v << u.s**-1 for k, v in self.radiative_rates.items()}
        return state

    def dimensioned_unit_view(self, shape):
        """
        Returns a view over the contents of AtomicState reshaped so all data
        has the correct (1/2/3D) dimensionality for the atmospheric model,
        and the correct `astropy.units`.

        Parameters
        ----------
        shape : tuple
            The shape to reshape to, this can be obtained from
            Atmosphere.structure.dimensioned_shape

        Returns
        -------
        state : AtomicState
            An instance of self with the arrays reshaped to the appropriate
            dimensionality.
        """
        state = self.dimensioned_view(shape)
        return state.unit_view()

    def update_n_total(self, atmos: Atmosphere):
        """
        Update n_total assuming either the abundance or nh_tot have changed.
        """
        self.n_total[:] = self.abundance * atmos.nh_tot  # type: ignore

    @property
    def element(self) -> Element:
        """
        The element associated with this model.
        """
        return self.model.element

    @property
    def mass(self) -> float:
        """
        The mass of the element associated with this model.
        """
        return self.element.mass

    @property
    def n(self) -> np.ndarray:
        """
        The NLTE populations, if present, or the LTE populations.
        """
        if self.pops is None:
            return self.n_star
        return self.pops

    @n.setter
    def n(self, val: np.ndarray):
        if val.shape != self.n_star.shape:
            raise ValueError(
                ('Incorrect dimensions for population array, expected %s') % self.n_star.shape
            )

        self.pops = val

    @property
    def name(self) -> str:
        """
        The name of the element associated with this model.
        """
        return self.model.element.name

    def fjk(self, atmos, k):
        # Nstage: int = (self.model.levels[-1].stage - self.model.levels[0].stage) + 1
        Nstage: int = self.model.levels[-1].stage + 1

        fjk = np.zeros(Nstage)
        # TODO(cmo): Proper derivative treatment
        dfjk = np.zeros(Nstage)

        for i, l in enumerate(self.model.levels):
            fjk[l.stage] += self.n[i, k]

        fjk /= self.n_total[k]

        return fjk, dfjk

    def fj(self, atmos):
        Nstage: int = self.model.levels[-1].stage + 1
        Nspace: int = atmos.Nspace

        fj = np.zeros((Nstage, Nspace))
        # TODO(cmo): Proper derivative treatment
        dfj = np.zeros((Nstage, Nspace))

        for i, l in enumerate(self.model.levels):
            fj[l.stage] += self.n[i]

        fj /= self.n_total

        return fj, dfj

    def set_n_to_lte(self):
        """
        Reset the NLTE populations to LTE.
        """
        if self.pops is not None:
            self.pops[:] = self.n_star


class AtomicStateTable:
    """
    Container for AtomicStates.

    The __getitem__ on this class is intended to be smart, and should work
    correctly with ints, strings, or Elements and return the associated
    AtomicState.
    This object is not normally constructed directly by the user, but will
    instead be interacted with as a means of transporting information to and
    from the backend.

    """

    def __init__(self, atoms: List[AtomicState]):
        self.atoms = {a.element: a for a in atoms}

    def __contains__(self, name: Union[int, Tuple[int, int], str, Element]) -> bool:
        try:
            x = PeriodicTable[name]
            return x in self.atoms
        except KeyError:
            return False

    def __len__(self) -> int:
        return len(self.atoms)

    def __getitem__(self, name: Union[int, Tuple[int, int], str, Element]) -> AtomicState:
        x = PeriodicTable[name]
        return self.atoms[x]

    def __iter__(self):
        return iter(sorted(self.atoms.values(), key=element_sort))

    def dimensioned_view(self, shape):
        """
        Returns a view over the contents of AtomicStateTable reshaped so all data
        has the correct (1/2/3D) dimensionality for the atmospheric model, as
        these are all stored under a flat scheme.

        Parameters
        ----------
        shape : tuple
            The shape to reshape to, this can be obtained from
            Atmosphere.structure.dimensioned_shape

        Returns
        -------
        state : AtomicStateTable
            An instance of self with the arrays reshaped to the appropriate
            dimensionality.
        """
        table = copy(self)
        table.atoms = {k: a.dimensioned_view(shape) for k, a in self.atoms.items()}
        return table

    def unit_view(self):
        """
        Returns a view over the contents of the AtomicStateTable with the correct
        `astropy.units`.
        """
        table = copy(self)
        table.atoms = {k: a.unit_view() for k, a in self.atoms.items()}
        return table

    def dimensioned_unit_view(self, shape):
        """
        Returns a view over the contents of AtomicStateTable reshaped so all data
        has the correct (1/2/3D) dimensionality for the atmospheric model,
        and the correct `astropy.units`.

        Parameters
        ----------
        shape : tuple
            The shape to reshape to, this can be obtained from
            Atmosphere.structure.dimensioned_shape

        Returns
        -------
        state : AtomicStateTable
            An instance of self with the arrays reshaped to the appropriate
            dimensionality.
        """
        table = self.dimensioned_view(shape)
        return table.unit_view()


@deprecated_names
@dataclass
class SpeciesStateTable:
    """
    Container for the species populations in the simulation. Similar to
    AtomicStateTable but also holding the molecular populations and the
    atmosphere object.

    The __getitem__ is intended to be smart, returning in order of priority
    on the name match (int, str, Element), H- populations, molecular
    populations, NLTE atomic populations, LTE atomic populations.
    This object is not normally constructed directly by the user, but will
    instead be interacted with as a means of transporting information to and
    from the backend.

    Attributes
    ----------
    atmosphere : Atmosphere
        The atmosphere object.
    abundance : AtomicAbundance
        The abundance of all species present in the atmosphere.
    atomic_pops : AtomicStateTable
        The atomic populations state container.
    molecular_table : MolecularTable
        The molecules present in the simulation.
    molecular_pops : list of np.ndarray
        The populations of each molecule in the molecular_table
    hmin_pops : np.ndarray
        H- ion populations throughout the atmosphere.
    """

    atmosphere: Atmosphere
    abundance: AtomicAbundance
    atomic_pops: AtomicStateTable
    molecular_table: MolecularTable
    molecular_pops: List[np.ndarray]
    hmin_pops: np.ndarray

    def dimensioned_view(self):
        """
        Returns a view over the contents of SpeciesStateTable reshaped so all data
        has the correct (1/2/3D) dimensionality for the atmospheric model, as
        these are all stored under a flat scheme.
        """
        shape = self.atmosphere.structure.dimensioned_shape
        table = copy(self)
        table.atmosphere = self.atmosphere.dimensioned_view()
        table.atomic_pops = self.atomic_pops.dimensioned_view(shape)
        table.molecular_pops = [m.reshape(shape) for m in self.molecular_pops]
        table.hmin_pops = self.hmin_pops.reshape(shape)
        return table

    def unit_view(self):
        """
        Returns a view over the contents of the SpeciesStateTable with the correct
        `astropy.units`.
        """
        table = copy(self)
        table.atmosphere = self.atmosphere.unit_view()
        table.atomic_pops = self.atomic_pops.unit_view()
        table.molecular_pops = [(m << u.m ** (-3)) for m in self.molecular_pops]
        table.hmin_pops = self.hmin_pops << u.m ** (-3)
        return table

    def dimensioned_unit_view(self):
        """
        Returns a view over the contents of SpeciesStateTable reshaped so all data
        has the correct (1/2/3D) dimensionality for the atmospheric model,
        and the correct `astropy.units`.
        """
        table = self.dimensioned_view()
        return table.unit_view()

    def __getitem__(self, name: Union[int, Tuple[int, int], str, Element]) -> np.ndarray:
        if isinstance(name, str) and name == 'H-':
            return self.hmin_pops

        if name in self.molecular_table:
            name = cast(str, name)
            key = self.molecular_table.indices[name.upper()]
            return self.molecular_pops[key]

        if name in self.atomic_pops:
            return self.atomic_pops[name].n

        raise LookupError(f'Element defined by "{name}" not found.')

    def __contains__(self, name: Union[int, Tuple[int, int], str, Element]) -> bool:
        if name == 'H-':
            return True

        if name in self.molecular_table:
            return True

        if name in self.atomic_pops:
            return True

        return False

    def update_lte_atoms_hmin_pops(
        self,
        atmos: Atmosphere,
        conserve_charge=False,
        update_totals=False,
        max_iter=2000,
        quiet=False,
        tol=1e-3,
    ):
        """
        Under the assumption that the atmosphere has changed, update the LTE
        atomic populations and the H- populations.

        Parameters
        ----------
        atmos : Atmosphere
            The atmosphere object.
        conserve_charge : bool
            Whether to conserve_charge and adjust the electron density in
            atmos based on the change in ionisation of the non-detailed
            species (default: False).
        update_totals : bool, optional
            Whether to update the totals of each species from the abundance
            and total hydrogen density (default: False).
        max_iter : int, optional
            The maximum number of iterations to take looking for a stable
            solution (default: 2000).
        quiet : bool, optional
            Whether to print information about the update (default: False)
        tol : float, optional
            The tolerance of relative change at which to consider the
            populations converged (default: 1e-3)
        """
        if update_totals:
            for atom in self.atomic_pops:
                atom.update_n_total(atmos)
        for i in range(max_iter):
            maxDiff = 0.0
            maxName = '--'
            ne = np.zeros_like(atmos.ne)
            diffs = [
                update_lte_pops_inplace(
                    atom.model, atmos.temperature, atmos.ne, atom.n_total, atom.n_star, debye=True
                )[1]
                for atom in self.atomic_pops
            ]

            for j, atom in enumerate(self.atomic_pops):
                if conserve_charge:
                    stages = np.array([l.stage for l in atom.model.levels])
                    if atom.pops is None:
                        ne += np.sum(atom.n_star * stages[:, None], axis=0)
                    else:
                        ne += np.sum(atom.n * stages[:, None], axis=0)

                diff = diffs[j]
                if diff > maxDiff:
                    maxDiff = diff
                    maxName = atom.name
            if conserve_charge:
                ne[ne < 1e6] = 1e6
                atmos.ne[:] = ne
            if maxDiff < tol:
                if not quiet:
                    print('LTE Iterations %d (%s slowest convergence)' % (i + 1, maxName))
                break

        else:
            raise ValueError('No convergence in LTE update')

        self.hmin_pops[:] = hminus_pops(atmos, self.atomic_pops['H'])


@deprecated_names(attrs=('active_set', 'detailed_static_set', 'passive_set'))
class RadiativeSet:
    """
    Used to configure the atomic models present in the simulation and then
    set up the global wavelength grid and initial populations.
    All atoms start passive.

    Parameters
    ----------
    atoms : list of AtomicModel
        The atomic models to be used in the simulation (active, detailed, and
        background).
    abundance : AtomicAbundance, optional
        The abundance to be used for each species.

    Attributes
    ----------
    abundance : AtomicAbundance
        The abundances in use.
    elements : list of Elements
        The elements present in the simulation.
    atoms : Dict[Element, AtomicModel]
        Mapping from Element to associated model.
    passive_set : set of Elements
        Set of atoms (designmated by their Elements) set to passive in the
        simulation.
    detailed_static_set : set of Elements
        Set of atoms (designmated by their Elements) set to "detailed static" in the
        simulation.
    active_set : set of Elements
        Set of atoms (designmated by their Elements) set to active in the
        simulation.
    """

    def __init__(
        self, atoms: List[AtomicModel], abundance: AtomicAbundance = DefaultAtomicAbundance
    ):
        self.abundance = abundance
        self.elements = [a.element for a in atoms]
        self.atoms = {k: v for k, v in zip(self.elements, atoms)}
        self.passive_set = set(self.elements)
        self.detailed_static_set: Set[Element] = set()
        self.active_set: Set[Element] = set()

        if len(self.passive_set) < len(self.elements):
            duplicates = sorted({e.name for e in self.elements if self.elements.count(e) > 1})
            raise ValueError('Multiple entries for an atom: %s' % duplicates)

    def __contains__(self, x: Union[int, Tuple[int, int], str, Element]) -> bool:
        return PeriodicTable[x] in self.elements

    def is_active(self, name: Union[int, Tuple[int, int], str, Element]) -> bool:
        """
        Check if an atom (designated by int, (int, int), str, or Element) is
        active.
        """
        x = PeriodicTable[name]
        return x in self.active_set

    def is_passive(self, name: Union[int, Tuple[int, int], str, Element]) -> bool:
        """
        Check if an atom (designated by int, (int, int), str, or Element) is
        passive.
        """
        x = PeriodicTable[name]
        return x in self.passive_set

    def is_detailed(self, name: Union[int, Tuple[int, int], str, Element]) -> bool:
        """
        Check if an atom (designated by int, (int, int), str, or Element) is
        passive.
        """
        x = PeriodicTable[name]
        return x in self.detailed_static_set

    @property
    def active_atoms(self) -> List[AtomicModel]:
        """
        List of AtomicModels set to active.
        """
        active_atoms: List[AtomicModel] = [self.atoms[e] for e in self.active_set]
        active_atoms = sorted(active_atoms, key=element_sort)
        return active_atoms

    @property
    def detailed_atoms(self) -> List[AtomicModel]:
        """
        List of AtomicModels set to detailed static.
        """
        detailed_atoms: List[AtomicModel] = [self.atoms[e] for e in self.detailed_static_set]
        detailed_atoms = sorted(detailed_atoms, key=element_sort)
        return detailed_atoms

    @property
    def passive_atoms(self) -> List[AtomicModel]:
        """
        List of AtomicModels set to passive.
        """
        passive_atoms: List[AtomicModel] = [self.atoms[e] for e in self.passive_set]
        passive_atoms = sorted(passive_atoms, key=element_sort)
        return passive_atoms

    def __getitem__(self, name: Union[int, Tuple[int, int], str, Element]) -> AtomicModel:
        x = PeriodicTable[name]
        return self.atoms[x]

    def __iter__(self):
        return iter(self.atoms.values())

    def set_active(self, *args: str):
        """
        Set one (or multiple) atoms active.
        """
        names = set(args)
        xs = [PeriodicTable[name] for name in names]
        for x in xs:
            self.active_set.add(x)
            self.detailed_static_set.discard(x)
            self.passive_set.discard(x)

    def set_detailed_static(self, *args: str):
        """
        Set one (or multiple) atoms to detailed static
        """
        names = set(args)
        xs = [PeriodicTable[name] for name in names]
        for x in xs:
            self.detailed_static_set.add(x)
            self.active_set.discard(x)
            self.passive_set.discard(x)

    def set_passive(self, *args: str):
        """
        Set one (or multiple) atoms passive.
        """
        names = set(args)
        xs = [PeriodicTable[name] for name in names]
        for x in xs:
            self.passive_set.add(x)
            self.active_set.discard(x)
            self.detailed_static_set.discard(x)

    def iterate_lte_ne_eq_pops(
        self,
        atmos: Atmosphere,
        mols: Optional[MolecularTable] = None,
        nlte_starting_pops: Optional[Dict[Element, np.ndarray]] = None,
        direct: bool = True,
        quiet: bool = True,
    ) -> SpeciesStateTable:
        """
        Compute the starting populations for the simulation with all NLTE
        atoms in LTE or otherwise using the provided populations.
        Additionally computes a self-consistent LTE electron density.

        Parameters
        ----------
        atmos : Atmosphere
            The atmosphere for which to compute the populations.
        mols : MolecularTable, optional
            Molecules to be included in the populations (default: None)
        nlte_starting_pops : Dict[Element, np.ndarray], optional
            Starting population override for any active or detailed static
            species.
        direct : bool
            Whether to use the direct electron density solver (essentially,
            damped Lambda iteration for the fixpoint). With the new damping this
            appears to converge very quickly with little need for the
            Newton-Krylov selected if this is set to False. Default: True
        quiet : bool, optional
            Whether to print convergence info (default: True, i.e. don't print).

        Returns
        -------
        eq_pops : SpeciesStatTable
            The configured initial populations.
        """
        if mols is None:
            mols = MolecularTable([])

        if nlte_starting_pops is None:
            nlte_starting_pops = {}
        else:
            for e in nlte_starting_pops:
                if (e not in self.active_set) and (e not in self.detailed_static_set):
                    raise ValueError(
                        (
                            'Provided NLTE Populations for %s assumed LTE. '
                            'Ensure these are indexed by `Element` '
                            'rather than str.'
                        )
                        % e
                    )

        if direct:
            max_iter = 3000
            prevNe = np.copy(atmos.ne)
            ne = np.copy(atmos.ne)
            atoms = sorted(self.atoms.values(), key=element_sort)
            for it in range(max_iter):
                atomic_pops = []
                prevNe[:] = ne
                ne.fill(0.0)
                for a in atoms:
                    abund = self.abundance[a.element]
                    n_total = abund * atmos.nh_tot
                    n_star = lte_pops(a, atmos.temperature, atmos.ne, n_total, debye=True)
                    atomic_pops.append(
                        AtomicState(model=a, abundance=abund, n_star=n_star, n_total=n_total)
                    )

                    # NOTE(cmo): Take into account NLTE pops if provided
                    if a.element in nlte_starting_pops:
                        if nlte_starting_pops[a.element].shape != n_star.shape:
                            raise ValueError(
                                ('Starting populations provided for %s do not match model.')
                                % a.element
                            )
                        n_star = nlte_starting_pops[a.element]

                    stages = np.array([l.stage for l in a.levels])
                    ne += np.sum(n_star * stages[:, None], axis=0)
                # NOTE(cmo): Damp correction: dramatically improves convergence.
                atmos.ne[:] = 0.55 * ne + 0.45 * prevNe

                max_err = np.nanmax(np.abs(1.0 - prevNe / atmos.ne))
                if max_err < 1e-5:
                    if not quiet:
                        print('Iterate LTE: %d iterations' % it)
                    break
            else:
                raise ValueError('LTE ne failed to converge')
        else:
            neRatio = np.copy(atmos.ne) / atmos.nh_tot
            iterator = LteNeIterator(
                self.atoms.values(),
                atmos.temperature,
                atmos.nh_tot,
                self.abundance,
                nlte_starting_pops,
            )
            neRatio += iterator(neRatio)
            newNeRatio = newton_krylov(iterator, neRatio)
            atmos.ne[:] = newNeRatio * atmos.nh_tot

            atomic_pops = iterator.atomic_pops

        detailedAtomicPops = []
        for pop in atomic_pops:
            ele = pop.model.element
            if ele in self.passive_set:
                if ele in nlte_starting_pops:
                    # NOTE(cmo): I don't believe this is possible; it would need
                    # to be detailed_static as per the contract on passive atoms
                    # being "true" LTE.  Leaving for now for safety.
                    pop.n = np.copy(nlte_starting_pops[ele])
                detailedAtomicPops.append(pop)
            else:
                nltePops = (
                    np.copy(nlte_starting_pops[ele])
                    if ele in nlte_starting_pops
                    else np.copy(pop.n_star)
                )
                detailedAtomicPops.append(
                    AtomicState(
                        model=pop.model,
                        abundance=self.abundance[ele],
                        n_star=pop.n_star,
                        n_total=pop.n_total,
                        detailed=True,
                        pops=nltePops,
                    )
                )

        table = AtomicStateTable(detailedAtomicPops)
        eq_pops = chemical_equilibrium_fixed_ne(atmos, mols, table, self.abundance, quiet=quiet)
        # NOTE(cmo): This is technically not quite correct, because we adjust
        # n_total and the atomic populations to account for the atoms bound up
        # in molecules, but not n_e, this is unlikely to make much difference
        # in reality, other than in very cool atmospheres with a lot of
        # molecules (even then it should be pretty tiny)
        return eq_pops

    def compute_eq_pops(
        self,
        atmos: Atmosphere,
        mols: Optional[MolecularTable] = None,
        nlte_starting_pops: Optional[Dict[Element, np.ndarray]] = None,
    ):
        """
        Compute the starting populations for the simulation with all NLTE
        atoms in LTE or otherwise using the provided populations.

        Parameters
        ----------
        atmos : Atmosphere
            The atmosphere for which to compute the populations.
        mols : MolecularTable, optional
            Molecules to be included in the populations (default: None)
        nlte_starting_pops : Dict[Element, np.ndarray], optional
            Starting population override for any active or detailed static
            species.

        Returns
        -------
        eq_pops : SpeciesStatTable
            The configured initial populations.
        """
        if mols is None:
            mols = MolecularTable([])

        if nlte_starting_pops is None:
            nlte_starting_pops = {}
        else:
            for e in nlte_starting_pops:
                if (e not in self.active_set) and (e not in self.detailed_static_set):
                    raise ValueError(
                        (
                            'Provided NLTE Populations for %s assumed LTE. '
                            'Ensure these are indexed by `Element` '
                            'rather than str.'
                        )
                        % e
                    )

        atomic_pops = []
        atoms = sorted(self.atoms.values(), key=element_sort)
        for a in atoms:
            n_total = self.abundance[a.element] * atmos.nh_tot
            n_star = lte_pops(a, atmos.temperature, atmos.ne, n_total, debye=True)

            ele = a.element
            if ele in self.passive_set:
                n = None
                atomic_pops.append(
                    AtomicState(
                        model=a,
                        abundance=self.abundance[ele],
                        n_star=n_star,
                        n_total=n_total,
                        pops=n,
                    )
                )
            else:
                nltePops = (
                    np.copy(nlte_starting_pops[ele])
                    if ele in nlte_starting_pops
                    else np.copy(n_star)
                )
                atomic_pops.append(
                    AtomicState(
                        model=a,
                        abundance=self.abundance[ele],
                        n_star=n_star,
                        n_total=n_total,
                        detailed=True,
                        pops=nltePops,
                    )
                )

        table = AtomicStateTable(atomic_pops)
        eq_pops = chemical_equilibrium_fixed_ne(atmos, mols, table, self.abundance)
        # NOTE(cmo): This is technically not quite correct, because we adjust
        # n_total and the atomic populations to account for the atoms bound up
        # in molecules, but not n_e, this is unlikely to make much difference
        # in reality, other than in very cool atmospheres with a lot of
        # molecules (even then it should be pretty tiny)
        return eq_pops

    def compute_wavelength_grid(
        self, extra_wavelengths: Optional[np.ndarray] = None, lambda_reference=500.0
    ) -> SpectrumConfiguration:
        """
        Compute the global wavelength grid from the current configuration of
        the RadiativeSet.

        Parameters
        ----------
        extra_wavelengths : np.ndarray, optional
            Extra wavelengths to add to the global array [nm].
        lambda_reference : float, optional
            If a difference reference wavelength is to be used then it should
            be specified here to ensure it is in the global array.

        Returns
        -------
        spect : SpectrumConfiguration
            The configured wavelength grids needed to set up the backend.
        """
        if len(self.active_set) == 0 and len(self.detailed_static_set) == 0:
            raise ValueError(
                'Need at least one atom active or in detailed calculation with static populations.'
            )
        extraGrids = []
        if extra_wavelengths is not None:
            extraGrids.append(extra_wavelengths)
        extraGrids.append(np.array([lambda_reference]))

        models: List[AtomicModel] = []
        ids: List[Tuple[Element, int, int]] = []
        grids = []

        for ele in self.active_set | self.detailed_static_set:
            atom = self.atoms[ele]
            models.append(atom)
            for trans in atom.transitions:
                grids.append(trans.wavelength())
                ids.append(trans.trans_id)

        grid = np.concatenate(grids + extraGrids)
        grid = np.sort(grid)
        grid = np.unique(grid)
        # grid = np.unique(np.floor(1e10*grid)) / 1e10
        blue_idx = {}
        red_idx = {}

        for i, g in enumerate(grids):
            ident = ids[i]
            blue_idx[ident] = np.searchsorted(grid, g[0])
            red_idx[ident] = np.searchsorted(grid, g[-1]) + 1

        transGrids: Dict[Tuple[Element, int, int], np.ndarray] = {}
        for ident in ids:
            transGrids[ident] = np.copy(grid[blue_idx[ident] : red_idx[ident]])

        active_wavelengths = {k: ((grid >= v[0]) & (grid <= v[-1])) for k, v in transGrids.items()}
        active_trans = {k: True for k in transGrids}

        return SpectrumConfiguration(
            rad_set=self,
            wavelength=grid,
            models=models,
            trans_wavelengths=transGrids,
            blue_idx=blue_idx,
            red_idx=red_idx,
            active_trans=active_trans,
            active_wavelengths=active_wavelengths,
        )


@accepts_old_kwargs
def hminus_pops(atmos: Atmosphere, h_pops: AtomicState) -> np.ndarray:
    """
    Compute the H- ion populations for a given atmosphere, in Saha
    equilibrium with the neutral hydrogen population.

    Parameters
    ----------
    atmos : Atmosphere
        The atmosphere object.

    h_pops : AtomicState
        The hydrogen populations state associated with atmos.

    Returns
    -------
    hmin_pops : np.ndarray
        The H- populations.
    """
    CI = (Const.HPlanck / (2.0 * np.pi * Const.MElectron)) * (Const.HPlanck / Const.KBoltzmann)
    PhiHmin = (
        0.25
        * (CI / atmos.temperature) ** 1.5
        * np.exp(Const.E_ION_HMIN / (Const.KBoltzmann * atmos.temperature))
    )
    neutral = np.array([l.stage == 0 for l in h_pops.model.levels])
    hmin_pops = atmos.ne * np.sum(h_pops.n[neutral], axis=0) * PhiHmin

    return hmin_pops


@accepts_old_kwargs
def chemical_equilibrium_fixed_ne(
    atmos: Atmosphere,
    molecules: MolecularTable,
    atomic_pops: AtomicStateTable,
    abundance: AtomicAbundance,
    quiet: bool = False,
) -> SpeciesStateTable:
    """
    Compute the molecular populations from the current atmospheric model and
    atomic populations.

    This method assumes that the number of electrons bound in molecules is
    insignificant, and neglects this.
    Intended for internal use.

    Parameters
    ----------
    atmos : Atmosphere
        The model atmosphere of the simulation.
    molecules : MolecularTable
        The molecules to consider.
    atomic_pops : AtomicStateTable
        The atomic populations.
    abundance : AtomicAbundance
        The abundance of each species in the simulation.
    quiet : bool, optional
        Whether to print convergence info (default: True).


    Returns
    -------
    state : SpeciesState
        The combined state object of atomic and molecular populations.
    """
    nucleiSet: Set[Element] = set()
    for mol in molecules:
        nucleiSet |= set(mol.elements)
    nuclei: List[Element] = list(nucleiSet)
    nuclei = sorted(nuclei)

    if len(nuclei) == 0:
        hmin_pops = hminus_pops(atmos, atomic_pops['H'])
        result = SpeciesStateTable(atmos, abundance, atomic_pops, molecules, [], hmin_pops)
        return result

    if nuclei[0] != PeriodicTable[1]:
        raise ValueError('H not list of nuclei -- check H2 molecule')
    # print([n.name for n in nuclei])

    nuclIndex = [[nuclei.index(ele) for ele in mol.elements] for mol in molecules]

    # Replace basic elements with full Models if present
    kuruczTable = KuruczPfTable(atomic_abundance=abundance)
    nucData: Dict[Element, Union[KuruczPf, AtomicState]] = {}
    for nuc in nuclei:
        if nuc in atomic_pops:
            nucData[nuc] = atomic_pops[nuc]
        else:
            nucData[nuc] = kuruczTable[nuc]

    Nnuclei = len(nuclei)

    Neqn = Nnuclei + len(molecules)
    f = np.zeros(Neqn)
    n = np.zeros(Neqn)
    df = np.zeros((Neqn, Neqn))
    a = np.zeros(Neqn)

    # Equilibrium constant per molecule
    Phi = np.zeros(len(molecules))
    # Neutral fraction
    fn0 = np.zeros(Nnuclei)

    CI = (Const.HPlanck / (2.0 * np.pi * Const.MElectron)) * (Const.HPlanck / Const.KBoltzmann)
    Nspace = atmos.Nspace
    hmin_pops = np.zeros(Nspace)
    molPops = [np.zeros(Nspace) for mol in molecules]
    max_iter = 0
    for k in range(Nspace):
        for i, nuc in enumerate(nuclei):
            nucleus = nucData[nuc]
            a[i] = nucleus.abundance * atmos.nh_tot[k]
            fjk, dfjk = nucleus.fjk(atmos, k)
            fn0[i] = fjk[0]

        PhiHmin = (
            0.25
            * (CI / atmos.temperature[k]) ** 1.5
            * np.exp(Const.E_ION_HMIN / (Const.KBoltzmann * atmos.temperature[k]))
        )
        fHmin = atmos.ne[k] * fn0[0] * PhiHmin

        # Eq constant for each molecule at this location
        for i, mol in enumerate(molecules):
            Phi[i] = mol.equilibrium_constant(atmos.temperature[k])

        # Setup initial solution. Everything dissociated
        # n[:Nnuclei] = a[:Nnuclei]
        # n[Nnuclei:] = 0.0
        n[:] = a[:]
        # print('a', a)

        nIter = 1
        max_iter = 50
        IterLimit = 1e-3
        prevN = n.copy()
        while nIter < max_iter:
            # print(k, ',', nIter)
            # Save previous solution
            prevN[:] = n[:]

            # Set up iteration
            f[:] = n - a
            df[:, :] = 0.0
            np.fill_diagonal(df, 1.0)

            # Add nHmin to number conservation for H
            f[0] += fHmin * n[0]
            df[0, 0] += fHmin

            # Fill population vector f and derivative matrix df
            for i, mol in enumerate(molecules):
                saha = Phi[i]
                for j, ele in enumerate(mol.elements):
                    nu = nuclIndex[i][j]
                    saha *= (fn0[nu] * n[nu]) ** mol.element_count[j]
                    # Contribution to conservation for each nucleus in this molecule
                    f[nu] += mol.element_count[j] * n[Nnuclei + i]

                saha /= atmos.ne[k] ** mol.charge
                f[Nnuclei + i] -= saha
                # if Nnuclei + i == f.shape[0]-1:
                #     print(i)
                #     print(saha)

                # Compute derivative matrix
                for j, ele in enumerate(mol.elements):
                    nu = nuclIndex[i][j]
                    df[nu, Nnuclei + i] += mol.element_count[j]
                    df[Nnuclei + i, nu] = -saha * (mol.element_count[j] / n[nu])

            correction = solve(df, f)
            n -= correction

            dnMax = np.nanmax(np.abs(1.0 - prevN / n))
            if dnMax <= IterLimit:
                max_iter = max(max_iter, nIter)
                break

            nIter += 1
        if dnMax > IterLimit:
            raise ValueError(
                ('ChemEq iteration not converged: T: %e [K], density %e [m^-3], dnmax %e')
                % (atmos.temperature[k], atmos.nh_tot[k], dnMax)
            )

        for i, ele in enumerate(nuclei):
            if ele in atomic_pops:
                atomPop = atomic_pops[ele]
                fraction = n[i] / atomPop.n_total[k]
                atomPop.n_star[:, k] *= fraction
                atomPop.n_total[k] *= fraction
                if atomPop.pops is not None:
                    atomPop.pops[:, k] *= fraction

        hmin_pops[k] = fHmin * n[0]

        for i, pop in enumerate(molPops):
            pop[k] = n[Nnuclei + i]

    result = SpeciesStateTable(atmos, abundance, atomic_pops, molecules, molPops, hmin_pops)
    if not quiet:
        print('chem_eq: maximum number of iterations taken: %d' % max_iter)
    return result
