from dataclasses import dataclass, field
from .atomic_table import PeriodicTable, Element, Isotope, AtomicAbundance, DefaultAtomicAbundance, KuruczPf, KuruczPfTable
from .atomic_model import AtomicTransition, AtomicLine, AtomicModel, AtomicContinuum, element_sort
from .atmosphere import Atmosphere
from .molecule import Molecule, MolecularTable
import lightweaver.constants as Const
from typing import List, Sequence, Set, Optional, Any, Union, Dict, Tuple, Iterable, cast
from copy import copy, deepcopy
from collections import OrderedDict
import numpy as np
from scipy.linalg import solve
from scipy.optimize import newton_krylov
from numba import njit

@njit
def lte_pops_impl(temperature, ne, nTotal, stages, energies,
                  gs, nStar=None, debye=True, computeDiff=False):
    Nlevel = stages.shape[0]
    Nspace = ne.shape[0]
    c1 = (Const.HPlanck / (2.0 * np.pi * Const.MElectron)) * (Const.HPlanck / Const.KBoltzmann)

    c2 = 0.0
    if debye:
        c2 = np.sqrt(8.0 * np.pi / Const.KBoltzmann) * (Const.QElectron**2 / (4.0 * np.pi * Const.Epsilon0))**1.5
        nDebye = np.zeros(Nlevel)
        for i in range(1, Nlevel):
            stage = stages[i]
            Z = stage
            for m in range(1, stage - stages[0] + 1):
                nDebye[i] += Z
                Z += 1

    if nStar is None:
        nStar = np.empty((Nlevel, Nspace))

    if computeDiff:
        prev = np.empty(Nlevel)
    # NOTE(cmo): Will remain 0 and be returned as second return value if
    # computeDiff is not set to True
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
        cNe_T = 0.5 * ne[k] * (c1 / temperature[k])**1.5
        total = 1.0
        if computeDiff:
            for i in range(Nlevel):
                prev[i] = nStar[i, k]
        for i in range(1, Nlevel):
            dE_kT = (dE[i] - nDebye[i] * dEion) / (Const.KBoltzmann * temperature[k])
            neFactor = cNe_T**dZ[i]

            nst = gi0[i] * np.exp(-dE_kT)
            nStar[i, k] = nst
            nStar[i, k] /= neFactor
            total += nStar[i, k]
        nStar[0, k] = nTotal[k] / total

        for i in range(1, Nlevel):
            nStar[i, k] *= nStar[0, k]

        if computeDiff:
            for i in range(Nlevel):
                maxDiff = max((nStar[i, k] - prev[i]) / nStar[i, k], maxDiff)

    return nStar, maxDiff

def lte_pops(atomicModel: AtomicModel, temperature: np.ndarray,
             ne: np.ndarray, nTotal: np.ndarray, nStar=None, debye: bool=True) -> np.ndarray:
    '''
    Compute the LTE populations for a given atomic model under given
    thermodynamic conditions.

    Parameters
    ----------
    atomicModel : AtomicModel
        The atomic model to consider.
    temperature : np.ndarray
        The temperature structure in the atmosphere.
    ne : np.ndarray
        The electron density in the atmosphere.
    nTotal : np.ndarray
        The total population of the species at each point in the atmosphere.
    nStar : np.ndarray, optional
        An optional array to store the result in.
    debye : bool, optional
        Whether to consider Debye shielding (default: True).1

    Returns
    -------
    ltePops : np.ndarray
        The ltePops for the species.
    '''
    stages = np.array([l.stage for l in atomicModel.levels])
    energies = np.array([l.E_SI for l in atomicModel.levels])
    gs = np.array([l.g for l in atomicModel.levels])
    return lte_pops_impl(temperature, ne, nTotal, stages, energies, gs, nStar=nStar, debye=debye)[0]

def update_lte_pops_inplace(atomicModel: AtomicModel, temperature: np.ndarray,
                            ne: np.ndarray, nTotal: np.ndarray,
                            nStar: np.ndarray, debye: bool=True) -> Tuple[np.ndarray, float]:
    stages = np.array([l.stage for l in atomicModel.levels])
    energies = np.array([l.E_SI for l in atomicModel.levels])
    gs = np.array([l.g for l in atomicModel.levels])
    return lte_pops_impl(temperature, ne, nTotal, stages, energies, gs,
                         debye=debye, nStar=nStar, computeDiff=True)

class LteNeIterator:
    def __init__(self, atoms: Iterable[AtomicModel], temperature: np.ndarray,
                 nHTot: np.ndarray, abundance: AtomicAbundance,
                 nlteStartingPops: Dict[Element, np.ndarray]):
        sortedAtoms = sorted(atoms, key=element_sort)
        self.nTotal = [abundance[a.element] * nHTot
                       for a in sortedAtoms]
        self.stages = [np.array([l.stage for l in a.levels])
                       for a in sortedAtoms]
        self.temperature = temperature
        self.nHTot = nHTot
        self.sortedAtoms = sortedAtoms
        self.abundances = [abundance[a.element] for a in sortedAtoms]
        self.nlteStartingPops = nlteStartingPops

    def __call__(self, prevNeRatio: np.ndarray) -> np.ndarray:
        atomicPops = []
        ne = np.zeros_like(prevNeRatio)
        prevNe = prevNeRatio * self.nHTot

        for i, a in enumerate(self.sortedAtoms):
            nStar = lte_pops(a, self.temperature, prevNe,
                             self.nTotal[i], debye=True)
            atomicPops.append(AtomicState(model=a, abundance=self.abundances[i], nStar=nStar, nTotal=self.nTotal[i]))
            # NOTE(cmo): Take into account NLTE pops if provided
            if a.element in self.nlteStartingPops:
                if self.nlteStartingPops[a.element].shape != nStar.shape:
                    raise ValueError('Starting populations provided for %s do not match model.' % a.element)
                nStar = self.nlteStartingPops[a.element]

            ne += np.sum(nStar * self.stages[i][:, None], axis=0)

        self.atomicPops = atomicPops
        diff = (ne - prevNe) / self.nHTot
        return diff


@dataclass
class SpectrumConfiguration:
    '''
    Container for the configuration of common wavelength grid and species
    active at each wavelength.

    Attributes
    ----------
    radSet : RadiativeSet
        The set of atoms involved in the creation of this simulation.
    wavelength : np.ndarray
        The common wavelength array used for this simulation.
    models : list of AtomicModel
        The models for the active and detailed static atoms present in this
        simulation.
    transWavelengths : Dict[(Element, i, j), np.ndarray]
        The local wavelength grid for each transition stored in a dictionary
        by transition ID.
    blueIdx : Dict[(Element, i, j), int]
        The index at which each local grid starts in the global wavelength
        array.
    activeTrans : Dict[(Element, i, j), bool]
        Whether this transition is ever active (contributing in either an
        active or detailed static sense) over the range of wavelength.
    activeWavelengths : Dict[(Element, i, j), np.ndarray]
        A mask of the wavelengths at which this transition is active.
    '''
    radSet: 'RadiativeSet'
    wavelength: np.ndarray
    models: List[AtomicModel]
    transWavelengths: Dict[Tuple[Element, int, int], np.ndarray]
    blueIdx: Dict[Tuple[Element, int, int], int]
    activeTrans: Dict[Tuple[Element, int, int], bool]
    activeWavelengths: Dict[Tuple[Element, int, int], np.ndarray]

    def subset_configuration(self, wavelengths) -> 'SpectrumConfiguration':
        '''
        Computes a SpectrumConfiguration for a sub-region of the global wavelength array.

        This is typically used for computing a final formal solution on a single ray through the atmosphere. In this situation all lines are set to contribute throughout the entire grid, to avoid situations where small jumps in intensity occur from lines being cut off.

        Parameters
        ----------
        wavelengths : np.ndarray
            The grid on which to produce the new SpectrumConfiguration.

        Returns
        -------
        spectrumConfig : SpectrumConfiguration
            The subset spectrum configuration.
        '''
        Nblue = np.searchsorted(self.wavelength, wavelengths[0])
        Nred = min(np.searchsorted(self.wavelength, wavelengths[-1])+1, self.wavelength.shape[0])

        activeTrans = {k: np.any(v[Nblue:Nred]) for k, v in self.activeWavelengths.items()}
        transGrids = {k: np.copy(wavelengths) for k, active in activeTrans.items() if active}
        activeWavelengths = {k: np.ones_like(wavelengths, dtype=np.bool8) for k in transGrids}
        blueIdx = {k: 0 for k in transGrids}

        def test_atom_active(atom: AtomicModel) -> bool:
            for t in atom.transitions:
                if activeTrans[t.transId]:
                    return True
            return False

        models = []
        for atom in self.models:
            if test_atom_active(atom):
                models.append(atom)

        return SpectrumConfiguration(radSet=self.radSet, wavelength=wavelengths, models=models, transWavelengths=transGrids,
                                     blueIdx=blueIdx, activeTrans=activeTrans, activeWavelengths=activeWavelengths)


@dataclass
class AtomicState:
    '''
    Container for the state of an atomic model during a simulation.

    This hold both the model, as well as the simulations properties such as abundance, populations and radiative rates.

    Attributes
    ----------
    model : AtomicModel
        The python model of the atom.
    abundance : float
        The abundance of the species as a fraction of H abundance.
    nStar : np.ndarray
        The LTE populations of the species.
    nTotal : np.ndarray
        The total species population at each point in the atmosphere.
    detailed : bool
        Whether the species has detailed populations.
    pops : np.ndarray, optional
        The NLTE populations for the species, if detailed is True.
    radiativeRates: Dict[(int, int), np.ndarray], optional
        If detailed the radiative rates for the species will be present here,
        stored under (i, j) and (j, i) for each transition.
    '''
    model: AtomicModel
    abundance: float
    nStar: np.ndarray
    nTotal: np.ndarray
    detailed: bool = False
    pops: Optional[np.ndarray] = None
    radiativeRates: Optional[Dict[Tuple[int, int], np.ndarray]] = None

    def __post_init__(self):
        if self.detailed:
            self.radiativeRates = {}
            ratesShape = self.nStar.shape[1:]
            for t in self.model.transitions:
                self.radiativeRates[(t.i, t.j)] = np.zeros(ratesShape)
                self.radiativeRates[(t.j, t.i)] = np.zeros(ratesShape)

    def __str__(self):
        s = 'AtomicState(%s)' % self.element
        return s

    def __hash__(self):
        # return hash(repr(self))
        raise NotImplementedError

    def dimensioned_view(self, shape):
        '''
        Returns a view over the contents of AtomicState reshaped so all data
        has the correct (1/2/3D) dimensionality for the atmospheric model, as
        these are all stored under a flat scheme.

        Parameters
        ----------
        shape : tuple
            The shape to reshape to, this can be obtained from Atmosphere.structure.dimensioned_shape

        Returns
        -------
        state : AtomicState
            An instance of self with the arrays reshaped to the appropriate
            dimensionality.
        '''
        state = copy(self)
        state.nStar = self.nStar.reshape(-1, *shape)
        state.nTotal = self.nTotal.reshape(shape)
        if self.pops is not None:
            state.pops = self.pops.reshape(-1, *shape)
            state.radiativeRates = {k: v.reshape(shape) for k, v in self.radiativeRates.items()}
        return state

    def unit_view(self):
        '''
        Returns a view over the contents of the AtomicState with the correct
        `astropy.units`.
        '''
        state = copy(self)
        m3 = u.m**(-3)
        state.nStar = self.nStar << m3
        state.nTotal = self.nTotal << m3
        if self.pops is not None:
            state.pops = self.pops << m3
            state.radiativeRates = {k: v << u.s**-1 for k, v in self.radiativeRates.items()}
        return state

    def dimensioned_unit_view(shape):
        '''
        Returns a view over the contents of AtomicState reshaped so all data
        has the correct (1/2/3D) dimensionality for the atmospheric model,
        and the correct `astropy.units`.

        Parameters
        ----------
        shape : tuple
            The shape to reshape to, this can be obtained from Atmosphere.structure.dimensioned_shape

        Returns
        -------
        state : AtomicState
            An instance of self with the arrays reshaped to the appropriate
            dimensionality.
        '''
        state = self.dimensioned_view(shape)
        return state.unit_view()

    def update_nTotal(self, atmos: Atmosphere):
        '''
        Update nTotal assuming either the abundance or nHTot have changed.
        '''
        self.nTotal[:] = self.abundance * atmos.nHTot # type: ignore

    @property
    def element(self) -> Element:
        '''
        The element associated with this model.
        '''
        return self.model.element

    @property
    def mass(self) -> float:
        '''
        The mass of the element associated with this model.
        '''
        return self.element.mass

    @property
    def n(self) -> np.ndarray:
        '''
        The NLTE populations, if present, or the LTE populations.
        '''
        if self.pops is None:
            return self.nStar
        return self.pops

    @n.setter
    def n(self, val: np.ndarray):
        if val.shape != self.nStar.shape:
            raise ValueError('Incorrect dimensions for population array, expected %s' % self.nStar.shape)

        self.pops = val

    @property
    def name(self) -> str:
        '''
        The name of the element associated with this model.
        '''
        return self.model.element.name

    def fjk(self, atmos, k):
        # Nstage: int = (self.model.levels[-1].stage - self.model.levels[0].stage) + 1
        Nstage: int = self.model.levels[-1].stage + 1

        fjk = np.zeros(Nstage)
        # TODO(cmo): Proper derivative treatment
        dfjk = np.zeros(Nstage)

        for i, l in enumerate(self.model.levels):
            fjk[l.stage] += self.n[i, k]

        fjk /= self.nTotal[k]

        return fjk, dfjk

    def fj(self, atmos):
        Nstage: int = self.model.levels[-1].stage + 1
        Nspace: int = atmos.Nspace

        fj = np.zeros((Nstage, Nspace))
        # TODO(cmo): Proper derivative treatment
        dfj = np.zeros((Nstage, Nspace))

        for i, l in enumerate(self.model.levels):
            fj[l.stage] += self.n[i]

        fj /= self.nTotal

        return fj, dfj

    def set_n_to_lte(self):
        '''
        Reset the NLTE populations to LTE.
        '''
        if self.pops is not None:
            self.pops[:] = self.nStar


class AtomicStateTable:
    '''
    Container for AtomicStates.

    The __getitem__ on this class is intended to be smart, and should work
    correctly with ints, strings, or Elements and return the associated
    AtomicState.
    This object is not normally constructed directly by the user, but will
    instead be interacted with as a means of transporting information to and
    from the backend.

    '''
    def __init__(self, atoms: List[AtomicState]):
        self.atoms = {a.element: a for a in atoms}

    def __contains__(self, name: Union[int, Tuple[int, int], str, Element]) -> bool:
        try:
            x = PeriodicTable[name]
            return x in self.atoms
        except:
            return False

    def __len__(self) -> int:
        return len(self.atoms)

    def __getitem__(self, name: Union[int, Tuple[int, int], str, Element]) -> AtomicState:
        x = PeriodicTable[name]
        return self.atoms[x]

    def __iter__(self):
        return iter(sorted(self.atoms.values(), key=element_sort))

    def dimensioned_view(self, shape):
        '''
        Returns a view over the contents of AtomicStateTable reshaped so all data
        has the correct (1/2/3D) dimensionality for the atmospheric model, as
        these are all stored under a flat scheme.

        Parameters
        ----------
        shape : tuple
            The shape to reshape to, this can be obtained from Atmosphere.structure.dimensioned_shape

        Returns
        -------
        state : AtomicStateTable
            An instance of self with the arrays reshaped to the appropriate
            dimensionality.
        '''
        table = copy(self)
        table.atoms = {a.element: a.dimensioned_view(shape) for a in self.atoms}
        return table

    def unit_view(self):
        '''
        Returns a view over the contents of the AtomicStateTable with the correct
        `astropy.units`.
        '''
        table = copy(self)
        table.atoms = {a.element: a.unit_view() for a in self.atoms}
        return table

    def dimensioned_unit_view(self, shape):
        '''
        Returns a view over the contents of AtomicStateTable reshaped so all data
        has the correct (1/2/3D) dimensionality for the atmospheric model,
        and the correct `astropy.units`.

        Parameters
        ----------
        shape : tuple
            The shape to reshape to, this can be obtained from Atmosphere.structure.dimensioned_shape

        Returns
        -------
        state : AtomicStateTable
            An instance of self with the arrays reshaped to the appropriate
            dimensionality.
        '''
        table = self.dimensioned_view(shape)
        return table.unit_view()


@dataclass
class SpeciesStateTable:
    '''
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
    atomicPops : AtomicStateTable
        The atomic populations state container.
    molecularTable : MolecularTable
        The molecules present in the simulation.
    molecularPops : list of np.ndarray
        The populations of each molecule in the molecularTable
    HminPops : np.ndarray
        H- ion populations throughout the atmosphere.
    '''
    atmosphere: Atmosphere
    abundance: AtomicAbundance
    atomicPops: AtomicStateTable
    molecularTable: MolecularTable
    molecularPops: List[np.ndarray]
    HminPops: np.ndarray

    def dimensioned_view(self):
        '''
        Returns a view over the contents of SpeciesStateTable reshaped so all data
        has the correct (1/2/3D) dimensionality for the atmospheric model, as
        these are all stored under a flat scheme.
        '''
        shape = self.atmosphere.structure.dimensioned_shape
        table = copy(self)
        table.atmosphere = self.atmosphere.dimensioned_view()
        table.atomicPops = self.atomicPops.dimensioned_view(shape)
        table.molecularPops = [m.reshape(shape) for m in self.molecularPops]
        table.HminPops = self.HminPops.reshape(shape)
        return table

    def unit_view(self):
        '''
        Returns a view over the contents of the SpeciesStateTable with the correct
        `astropy.units`.
        '''
        table = copy(self)
        table.atmosphere = self.atmosphere.unit_view()
        table.atomicPops = self.atomicPops.unit_view()
        table.molecularPops = [(m << u.m**(-3)) for m in self.molecularPops]
        table.HminPops = self.HminPops << u.m**(-3)
        return table

    def dimensioned_unit_view(self):
        '''
        Returns a view over the contents of SpeciesStateTable reshaped so all data
        has the correct (1/2/3D) dimensionality for the atmospheric model,
        and the correct `astropy.units`.
        '''
        table = self.dimensioned_view()
        return table.unit_view()

    def __getitem__(self, name: Union[int, Tuple[int, int], str, Element]) -> np.ndarray:
        if isinstance(name, str) and  name == 'H-':
            return self.HminPops
        else:
            if name in self.molecularTable:
                name = cast(str, name)
                key = self.molecularTable.indices[name]
                return self.molecularPops[key]
            elif name in self.atomicPops:
                return self.atomicPops[name].n

    def __contains__(self, name: Union[int, Tuple[int, int], str, Element]) -> bool:
        if name == 'H-':
            return True

        if name in self.molecularTable:
            return True

        if name in self.atomicPops:
            return True

        return False

    def update_lte_atoms_Hmin_pops(self, atmos: Atmosphere, conserveCharge=False,
                                   updateTotals=False, maxIter=2000, quiet=False, tol=1e-3):
        '''
        Under the assumption that the atmosphere has changed, update the LTE
        atomic populations and the H- populations.

        Parameters
        ----------
        atmos : Atmosphere
            The atmosphere object.
        conserveCharge : bool
            Whether to conserveCharge and adjust the electron density in
            atmos based on the change in ionisation of the non-detailed
            species (default: False).
        updateTotals : bool, optional
            Whether to update the totals of each species from the abundance
            and total hydrogen density (default: False).
        maxIter : int, optional
            The maximum number of iterations to take looking for a stable
            solution (default: 2000).
        quiet : bool, optional
            Whether to print information about the update (default: False)
        tol : float, optional
            The tolerance of relative change at which to consider the
            populations converged (default: 1e-3)
        '''
        if updateTotals:
            for atom in self.atomicPops:
                atom.update_nTotal(atmos)
        for i in range(maxIter):
            maxDiff = 0.0
            maxName = '--'
            ne = np.zeros_like(atmos.ne)
            diffs = [update_lte_pops_inplace(atom.model, atmos.temperature,
                                             atmos.ne, atom.nTotal, atom.nStar,
                                             debye=True)[1] for atom in self.atomicPops]

            for j, atom in enumerate(self.atomicPops):
                if conserveCharge:
                    stages = np.array([l.stage for l in atom.model.levels])
                    if atom.pops is None:
                        ne += np.sum(atom.nStar * stages[:, None], axis=0)
                    else:
                        ne += np.sum(atom.n * stages[:, None], axis=0)

                diff = diffs[j]
                if diff > maxDiff:
                    maxDiff = diff
                    maxName = atom.name
            if conserveCharge:
                ne[ne < 1e6] = 1e6
                atmos.ne[:] = ne
            if maxDiff < tol:
                if not quiet:
                    print('LTE Iterations %d (%s slowest convergence)' % (i+1, maxName))
                break

        else:
            raise ValueError('No convergence in LTE update')

        self.HminPops[:] = hminus_pops(atmos, self.atomicPops['H'])


class RadiativeSet:
    '''
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
    passiveSet : set of Elements
        Set of atoms (designmated by their Elements) set to passive in the
        simulation.
    detailedStaticSet : set of Elements
        Set of atoms (designmated by their Elements) set to "detailed static" in the
        simulation.
    activeSet : set of Elements
        Set of atoms (designmated by their Elements) set to active in the
        simulation.
    '''
    def __init__(self, atoms: List[AtomicModel], abundance: AtomicAbundance=DefaultAtomicAbundance):
        self.abundance = abundance
        self.elements = [a.element for a in atoms]
        self.atoms = {k: v for k, v in zip(self.elements, atoms)}
        self.passiveSet = set(self.elements)
        self.detailedStaticSet: Set[Element] = set()
        self.activeSet: Set[Element] = set()

        if len(self.passiveSet) > len(self.elements):
            raise ValueError('Multiple entries for an atom: %s' % self.atoms)

    def __contains__(self, x: Union[int, Tuple[int, int], str, Element]) -> bool:
        return PeriodicTable[x] in self.elements

    def is_active(self, name: Union[int, Tuple[int, int], str, Element]) -> bool:
        '''
        Check if an atom (designated by int, (int, int), str, or Element) is
        active.
        '''
        x = PeriodicTable[name]
        return x in self.activeSet

    def is_passive(self, name: Union[int, Tuple[int, int], str, Element]) -> bool:
        '''
        Check if an atom (designated by int, (int, int), str, or Element) is
        passive.
        '''
        x = PeriodicTable[name]
        return x in self.passiveSet

    def is_detailed(self, name: Union[int, Tuple[int, int], str, Element]) -> bool:
        '''
        Check if an atom (designated by int, (int, int), str, or Element) is
        passive.
        '''
        x = PeriodicTable[name]
        return x in self.detailedStaticSet

    @property
    def activeAtoms(self) -> List[AtomicModel]:
        '''
        List of AtomicModels set to active.
        '''
        activeAtoms : List[AtomicModel] = [self.atoms[e] for e in self.activeSet]
        activeAtoms = sorted(activeAtoms, key=element_sort)
        return activeAtoms

    @property
    def detailedAtoms(self) -> List[AtomicModel]:
        '''
        List of AtomicModels set to detailed static.
        '''
        detailedAtoms : List[AtomicModel] = [self.atoms[e] for e in self.detailedStaticSet]
        detailedAtoms = sorted(detailedAtoms, key=element_sort)
        return detailedAtoms

    @property
    def passiveAtoms(self) -> List[AtomicModel]:
        '''
        List of AtomicModels set to passive.
        '''
        passiveAtoms : List[AtomicModel] = [self.atoms[e] for e in self.passiveSet]
        passiveAtoms = sorted(passiveAtoms, key=element_sort)
        return passiveAtoms

    def __getitem__(self, name: Union[int, Tuple[int, int], str, Element]) -> AtomicModel:
        x = PeriodicTable[name]
        return self.atoms[x]

    def __iter__(self):
        return iter(self.atoms.values())

    def set_active(self, *args: str):
        '''
        Set one (or multiple) atoms active.
        '''
        names = set(args)
        xs = [PeriodicTable[name] for name in names]
        for x in xs:
            self.activeSet.add(x)
            self.detailedStaticSet.discard(x)
            self.passiveSet.discard(x)

    def set_detailed_static(self, *args: str):
        '''
        Set one (or multiple) atoms to detailed static
        '''
        names = set(args)
        xs = [PeriodicTable[name] for name in names]
        for x in xs:
            self.detailedStaticSet.add(x)
            self.activeSet.discard(x)
            self.passiveSet.discard(x)

    def set_passive(self, *args: str):
        '''
        Set one (or multiple) atoms passive.
        '''
        names = set(args)
        xs = [PeriodicTable[name] for name in names]
        for x in xs:
            self.passiveSet.add(x)
            self.activeSet.discard(x)
            self.detailedStaticSet.discard(x)

    def iterate_lte_ne_eq_pops(self, atmos: Atmosphere,
                               mols: Optional[MolecularTable]=None,
                               nlteStartingPops: Optional[Dict[Element, np.ndarray]]=None,
                               direct: bool=False) -> SpeciesStateTable:
        '''
        Compute the starting populations for the simulation with all NLTE
        atoms in LTE or otherwise using the provided populations.
        Additionally computes a self-consistent LTE electron density.

        Parameters
        ----------
        atmos : Atmosphere
            The atmosphere for which to compute the populations.
        mols : MolecularTable, optional
            Molecules to be included in the populations (default: None)
        nlteStartingPops : Dict[Element, np.ndarray], optional
            Starting population override for any active or detailed static
            species.
        direct : bool
            Whether to use the direct electron density solver (essentially, Lambda iteration for the fixpoint), this may require many more iterations than the Newton-Krylov solver used otherwise, but may also be more robust.

        Returns
        -------
        eqPops : SpeciesStatTable
            The configured initial populations.
        '''
        if mols is None:
            mols = MolecularTable([])

        if nlteStartingPops is None:
            nlteStartingPops = {}
        else:
            for e in nlteStartingPops:
                if (e not in self.activeSet) \
                   and (e not in self.detailedStaticSet):
                    raise ValueError('Provided NLTE Populations for %s assumed LTE. Ensure these are indexed by `Element` rather than str.' % e)

        if direct:
            maxIter = 3000
            prevNe = np.copy(atmos.ne)
            ne = np.copy(atmos.ne)
            atoms = sorted(self.atoms.values(), key=element_sort)
            for it in range(maxIter):
                atomicPops = []
                prevNe[:] = ne
                ne.fill(0.0)
                for a in atoms:
                    abund = self.abundance[a.element]
                    nTotal = abund * atmos.nHTot
                    nStar = lte_pops(a, atmos.temperature, atmos.ne, nTotal, debye=True)
                    atomicPops.append(AtomicState(model=a, abundance=abund, nStar=nStar, nTotal=nTotal))

                    # NOTE(cmo): Take into account NLTE pops if provided
                    if a.element in nlteStartingPops:
                        if nlteStartingPops[a.element].shape != nStar.shape:
                            raise ValueError('Starting populations provided for %s do not match model.' % a.element)
                        nStar = nlteStartingPops[a.element]

                    stages = np.array([l.stage for l in a.levels])
                    ne += np.sum(nStar * stages[:, None], axis=0)
                atmos.ne[:] = ne

                relDiff = np.nanmax(np.abs(1.0 - prevNe / ne))
                print(relDiff)
                maxRelDiff = np.nanmax(relDiff)
                if maxRelDiff < 1e-3:
                    print("Iterate LTE: %d iterations" % it)
                    break
            else:
                print("LTE ne failed to converge")
        else:
            neRatio = np.copy(atmos.ne) / atmos.nHTot
            iterator = LteNeIterator(self.atoms.values(), atmos.temperature,
                                     atmos.nHTot, self.abundance, nlteStartingPops)
            neRatio += iterator(neRatio)
            newNeRatio = newton_krylov(iterator, neRatio)
            atmos.ne[:] = newNeRatio * atmos.nHTot

            atomicPops = iterator.atomicPops

        detailedAtomicPops = []
        for pop in atomicPops:
            ele = pop.model.element
            if ele in self.passiveSet:
                if ele in nlteStartingPops:
                    # NOTE(cmo): I don't believe this is possible; it would need
                    # to be detailed_static as per the contract on passive atoms
                    # being "true" LTE.  Leaving for now for safety.
                    pop.n = np.copy(nlteStartingPops[ele])
                detailedAtomicPops.append(pop)
            else:
                nltePops = np.copy(nlteStartingPops[ele]) if ele in nlteStartingPops else np.copy(pop.nStar)
                detailedAtomicPops.append(AtomicState(model=pop.model, abundance=self.abundance[ele],
                                                      nStar=pop.nStar, nTotal=pop.nTotal, detailed=True,
                                                      pops=nltePops))

        table = AtomicStateTable(detailedAtomicPops)
        eqPops = chemical_equilibrium_fixed_ne(atmos, mols, table, self.abundance)
        # NOTE(cmo): This is technically not quite correct, because we adjust
        # nTotal and the atomic populations to account for the atoms bound up
        # in molecules, but not n_e, this is unlikely to make much difference
        # in reality, other than in very cool atmospheres with a lot of
        # molecules (even then it should be pretty tiny)
        return eqPops

    def compute_eq_pops(self, atmos: Atmosphere,
                        mols: Optional[MolecularTable]=None,
                        nlteStartingPops: Optional[Dict[Element, np.ndarray]]=None):
        '''
        Compute the starting populations for the simulation with all NLTE
        atoms in LTE or otherwise using the provided populations.

        Parameters
        ----------
        atmos : Atmosphere
            The atmosphere for which to compute the populations.
        mols : MolecularTable, optional
            Molecules to be included in the populations (default: None)
        nlteStartingPops : Dict[Element, np.ndarray], optional
            Starting population override for any active or detailed static
            species.

        Returns
        -------
        eqPops : SpeciesStatTable
            The configured initial populations.
        '''
        if mols is None:
            mols = MolecularTable([])

        if nlteStartingPops is None:
            nlteStartingPops = {}
        else:
            for e in nlteStartingPops:
                if (e not in self.activeSet) \
                   and (e not in self.detailedStaticSet):
                    raise ValueError('Provided NLTE Populations for %s assumed LTE. Ensure these are indexed by `Element` rather than str.' % e)

        atomicPops = []
        atoms = sorted(self.atoms.values(), key=element_sort)
        for a in atoms:
            nTotal = self.abundance[a.element] * atmos.nHTot
            nStar = lte_pops(a, atmos.temperature, atmos.ne, nTotal, debye=True)

            ele = a.element
            if ele in self.passiveSet:
                n = None
                atomicPops.append(AtomicState(model=a, abundance=self.abundance[ele], nStar=nStar,
                                              nTotal=nTotal, pops=n))
            else:
                nltePops = np.copy(nlteStartingPops[ele]) if ele in nlteStartingPops else np.copy(nStar)
                atomicPops.append(AtomicState(model=a, abundance=self.abundance[ele],
                                              nStar=nStar, nTotal=nTotal, detailed=True,
                                              pops=nltePops))

        table = AtomicStateTable(atomicPops)
        eqPops = chemical_equilibrium_fixed_ne(atmos, mols, table, self.abundance)
        # NOTE(cmo): This is technically not quite correct, because we adjust
        # nTotal and the atomic populations to account for the atoms bound up
        # in molecules, but not n_e, this is unlikely to make much difference
        # in reality, other than in very cool atmospheres with a lot of
        # molecules (even then it should be pretty tiny)
        return eqPops

    def compute_wavelength_grid(self, extraWavelengths: Optional[np.ndarray]=None, lambdaReference=500.0) -> SpectrumConfiguration:
        '''
        Compute the global wavelength grid from the current configuration of
        the RadiativeSet.

        Parameters
        ----------
        extraWavelengths : np.ndarray, optional
            Extra wavelengths to add to the global array [nm].
        lambdaReference : float, optional
            If a difference reference wavelength is to be used then it should
            be specified here to ensure it is in the global array.

        Returns
        -------
        spect : SpectrumConfiguration
            The configured wavelength grids needed to set up the backend.
        '''
        if len(self.activeSet) == 0 and len(self.detailedStaticSet) == 0:
            raise ValueError('Need at least one atom active or in detailed calculation with static populations.')
        extraGrids = []
        if extraWavelengths is not None:
            extraGrids.append(extraWavelengths)
        extraGrids.append(np.array([lambdaReference]))

        models: List[AtomicModel] = []
        ids: List[Tuple[Element, int, int]] = []
        grids = []

        for ele in (self.activeSet | self.detailedStaticSet):
            atom = self.atoms[ele]
            models.append(atom)
            for trans in atom.transitions:
                grids.append(trans.wavelength())
                ids.append(trans.transId)

        grid = np.concatenate(grids + extraGrids)
        grid = np.sort(grid)
        grid = np.unique(grid)
        # grid = np.unique(np.floor(1e10*grid)) / 1e10
        blueIdx = {}
        redIdx = {}

        for i, g in enumerate(grids):
            ident = ids[i]
            blueIdx[ident] = np.searchsorted(grid, g[0])
            redIdx[ident] = np.searchsorted(grid, g[-1])+1

        transGrids: Dict[Tuple[Element, int, int], np.ndarray] = {}
        for ident in ids:
            transGrids[ident] = np.copy(grid[blueIdx[ident]:redIdx[ident]])

        activeWavelengths = {k: ((grid >= v[0]) & (grid <= v[-1])) for k, v in transGrids.items()}
        activeTrans = {k: True for k in transGrids}

        return SpectrumConfiguration(radSet=self, wavelength=grid, models=models, transWavelengths=transGrids,
                                     blueIdx=blueIdx, activeTrans=activeTrans, activeWavelengths=activeWavelengths)


def hminus_pops(atmos: Atmosphere, hPops: AtomicState) -> np.ndarray:
    '''
    Compute the H- ion populations for a given atmosphere

    Parameters
    ----------
    atmos : Atmosphere
        The atmosphere object.

    hPops : AtomicState
        The hydrogen populations state associated with atmos.

    Returns
    -------
    HminPops : np.ndarray
        The H- populations.
    '''
    CI = (Const.HPlanck / (2.0 * np.pi * Const.MElectron)) * (Const.HPlanck / Const.KBoltzmann)
    Nspace = atmos.Nspace

    PhiHmin = 0.25 * (CI / atmos.temperature)**1.5 \
                * np.exp(Const.E_ION_HMIN / (Const.KBoltzmann * atmos.temperature))
    HminPops = atmos.ne * np.sum(hPops.n, axis=0) * PhiHmin

    return HminPops

def chemical_equilibrium_fixed_ne(atmos: Atmosphere, molecules: MolecularTable,
                                  atomicPops: AtomicStateTable,
                                  abundance: AtomicAbundance) -> SpeciesStateTable:
    '''
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
    atomicPops : AtomicStateTable
        The atomic populations.
    abundance : AtomicAbundance
        The abundance of each species in the simulation.

    Returns
    -------
    state : SpeciesState
        The combined state object of atomic and molecular populations.
    '''
    nucleiSet: Set[Element] = set()
    for mol in molecules:
        nucleiSet |= set(mol.elements)
    nuclei: List[Element] = list(nucleiSet)
    nuclei = sorted(nuclei)

    if len(nuclei) == 0:
        HminPops = hminus_pops(atmos, atomicPops['H'])
        result = SpeciesStateTable(atmos, abundance, atomicPops, molecules, [], HminPops)
        return result

    if nuclei[0] != PeriodicTable[1]:
        raise ValueError('H not list of nuclei -- check H2 molecule')
    # print([n.name for n in nuclei])

    nuclIndex = [[nuclei.index(ele) for ele in mol.elements] for mol in molecules]

    # Replace basic elements with full Models if present
    kuruczTable = KuruczPfTable(atomicAbundance=abundance)
    nucData: Dict[Element, Union[KuruczPf, AtomicState]] = {}
    for nuc in nuclei:
        if nuc in atomicPops:
            nucData[nuc] = atomicPops[nuc]
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
    HminPops = np.zeros(Nspace)
    molPops = [np.zeros(Nspace) for mol in molecules]
    maxIter = 0
    for k in range(Nspace):
        for i, nuc in enumerate(nuclei):
            nucleus = nucData[nuc]
            a[i] = nucleus.abundance * atmos.nHTot[k]
            fjk, dfjk = nucleus.fjk(atmos, k)
            fn0[i] = fjk[0]

        PhiHmin = 0.25 * (CI / atmos.temperature[k])**1.5 \
                    * np.exp(Const.E_ION_HMIN / (Const.KBoltzmann * atmos.temperature[k]))
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
        NmaxIter = 50
        IterLimit = 1e-3
        prevN = n.copy()
        while nIter < NmaxIter:
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
                    saha *= (fn0[nu] * n[nu])**mol.elementCount[j]
                    # Contribution to conservation for each nucleus in this molecule
                    f[nu] += mol.elementCount[j] * n[Nnuclei + i]

                saha /= atmos.ne[k]**mol.charge
                f[Nnuclei + i] -= saha
                # if Nnuclei + i == f.shape[0]-1:
                #     print(i)
                #     print(saha)

                # Compute derivative matrix
                for j, ele in enumerate(mol.elements):
                    nu = nuclIndex[i][j]
                    df[nu, Nnuclei + i] += mol.elementCount[j]
                    df[Nnuclei + i, nu] = -saha * (mol.elementCount[j] / n[nu])

            correction = solve(df, f)
            n -= correction

            dnMax = np.nanmax(np.abs(1.0 - prevN / n))
            if dnMax <= IterLimit:
                maxIter = max(maxIter, nIter)
                break

            nIter += 1
        if dnMax > IterLimit:
            raise ValueError("ChemEq iteration not converged: T: %e [K], density %e [m^-3], dnmax %e" % (atmos.temperature[k], atmos.nHTot[k], dnMax))

        for i, ele in enumerate(nuclei):
            if ele in atomicPops:
                atomPop = atomicPops[ele]
                fraction = n[i] / atomPop.nTotal[k]
                atomPop.nStar[:, k] *= fraction
                atomPop.nTotal[k] *= fraction
                if atomPop.pops is not None:
                    atomPop.pops[:, k] *= fraction

        HminPops[k] = atmos.ne[k] * n[0] * PhiHmin

        for i, pop in enumerate(molPops):
            pop[k] = n[Nnuclei + i]

    result = SpeciesStateTable(atmos, abundance, atomicPops, molecules, molPops, HminPops)
    print("chem_eq: maximum number of iterations taken: %d" % maxIter)
    return result