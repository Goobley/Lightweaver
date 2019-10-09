from dataclasses import dataclass, field
from AtomicModel import *
from Molecule import Molecule
from typing import List, Sequence, Set, Optional, Any, Union, Dict
from copy import copy

@dataclass
class SpectrumConfiguration:
    wavelength: np.ndarray
    transitions: List[Union[AtomicLine, AtomicContinuum]]
    blueIdx: List[int]
    redIdx: List[int]
    activeSet: List[List[Union[AtomicLine, AtomicContinuum]]]
    activeLines: List[List[AtomicLine]]
    activeContinua: List[List[AtomicContinuum]]
    contributors: List[List[AtomicModel]]
    continuaPerAtom: Dict[str, List[List[AtomicContinuum]]]
    linesPerAtom: Dict[str, List[List[AtomicLine]]]
    lowerLevels: Dict[str, List[Set[int]]]
    upperLevels: Dict[str, List[Set[int]]]


@dataclass
class RadiativeSet:
    atoms: List[AtomicModel]
    molecules: List[Molecule]
    activeAtoms: Set[AtomicModel] = field(default_factory=set)
    detailedLteAtoms: Set[AtomicModel] = field(default_factory=set)
    passiveAtoms: Set[AtomicModel] = field(init=False)
    passiveMolecules: Set[Molecule] = field(init=False)

    def __post_init__(self):
        self.passiveAtoms = set(self.atoms)
        self.passiveMolecules = set(self.molecules)
        self.atomicNames = []
        for atom in self.atoms:
            self.atomicNames.append(atom.name)

        if len(self.atomicNames) > len(set(self.atomicNames)):
            raise ValueError('Multiple entries for an atom: %s' % self.atoms)

    def __getitem__(self, name: str) -> AtomicModel:
        name = name.upper()
        if len(name) == 1 and name not in self.atomicNames:
            name += ' '

        return self.atoms[self.atomicNames.index(name)]

    def validate_sets(self):
        if self.activeAtoms | self.passiveAtoms | self.detailedLteAtoms != set(self.atoms):
            raise ValueError('Problem with distribution of Atoms inside AtomicSet')
    
    def set_active(self, *args: str):
        names = set(args)
        for atomName in names:
            self.activeAtoms.add(self[atomName])
            self.detailedLteAtoms.discard(self[atomName])
            self.passiveAtoms.discard(self[atomName])
        self.validate_sets()

    def set_detailed_lte(self, *args: str):
        names = set(args)
        for atomName in names:
            self.detailedLteAtoms.add(self[atomName])
            self.activeAtoms.discard(self[atomName])
            self.passiveAtoms.discard(self[atomName])
        self.validate_sets()

    def set_passive(self, *args: str):
        names = set(args)
        for atomName in names:
            self.passiveAtoms.add(self[atomName])
            self.activeAtoms.discard(self[atomName])
            self.detailedLteAtoms.discard(self[atomName])
        self.validate_sets()

    def compute_wavelength_grid(self, extraWavelengths: Optional[np.ndarray]=None, lambdaReference=500.0) -> SpectrumConfiguration:
        if len(self.activeAtoms) == 0 and len(self.detailedLteAtoms) == 0:
            raise ValueError('Need at least one atom active or in detailed LTE')
        grids = []
        if extraWavelengths is not None:
            grids.append(extraWavelengths)
        grids.append(np.array([lambdaReference]))

        transitions: List[Union[AtomicLine, AtomicContinuum]] = []
        continuaPerAtom: Dict[str, List[List[AtomicContinuum]]] = {}
        linesPerAtom: Dict[str, List[List[AtomicLine]]]= {}
        upperLevels: Dict[str, List[Set[int]]] = {}
        lowerLevels: Dict[str, List[Set[int]]] = {}

        for atom in self.activeAtoms:
            continuaPerAtom[atom.name] = []
            linesPerAtom[atom.name] = []
            lowerLevels[atom.name] = []
            upperLevels[atom.name] = []
            for line in atom.lines:
                transitions.append(line)
                grids.append(line.wavelength)
            for cont in atom.continua:
                transitions.append(cont)
                grids.append(cont.wavelength)

        for atom in self.detailedLteAtoms:
            for line in atom.lines:
                grids.append(line.wavelength)
            for cont in atom.continua:
                grids.append(cont.wavelength)

        grid = np.concatenate(grids)
        grid = np.sort(grid)
        grid = np.unique(grid)
        blueIdx = []
        redIdx = []
        Nlambda = []

        for t in transitions:
            blueIdx.append(np.searchsorted(grid, t.wavelength[0]))
            redIdx.append(np.searchsorted(grid, t.wavelength[-1])+1)
            Nlambda.append(redIdx[-1] - blueIdx[-1])

        for i, t in enumerate(transitions):
            wavelength = np.copy(grid[blueIdx[i]:redIdx[i]])
            if isinstance(t, AtomicContinuum):
                t.alpha = t.compute_alpha(wavelength)
            t.wavelength = wavelength
            t.Nlambda = Nlambda[i] # type: ignore

        activeSet: List[List[Union[AtomicLine, AtomicContinuum]]] = []
        activeLines: List[List[AtomicLine]] = []
        activeContinua: List[List[AtomicContinuum]] = []
        contributors: List[List[AtomicModel]] = []
        for i in range(grid.shape[0]):
            activeSet.append([])
            activeLines.append([])
            activeContinua.append([])
            contributors.append([])
            for atom in self.activeAtoms:
                continuaPerAtom[atom.name].append([])
                linesPerAtom[atom.name].append([])
                upperLevels[atom.name].append(set())
                lowerLevels[atom.name].append(set())
            for kr, t in enumerate(transitions):
                if blueIdx[kr] <= i < redIdx[kr]:
                    activeSet[-1].append(t)
                    contributors[-1].append(t.atom)
                    if isinstance(t, AtomicContinuum):
                        activeContinua[-1].append(t)
                        continuaPerAtom[t.atom.name][-1].append(t)
                        upperLevels[t.atom.name][-1].add(t.j)
                        lowerLevels[t.atom.name][-1].add(t.i)
                    elif isinstance(t, AtomicLine):
                        activeLines[-1].append(t)
                        linesPerAtom[t.atom.name][-1].append(t)
                        upperLevels[t.atom.name][-1].add(t.j)
                        lowerLevels[t.atom.name][-1].add(t.i)


        return SpectrumConfiguration(grid, transitions, blueIdx, redIdx, activeSet, activeLines, activeContinua, contributors, continuaPerAtom, linesPerAtom, lowerLevels, upperLevels)


