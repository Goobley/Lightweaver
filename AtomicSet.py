from dataclasses import dataclass, field
from AtomicModel import *
from typing import List, Sequence, Set, Optional, Any, Union
from copy import copy

@dataclass
class SpectrumConfiguration:
    wavelength: np.ndarray
    transitions: List[Union[AtomicLine, AtomicContinuum]]
    blueIdx: List[int]
    redIdx: List[int]
    activeSet: List[List[Union[AtomicLine, AtomicContinuum]]]


@dataclass
class AtomicSet:
    atoms: List[AtomicModel]
    active: Set[AtomicModel] = field(default_factory=set)
    detailedLte: Set[AtomicModel] = field(default_factory=set)
    passive: Set[AtomicModel] = field(init=False)

    def __post_init__(self):
        self.passive = set(self.atoms)
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
        if self.active | self.passive | self.detailedLte != set(self.atoms):
            raise ValueError('Problem with distribution of Atoms inside AtomicSet')
    
    def set_active(self, *args: str):
        names = set(args)
        for atomName in names:
            self.active.add(self[atomName])
            self.detailedLte.discard(self[atomName])
            self.passive.discard(self[atomName])
        self.validate_sets()

    def set_detailed_lte(self, *args: str):
        names = set(args)
        for atomName in names:
            self.detailedLte.add(self[atomName])
            self.active.discard(self[atomName])
            self.passive.discard(self[atomName])
        self.validate_sets()

    def set_passive(self, *args: str):
        names = set(args)
        for atomName in names:
            self.passive.add(self[atomName])
            self.active.discard(self[atomName])
            self.detailedLte.discard(self[atomName])
        self.validate_sets()

    def compute_wavelength_grid(self, extraWavelengths: Optional[np.ndarray]=None) -> SpectrumConfiguration:
        grids = []
        if extraWavelengths is not None:
            grids.append(extraWavelengths)

        transitions: List[Union[AtomicLine, AtomicContinuum]] = []

        for atom in self.active:
            for line in atom.lines:
                transitions.append(line)
                grids.append(line.wavelength)
            for cont in atom.continua:
                transitions.append(cont)
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
                t.alpha = t.compute_alpha(grid)
            t.wavelength = wavelength
            t.Nlambda = Nlambda[i] # type: ignore

        activeSet: List[List[Union[AtomicLine, AtomicContinuum]]] = []

        for i in range(grid.shape[0]):
            activeSet.append([])
            for kr, t in enumerate(transitions):
                if blueIdx[kr] <= i < redIdx[kr]:
                    activeSet[-1].append(t)

        return SpectrumConfiguration(grid, transitions, blueIdx, redIdx, activeSet)








    
