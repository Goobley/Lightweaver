from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from . import Context

@dataclass
class IterationUpdate:
    '''
    Stores the results of an iteration of one of the backend functions, and
    determines how to format this for printing. All changes refer to relative
    change.

    Attributes
    ----------
    ctx : Context
        The context with which this update is associated.
    crsw :  float
        The current value of the collisional radiative switching parameter.
    updatedJ : bool
        Whether the iteration affected the global J grid.
    dJMax : float
        The maximum change in J.
    dJMaxIdx : int
        The index of the maximum change of J in a flattened array of J.
    updatedPops : bool
        Whether the active atomic populations were modified by the iteration.
    dPops : List[float]
        The maximum change in each active population.
    dPopsMaxIdx : List[int]
        The location of the maximum change in each population in the flattened
        population array.
    ngAccelerated : bool
        Whether the atomic populations were modified by Ng Acceleration.
    updatedNe : bool
        Whether the electron density in the atmosphere was affected by the iteration.
    dNeMax : float
        The maximum change in the electron density.
    dNeMaxIdx : int
        The location of the maximum change in the electron density array.
    updatedRho : bool
        Whether the iteration affected the value of rhoPrd on PRD lines.
    NprdSubIter : int
        The number of PRD sub-iterations taken (if multiple),
    dRho : List[float]
        The maximum change in rho for each spectral line treated with PRD, in
        the order of the lines on each activeAtom. These values are repeated for
        each sub-iteration < NprdSubIter.
    dRhoMaxIdx : List[int]
        The location of the maximum change in rho for each PRD line.
    updatedJPrd : bool
        Whether the PRD iteration affected J.
    dJPrdMax : float
        The maximum change in J during each PRD sub-iteration.
    dJPrdMaxIdx : int
        The location of the maximum change in J for each PRD sub-iteration.
    dPopsMax : float
        The maximum population change (including ne) over the iteration
        (read-only property).
    dRhoMax : float
        The maximum change in the PRD rho value for any line in the final
        subiteration (read-only property).
    '''
    ctx: 'Context'
    crsw: float = 1.0
    updatedJ: bool = False
    dJMax: float = 0.0
    dJMaxIdx: int = 0

    updatedPops: bool = False
    dPops: List[float] = field(default_factory=list)
    dPopsMaxIdx: List[int] = field(default_factory=list)
    ngAccelerated: bool = False

    updatedNe: bool = False
    dNeMax: float = 0.0
    dNeMaxIdx: int = 0

    updatedRho: bool = False
    NprdSubIter: int = 0
    dRho: List[float] = field(default_factory=list)
    dRhoMaxIdx: List[int] = field(default_factory=list)
    updatedJPrd: bool = False
    dJPrdMax: List[float] = field(default_factory=list)
    dJPrdMaxIdx: List[int] = field(default_factory=list)

    @property
    def dPopsMax(self) -> float:
        if len(self.dPops) == 0:
            if self.updatedNe:
                return self.dNeMax
            else:
                return 0.0

        result = max(self.dPops)
        if self.updatedNe:
            result = max(result, self.dNeMax)
        return result

    @property
    def dRhoMax(self) -> float:
        if self.NprdSubIter == 0:
            return 0.0
        finalSubIterStart = (self.NprdSubIter - 1) * self.ctx.kwargs['spect'].NprdTrans
        return max(self.dRho[finalSubIterStart:])

    def compact_representation(self):
        '''
        Produce a compact string representation of the object (similar to
        Lightweaver < v0.8).
        '''
        chunks = []
        if self.crsw != 1.0:
            chunks.append(f'CRSW: {self.crsw:.2e}')

        if self.updatedJ:
            chunks.append(f'dJ = {self.dJMax:.2e}')

        if self.updatedPops:
            for idx, delta in enumerate(self.dPops):
                atomName = self.ctx.activeAtoms[idx].atomicModel.element.name
                accel = ' (accelerated)' if self.ngAccelerated else ''
                chunks.append(f'    {atomName} delta = {delta:6.4e}{accel}')

        if self.updatedNe:
            delta = self.dNeMax
            chunks.append(f'    ne delta = {delta:6.4e}')

        if self.updatedRho:
            iterCount = self.NprdSubIter
            dRhoMax = self.dRhoMax
            chunks.append(f'    PRD dRho = {dRhoMax:.2e}, (sub-iterations: {iterCount})')

        return '\n'.join(chunks)
