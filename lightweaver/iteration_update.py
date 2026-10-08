from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

from .deprecation import deprecated_names

if TYPE_CHECKING:
    from . import Context


@deprecated_names
@dataclass
class IterationUpdate:
    """
    Stores the results of an iteration of one of the backend functions, and
    determines how to format this for printing. All changes refer to relative
    change.

    Attributes
    ----------
    ctx : Context
        The context with which this update is associated.
    crsw :  float
        The current value of the collisional radiative switching parameter.
    updated_J : bool
        Whether the iteration affected the global J grid.
    dJ_max : float
        The maximum change in J.
    dJ_max_idx : int
        The index of the maximum change of J in a flattened array of J.
    updated_pops : bool
        Whether the active atomic populations were modified by the iteration.
    dpops : List[float]
        The maximum change in each active population.
    dpops_max_idx : List[int]
        The location of the maximum change in each population in the flattened
        population array.
    ng_accelerated : List[bool]
        Whether the atomic populations were modified by Ng Acceleration (per species, due to thresholding).
    updated_ne : bool
        Whether the electron density in the atmosphere was affected by the iteration.
    dne_max : float
        The maximum change in the electron density.
    dne_max_idx : int
        The location of the maximum change in the electron density array.
    updated_rho : bool
        Whether the iteration affected the value of rho_prd on PRD lines.
    Nprd_sub_iter : int
        The number of PRD sub-iterations taken (if multiple),
    drho : List[float]
        The maximum change in rho for each spectral line treated with PRD, in
        the order of the lines on each activeAtom. These values are repeated for
        each sub-iteration < Nprd_sub_iter.
    drho_max_idx : List[int]
        The location of the maximum change in rho for each PRD line, in the
        flattened rho_prd array (Nlambda, Nspace).
    updated_J_prd : bool
        Whether the PRD iteration affected J.
    dJ_prd_max : float
        The maximum change in J during each PRD sub-iteration.
    dJ_prd_max_idx : int
        The location of the maximum change in J for each PRD sub-iteration.
    dpops_max : float
        The maximum population change (including ne) over the iteration
        (read-only property).
    drho_max : float
        The maximum change in the PRD rho value for any line in the final
        subiteration (read-only property).
    """

    ctx: 'Context'
    crsw: float = 1.0
    updated_J: bool = False
    dJ_max: float = 0.0
    dJ_max_idx: int = 0

    updated_pops: bool = False
    dpops: List[float] = field(default_factory=list)
    dpops_max_idx: List[int] = field(default_factory=list)
    ng_accelerated: List[bool] = field(default_factory=list)

    updated_ne: bool = False
    dne_max: float = 0.0
    dne_max_idx: int = 0

    updated_rho: bool = False
    Nprd_sub_iter: int = 0
    drho: List[float] = field(default_factory=list)
    drho_max_idx: List[int] = field(default_factory=list)
    updated_J_prd: bool = False
    dJ_prd_max: List[float] = field(default_factory=list)
    dJ_prd_max_idx: List[int] = field(default_factory=list)

    @property
    def dpops_max(self) -> float:
        if len(self.dpops) == 0:
            if self.updated_ne:
                return self.dne_max
            else:
                return 0.0

        result = max(self.dpops)
        if self.updated_ne:
            result = max(result, self.dne_max)
        return result

    @property
    def drho_max(self) -> float:
        if self.Nprd_sub_iter == 0:
            return 0.0
        finalSubIterStart = (self.Nprd_sub_iter - 1) * self.ctx.kwargs['spect'].Nprd_trans
        return max(self.drho[finalSubIterStart:])

    def compact_representation(self):
        """
        Produce a compact string representation of the object (similar to
        Lightweaver < v0.8).
        """
        chunks = []
        if self.crsw != 1.0:
            chunks.append(f'CRSW: {self.crsw:.2e}')

        if self.updated_J:
            chunks.append(f'dJ = {self.dJ_max:.2e}')

        if self.updated_pops:
            for idx, delta in enumerate(self.dpops):
                atomName = self.ctx.active_atoms[idx].atomic_model.element.name
                accel = ' (accelerated)' if self.ng_accelerated[idx] else ''
                chunks.append(f'    {atomName} delta = {delta:6.4e}{accel}')

        if self.updated_ne:
            delta = self.dne_max
            chunks.append(f'    ne delta = {delta:6.4e}')

        if self.updated_rho:
            iterCount = self.Nprd_sub_iter
            drho_max = self.drho_max
            chunks.append(f'    PRD drho = {drho_max:.2e}, (sub-iterations: {iterCount})')

        return '\n'.join(chunks)
