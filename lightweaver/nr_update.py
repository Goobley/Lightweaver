import numpy as np

from ._renames import NR_DICT_KEY_RENAMES
from .atomic_set import lte_pops
from .atomic_table import PeriodicTable
from .deprecation import accepts_old_kwargs, remap_old_keys


@accepts_old_kwargs
def nr_post_update(
    self,
    fd_collision_rates=True,
    h_only=False,
    time_dependent_data=None,
    chunk_size=5,
    ng_update=None,
    extra_params=None,
):
    """
    Compute the Newton-Raphson terms for updating the electron density
    through charge conservation. Is attached to the Context object.

    Parameters
    ----------
    fd_collision_rates : bool, optional
        Whether to use a finite difference approximation to the collisional
        rates to find the population gradient WRT ne (default: True i.e. use
        finite-difference, if False, collisional rates are ignored for this
        process.)
    h_only : bool, optional
        Ignore atoms other than Hydrogen (the primary electron contributor)
        (default: False)
    time_dependent_data : dict, optional
        The presence of this argument indicates that the time-dependent
        formalism should be used. Should contain the keys 'dt' with a
        floating point timestep, and 'n_prev' with a list of population
        vectors from the start of the time integration step, in the order of
        the active atoms. This latter term can be obtained from the previous
        state provied by `Context.time_dep_update`.
    chunk_size : int, optional
        Not currently used.
    ng_update : bool, optional
        Whether to apply Ng Acceleration (default: None, to apply automatic
        behaviour), will only accelerate if the counter on the Ng accelerator
        has seen enough steps since the previous acceleration (set in Context
        initialisation).
    extra_params : dict, optional
        Dict of extra parameters to be converted through the
        `dict2ExtraParams` function and passed onto the C++ core.
    Returns
    -------
    dpops : float
        The maximum relative change of any of the NLTE populations in the
        atmosphere.
    """
    if self.active_atoms[0].element != PeriodicTable[1]:
        raise ValueError('Calling nr_post_update without Hydrogen active.')

    if ng_update is None:
        ng_update = self.conserve_charge

    atoms = self.active_atoms[:1] if h_only else self.active_atoms
    crswVal = self.crsw_callback.val

    if h_only:
        backgroundAtoms = [
            model for model in self.kwargs['spect'].rad_set if model.element != PeriodicTable[1]
        ]
    else:
        backgroundAtoms = (
            self.kwargs['spect'].rad_set.detailed_atoms + self.kwargs['spect'].rad_set.passive_atoms
        )

    backgroundNe = np.zeros_like(self.atmos.ne)
    for atomModel in backgroundAtoms:
        lteStages = np.array([l.stage for l in atomModel.levels])
        atom = self.kwargs['eq_pops'].atomic_pops[atomModel.element]
        backgroundNe += (lteStages[:, None] * atom.n[:, :]).sum(axis=0)

    neStart = np.copy(self.atmos.ne)

    dC = []
    if fd_collision_rates:
        for atom in atoms:
            atom.compute_collisions(fill_diagonal=True)
            Cprev = np.copy(atom.C)
            pertSize = 1e-4
            pert = neStart * pertSize
            self.atmos.ne[:] += pert
            nStarPrev = np.copy(atom.n_star)
            atom.n_star[:] = lte_pops(
                atom.atomic_model, self.atmos.temperature, self.atmos.ne, atom.n_total
            )
            atom.compute_collisions(fill_diagonal=True)
            self.atmos.ne[:] = neStart
            atom.n_star[:] = nStarPrev
            dC.append(crswVal * (atom.C - Cprev) / pert)
            atom.C[:] = Cprev

    if time_dependent_data is not None:
        time_dependent_data = remap_old_keys(
            time_dependent_data, stacklevel=3, renames=NR_DICT_KEY_RENAMES
        )
    self._nr_post_update_impl(
        atoms,
        dC,
        backgroundNe,
        time_dependent_data=time_dependent_data,
        chunk_size=chunk_size,
        extra_params=extra_params,
    )
    self.eq_pops.update_lte_atoms_hmin_pops(self.atmos.py_atmos, conserve_charge=False, quiet=True)

    if ng_update:
        update = self.rel_diff_ng_accelerate()
    else:
        update = self.rel_diff_pops()
    neDiff = np.abs((np.asarray(self.atmos.ne) - neStart) / np.asarray(self.atmos.ne))
    neDiffMaxIdx = neDiff.argmax()
    neDiffMax = neDiff[neDiffMaxIdx]
    update.updated_ne = True
    update.dne_max = neDiffMax
    update.dne_max_idx = neDiffMaxIdx
    return update
