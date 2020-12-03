import numpy as np
from .atomic_set import lte_pops
from scipy.linalg import solve
from .atomic_table import PeriodicTable

def nr_post_update(self, fdCollisionRates=True, hOnly=False,
                   timeDependentData=None, chunkSize=5,
                   ngUpdate=None, printUpdate=None):
    '''
    Compute the Newton-Raphson terms for updating the electron density
    through charge conservation. Is attached to the Context object.

    Parameters
    ----------
    fdCollisionRates : bool, optional
        Whether to use a finite difference approximation to the collisional
        rates to find the population gradient WRT ne (default: True i.e. use
        finite-difference, if False, collisional rates are ignored for this
        process.)
    hOnly : bool, optional
        Ignore atoms other than Hydrogen (the primary electron contributor)
        (default: False)
    timeDependentData : dict, optional
        The presence of this argument indicates that the time-dependent
        formalism should be used. Should contain the keys 'dt' with a
        floating point timestep, and 'nPrev' with a list of population
        vectors from the start of the time integration step, in the order of
        the active atoms. This latter term can be obtained from the previous
        state provied by `Context.time_dep_update`.
    chunkSize : int, optional
        Not currently used.
    ngUpdate : bool, optional
        Whether to apply Ng Acceleration (default: None, to apply automatic
        behaviour), will only accelerate if the counter on the Ng accelerator
        has seen enough steps since the previous acceleration (set in Context
        initialisation).
    printUpdate : bool, optional
        Whether to print information on the size of the update (default:
        None, to apply automatic behaviour).

    Returns
    -------
    dPops : float
        The maximum relative change of any of the NLTE populations in the
        atmosphere.
    '''
    if self.activeAtoms[0].element != PeriodicTable[1]:
        raise ValueError('Calling nr_post_update without Hydrogen active.')

    if ngUpdate is None:
        if self.conserveCharge:
            ngUpdate = True
        else:
            ngUpdate = False

    if printUpdate is None:
        printUpdate = ngUpdate

    timeDependent = (timeDependentData is not None)
    atoms = self.activeAtoms[:1] if hOnly else self.activeAtoms
    crswVal = self.crswCallback.val

    if hOnly:
        backgroundAtoms = [model for ele, model in self.kwargs['spect'].radSet.items() if ele != PeriodicTable[1]]
    else:
        backgroundAtoms = self.kwargs['spect'].radSet.passiveAtoms

    backgroundNe = np.zeros_like(self.atmos.ne)
    for idx, atomModel in enumerate(backgroundAtoms):
        lteStages = np.array([l.stage for l in atomModel.levels])
        atom = self.kwargs['eqPops'].atomicPops[atomModel.element]
        backgroundNe += (lteStages[:, None] * atom.n[:, :]).sum(axis=0)

    neStart = np.copy(self.atmos.ne)

    dC = []
    if fdCollisionRates:
        for atom in atoms:
            atom.compute_collisions(fillDiagonal=True)
            Cprev = np.copy(atom.C)
            pertSize = 1e-4
            pert = neStart * pertSize
            self.atmos.ne[:] += pert
            nStarPrev = np.copy(atom.nStar)
            atom.nStar[:] = lte_pops(atom.atomicModel, self.atmos.temperature,
                                     self.atmos.ne, atom.nTotal)
            atom.compute_collisions(fillDiagonal=True)
            self.atmos.ne[:] = neStart
            atom.nStar[:] = nStarPrev
            dC.append(crswVal * (atom.C - Cprev) / pert)
            atom.C[:] = Cprev

    self._nr_post_update_impl(atoms, dC, backgroundNe,
                              timeDependentData=timeDependentData, chunkSize=chunkSize)
    self.eqPops.update_lte_atoms_Hmin_pops(self.atmos.pyAtmos, conserveCharge=False, quiet=True)

    if ngUpdate:
        maxDelta = self.rel_diff_ng_accelerate(printUpdate=printUpdate)
    else:
        maxDelta = self.rel_diff_pops(printUpdate=printUpdate)
    neDiff = ((np.asarray(self.atmos.ne) - neStart)
                / np.asarray(self.atmos.ne)).max()
    maxDelta = max(maxDelta, neDiff)
    if printUpdate:
        print('    ne delta = %6.4e' % neDiff)
    return maxDelta