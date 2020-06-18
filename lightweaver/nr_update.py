import numpy as np
from .atomic_set import lte_pops
from scipy.linalg import solve
from .atomic_table import PeriodicTable

def Ftd(self, k, dt, nPrev, backgroundNe=0.0, atoms=None):
    Nlevel = 0
    if atoms is None:
        atoms = self.activeAtoms

    for atom in atoms:
        Nlevel += atom.Nlevel
    Neqn = Nlevel + 1

    stages = [np.array([l.stage for l in atom.atomicModel.levels])
              for atom in atoms]

    F = np.zeros(Neqn)
    F[-1] = self.atmos.ne[k]
    start = 0
    theta = 1.0
    for idx, atom in enumerate(atoms):
        F[start:start+atom.Nlevel] = theta * dt * (atom.Gamma[:, :, k] @ atom.n[:, k]) \
                                      - (atom.n[:, k] - nPrev[idx][:, k])
        F[start + atom.Nlevel - 1] = np.sum(atom.n[:, k]) - atom.nTotal[k]
        F[-1] -= stages[idx] @ atom.n[:, k]
        start += atom.Nlevel
    F[-1] -= backgroundNe

    return F

def F(self, k, backgroundNe=0.0, atoms=None):
    Nlevel = 0
    if atoms is None:
        atoms = self.activeAtoms

    for atom in atoms:
        Nlevel += atom.Nlevel
    Neqn = Nlevel + 1

    stages = [np.array([l.stage for l in atom.atomicModel.levels])
              for atom in atoms]

    F = np.zeros(Neqn)
    F[-1] = self.atmos.ne[k]
    start = 0
    for idx, atom in enumerate(atoms):
        F[start:start+atom.Nlevel] = -(atom.Gamma[:, :, k] @ atom.n[:, k])
        F[start + atom.Nlevel - 1] = np.sum(atom.n[:, k]) - atom.nTotal[k]
        F[-1] -= stages[idx] @ atom.n[:, k]
        start += atom.Nlevel
    F[-1] -= backgroundNe

    return F


def nr_post_update(self, fdCollisionRates=True, hOnly=False, timeDependentData=None):
    assert self.activeAtoms[0].element == PeriodicTable[1]
    crswVal = self.crswCallback.val

    timeDependent = (timeDependentData is not None)
    atoms = self.activeAtoms[:1] if hOnly else self.activeAtoms

    Nlevel = 0
    for atom in atoms:
        Nlevel += atom.Nlevel
    Neqn = Nlevel + 1

    Nspace = self.atmos.Nspace
    stages = [np.array([l.stage for l in atom.atomicModel.levels])
              for atom in atoms]


    if hOnly:
        backgroundAtoms = [model for ele, model in self.arguments['spect'].radSet.items() if ele != PeriodicTable[1]]
    else:
        backgroundAtoms = self.arguments['spect'].radSet.passiveAtoms

    backgroundNe = np.zeros_like(self.atmos.ne)
    for idx, atomModel in enumerate(backgroundAtoms):
        lteStages = np.array([l.stage for l in atomModel.levels])
        atom = self.arguments['eqPops'].atomicPops[atomModel.element]
        backgroundNe += (lteStages[:, None] * atom.n[:, :]).sum(axis=0)

    neStart = np.copy(self.atmos.ne)

    if fdCollisionRates:
        dC = []
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

    maxChange = 0.0
    maxIdx = -1
    maxk = -1
    dF = np.zeros((Neqn, Neqn))
    Fnew = np.zeros(Neqn)
    theta = 1.0
    for k in range(Nspace):
        dF[...] = 0.0
        if timeDependent:
            Fg = self.Ftd(k, dt=timeDependentData['dt'],
                          nPrev=timeDependentData['nPrev'],
                          backgroundNe=backgroundNe[k], atoms=atoms)
        else:
            Fg = self.F(k, backgroundNe=backgroundNe[k], atoms=atoms)

        start = 0
        for idx, atom in enumerate(atoms):
            Nlevel = atom.Nlevel
            dF[start:start+Nlevel, start:start+Nlevel] = -atom.Gamma[:, :, k]
            if timeDependent:
                dF[start:start+Nlevel, start:start+Nlevel] *= -theta * timeDependentData['dt']
                dF[start:start+Nlevel, start:start+Nlevel] -= np.eye(Nlevel)

            for t in atom.trans:
                if t.type == 'Continuum':
                    # dF[start + t.i, Neqn-1] -= (t.Rji[k] / self.atmos.ne[k]) * atom.n[t.j, k]
                    preconRji = atom.Gamma[t.i, t.j, k] - crswVal * atom.C[t.i, t.j, k]
                    entry = -(preconRji / self.atmos.ne[k]) * atom.n[t.j, k]
                    if timeDependent:
                        entry *= -theta * timeDependentData['dt']
                    dF[start + t.i, Neqn-1] += entry


            if fdCollisionRates:
                for i in range(Nlevel):
                    entry = -dC[idx][i, :, k] @ atom.n[:, k]
                    if timeDependent:
                        entry *= -theta * timeDependentData['dt']
                    dF[start + i, Neqn-1] += entry

            dF[start+Nlevel-1, :] = 0.0
            dF[start+Nlevel-1, start:start+Nlevel] = 1.0

            dF[-1, start:start+Nlevel] = -stages[idx]
            start += Nlevel

        dF[-1, -1] = 1.0
        Fg *= -1.0

        update = solve(dF, Fg)
        # if k == 40:
        #     print('------ %d ------' % k)
        #     print(Fg)
        #     print(dF)
        #     print(update)
        #     # print(atom.n[:, k], self.atmos.ne[k])
        #     print(dF @ update)
        #     print('------------')
        #     # raise ValueError

        start = 0
        for atom in atoms:
            atom.n[:, k] += update[start:start+atom.Nlevel]
            start += atom.Nlevel
        self.atmos.ne[k] += update[-1]

        Fnew[:] = 0.0
        start = 0
        for atom in atoms:
            Fnew[start:start+atom.Nlevel] = atom.n[:, k]
            start += atom.Nlevel
        Fnew[-1] = self.atmos.ne[k]

        if np.max(np.abs(update/Fnew)) > maxChange:
            maxChange = np.max(np.abs(update/Fnew))
            maxIdx = np.argmax(np.abs(update/Fnew))
            maxk = k

    # NOTE(cmo): If we're here then conserveCharge has to be True, but we don't
    # actually want to mess with n_e, that's the point in this function. Should
    # handle LTE atoms here if we want to include their effects
    self.eqPops.update_lte_atoms_Hmin_pops(self.arguments['atmos'], conserveCharge=False, quiet=True)
    print('    NR Update dPops: %.2e (%d, k: %d)' % (maxChange, maxIdx, maxk))
    return maxChange