import numpy as np
from .atomic_set import lte_pops
from scipy.linalg import solve

def F(self, k, backgroundNe=0.0):
    Nlevel = 0
    for atom in self.activeAtoms:
        Nlevel += atom.Nlevel
    Neqn = Nlevel + 1

    stages = [np.array([l.stage for l in atom.atomicModel.levels]) 
              for atom in self.activeAtoms]

    F = np.zeros(Neqn)
    F[-1] = self.atmos.ne[k]
    start = 0
    for idx, atom in enumerate(self.activeAtoms):
        F[start:start+atom.Nlevel] = -(atom.Gamma[:, :, k] @ atom.n[:, k])
        F[start + atom.Nlevel - 1] = np.sum(atom.n[:, k]) - atom.nTotal[k]
        F[-1] -= stages[idx] @ atom.n[:, k]
        start += atom.Nlevel
    F[-1] -= backgroundNe

    return F


def nr_post_update(self, fdCollisionRates=True):
    assert self.activeAtoms[0].atomicModel.name.startswith('H')
    crswVal = self.crswCallback.val

    Nlevel = 0
    for atom in self.activeAtoms:
        Nlevel += atom.Nlevel
    Neqn = Nlevel + 1

    Nspace = self.atmos.Nspace
    stages = [np.array([l.stage for l in atom.atomicModel.levels]) 
              for atom in self.activeAtoms]
    
    backgroundNe = np.zeros_like(self.atmos.ne)
    for idx, atomModel in enumerate(self.arguments['spect'].radSet.passiveAtoms):
        lteStages = np.array([l.stage for l in atomModel.levels]) 
        atom = self.arguments['eqPops'].atomicPops[atomModel.name]
        backgroundNe += (lteStages[:, None] * atom.n[:, :]).sum(axis=0)
              
    neStart = np.copy(self.atmos.ne)

    if fdCollisionRates:
        dC = []
        for atom in self.activeAtoms:
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
    for k in range(Nspace):
        dF[...] = 0.0
        Fg = self.F(k, backgroundNe[k])

        start = 0
        for idx, atom in enumerate(self.activeAtoms):
            Nlevel = atom.Nlevel
            dF[start:start+Nlevel, start:start+Nlevel] = -atom.Gamma[:, :, k]
            for t in atom.trans:
                if t.type == 'Continuum':
                    # dF[start + t.i, Neqn-1] -= (t.Rji[k] / self.atmos.ne[k]) * atom.n[t.j, k]
                    preconRji = atom.Gamma[t.i, t.j, k] - crswVal * atom.C[t.i, t.j, k]
                    dF[start + t.i, Neqn-1] -= (preconRji / self.atmos.ne[k]) * atom.n[t.j, k]

            if fdCollisionRates:
                for i in range(Nlevel):
                    dF[start + i, Neqn-1] -= dC[idx][i, :, k] @ atom.n[:, k]

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
        for atom in self.activeAtoms:
            atom.n[:, k] += update[start:start+atom.Nlevel]
            start += atom.Nlevel
        self.atmos.ne[k] += update[-1]

        Fnew[:] = 0.0
        start = 0
        for atom in self.activeAtoms:
            Fnew[start:start+atom.Nlevel] = atom.n[:, k]
            start += atom.Nlevel
        Fnew[-1] = self.atmos.ne[k]

        if np.max(np.abs(update/Fnew)) > maxChange:
            maxChange = np.max(np.abs(update/Fnew))
            maxIdx = np.argmax(np.abs(update/Fnew))
            maxk = k


    print('NR Update dPops: %.2e (%d, k: %d)' % (maxChange, maxIdx, maxk))
    return maxChange