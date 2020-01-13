from .atomic_model import CollisionalRates, AtomicModel
import lightweaver.constants as Const
from dataclasses import dataclass
import numpy as np

@dataclass
class Ar85Cdi(CollisionalRates):
    cdi: np.ndarray
    def setup(self, atom):
        i, j = self.i, self.j
        self.i = min(i, j)
        self.j = max(i, j)
        self.atom = atom
        self.iLevel = atom.levels[self.i]
        self.jLevel = atom.levels[self.j]

    def compute_rates(self, atmos, nstar, Cmat):
        Cup = np.zeros(atmos.Nspace)
        for m in range(self.cdi.shape[0]):
            xj = self.cdi[m, 0] * Const.EV / (Const.KBoltzmann * atmos.temperature)
            fac = np.exp(-xj) * np.sqrt(xj)
            fxj = self.cdi[m, 1] + self.cdi[m, 2] * (1.0 + xj) + (self.cdi[m, 3] - xj * (self.cdi[m, 1] + self.cdi[m, 2] * (2.0 + xj))) * fone(xj) + self.cdi[m, 4] * xj * ftwo(xj)

            fxj *= fac
            fac = 6.69e-7 / self.cdi[m, 0]**1.5
            Cup += fac * fxj * Const.CM_TO_M**3
        Cup[Cup < 0] = 0.0

        Cup *= atmos.ne
        Cdown = Cup * nstar[self.i] / nstar[self.j]
        Cmat[self.i, self.j, :] += Cdown
        Cmat[self.j, self.i, :] += Cup

# NOTE(cmo): It's probably better to write an AR85-CEA function per atomic model, and skip all the series checking stuff
# For Helium, after checking gencol and rh it seems to return 0.0 for helium, so it can probably just be removed from the model