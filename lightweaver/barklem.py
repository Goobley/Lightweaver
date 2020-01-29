from typing import Tuple, Iterable, List, Sequence, TYPE_CHECKING
from dataclasses import dataclass, field
from parse import parse
import numpy as np
if TYPE_CHECKING:
    from .atomic_model import AtomicLevel, AtomicLine, AtomicModel, determinate, CompositeLevelError
import string
from .atomic_table import AtomicTable
from .utils import get_data_path
from scipy.interpolate import RectBivariateSpline
from scipy.special import gamma
import lightweaver.constants as Const
import os

DeltaNeff = 0.1

class BarklemTable:
    def __init__(self, path: str, neff0: Tuple[float, float]):
        data = np.genfromtxt(path, comments='c')
        shape = data.shape
        self.cross = data[:shape[0]//2]
        self.alpha = data[shape[0]//2:]

        self.neff1 = neff0[0] + np.arange(shape[0]//2) * DeltaNeff
        self.neff2 = neff0[1] + np.arange(shape[1]) * DeltaNeff

class BarklemCrossSectionError(Exception):
    pass

class Barklem:
    def __init__(self, table: AtomicTable):
        # Should probably make these static class variables, but also they're tiny
        # TODO(cmo): Move this to be ResourceReader/Loader based?
        self.barklem_sp = BarklemTable(get_data_path() + 'Barklem_spdata.dat', (1.0, 1.3))
        self.barklem_pd = BarklemTable(get_data_path() + 'Barklem_pddata.dat', (1.3, 2.3))
        self.barklem_df = BarklemTable(get_data_path() + 'Barklem_dfdata.dat', (2.3, 3.3))

        self.atomicTable = table

    def get_active_cross_section(self, atom: 'AtomicModel', line: 'AtomicLine') -> Sequence[float]:
        i = line.i
        j = line.j

        SOrbit = 0
        POrbit = 1
        DOrbit = 2
        FOrbit = 3

        # Follows original RH version. Interpolate tables if sigma < 20.0, otherwise
        # assume the provided values are the coefficients
        if line.vdw.vals[0] < 20.0:
            if atom.levels[i].stage > 0:
                raise BarklemCrossSectionError()

            # Find principal quantum numbers
            # try:
            #     lowerNum = determinate(atom.levels[i])
            #     upperNum = determinate(atom.levels[j])
            # except CompositeLevelError:
            #     raise BarklemCrossSectionError()
            lowerNum = atom.levels[i].L
            upperNum = atom.levels[j].L
            if lowerNum is None or upperNum is None:
                raise BarklemCrossSectionError()

            nums = (lowerNum, upperNum)

            # Check is a Barklem case applies
            if nums == (SOrbit, POrbit) or nums == (POrbit, SOrbit):
                table = self.barklem_sp
            elif nums == (POrbit, DOrbit) or nums == (DOrbit, POrbit):
                table = self.barklem_pd
            elif nums == (DOrbit, FOrbit) or nums == (FOrbit, DOrbit):
                table = self.barklem_df
            else:
                raise BarklemCrossSectionError()

            Z = atom.levels[j].stage + 1
            # Find index of continuum level
            ic = j + 1
            while atom.levels[ic].stage < atom.levels[j].stage + 1:
                ic += 1

            # atom->E[i] *= (HPLANCK * CLIGHT) / CM_TO_M;

            deltaEi = (atom.levels[ic].E - atom.levels[i].E) * Const.HC / Const.CM_TO_M
            deltaEj = (atom.levels[ic].E - atom.levels[j].E) * Const.HC / Const.CM_TO_M
            E_Rydberg = Const.ERydberg / (1.0 + Const.MElectron / (self.atomicTable[atom.name].weight * Const.Amu))

            neff1 = Z * np.sqrt(E_Rydberg / deltaEi)
            neff2 = Z * np.sqrt(E_Rydberg / deltaEj)

            if nums[0] > nums[1]:
                neff1, neff2 = neff2, neff1

            if not (table.neff1[0] <= neff1 <= table.neff1[-1]):
                raise BarklemCrossSectionError()
            if not (table.neff2[0] <= neff2 <= table.neff2[-1]):
                raise BarklemCrossSectionError()


            vals = np.zeros(4)
            vals[0] = RectBivariateSpline(table.neff1, table.neff2, table.cross)(neff1, neff2)
            vals[1] = RectBivariateSpline(table.neff1, table.neff2, table.alpha)(neff1, neff2)

        reducedMass = Const.Amu / (1.0 / self.atomicTable['H'].weight + 1.0 / self.atomicTable[atom.name].weight)
        meanVel = np.sqrt(8.0 * Const.KBoltzmann / (np.pi * reducedMass))
        meanCross = Const.RBohr**2 * (meanVel / 1.0e4)**(-vals[1])

        vals[0] *= 2.0 * (4.0 / np.pi)**(vals[1]/2.0) * gamma(4.0 - vals[1] / 2.0) * meanVel * meanCross

        # Use Unsold for Helium contribution
        vals[2] = 1.0
        # vals[3] = 0.0

        return vals


    
