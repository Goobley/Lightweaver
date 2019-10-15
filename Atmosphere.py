from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence, TYPE_CHECKING, Optional
import numpy as np
from witt import witt
import Constants as Const
from scipy.interpolate import interp1d
from numpy.polynomial.legendre import leggauss

class ScaleType(Enum):
    Geometric = 0
    ColumnMass = auto()
    Tau5000 = auto()

class BoundaryCondition(Enum):
    Zero = auto()
    Thermalised = auto()

@dataclass
class Atmosphere:
    scale: ScaleType
    depthScale: np.ndarray
    temperature: np.ndarray
    ne: np.ndarray
    vlos: np.ndarray
    vturb: np.ndarray
    hydrogenPops: np.ndarray
    Nspace: int = field(init=False)
    Nrays: int = field(init=False)
    lowerBc: BoundaryCondition = field(default=BoundaryCondition.Thermalised)
    upperBc: BoundaryCondition = field(default=BoundaryCondition.Zero)

    def __post_init__(self):
        self.Nspace = self.depthScale.shape[0]
        self.nHTot = np.sum(self.hydrogenPops, axis=0)
        self.B = None

    def convert_scales(self, atomicTable):
        # This is only temporary
        eos = witt()
        rhoSI = Const.AMU * atomicTable.weightPerH * self.nHTot
        rho = Const.AMU * atomicTable.weightPerH * self.nHTot * Const.CM_TO_M**3 / Const.G_TO_KG
        pgas = np.zeros_like(self.depthScale)
        pe = np.zeros_like(self.depthScale)
        for k in range(self.depthScale.shape[0]):
            pgas[k] = eos.pg_from_rho(self.temperature[k], rho[k])
            pe[k] = eos.pe_from_rho(self.temperature[k], rho[k])

        chi_c = np.zeros_like(self.depthScale)
        for k in range(self.depthScale.shape[0]):
            chi_c[k] = eos.contOpacity(self.temperature[k], pgas[k], pe[k], np.array([5000.0])) / Const.CM_TO_M

        self.pgas = pgas
        self.pe = pe

        if self.scale == ScaleType.ColumnMass:
            height = np.zeros_like(self.depthScale)
            tau_ref = np.zeros_like(self.depthScale)
            cmass = self.depthScale

            height[0] = 0.0
            tau_ref[0] = chi_c[0] / rhoSI[0] * cmass[0]
            for k in range(1, cmass.shape[0]):
                height[k] = height[k-1] - 2.0 * (cmass[k] - cmass[k-1]) / (rhoSI[k-1] + rhoSI[k])
                tau_ref[k] = tau_ref[k-1] + 0.5 * (chi_c[k-1] + chi_c[k]) * (height[k-1] - height[k])

            hTau1 = interp1d(tau_ref, height)(1.0)
            height -= hTau1

            self.cmass = cmass
            self.height = height
            self.tau_ref = tau_ref
        else:
            raise ValueError("Other scales not handled yet")

    def quadrature(self, Nrays: int, mu: Optional[float]=None):

        if Nrays > 1:        
            self.Nrays = Nrays
            x, w = leggauss(Nrays)
            mid, halfWidth = 0.5, 0.5
            x = mid + halfWidth * x
            w *= halfWidth

            self.muz = x
            self.wmu = w
        elif Nrays == 1:
            if mu is None:
                raise ValueError('For Nrays=1, mu must be provided')
            
            self.Nrays = 1
            self.muz = np.array([mu])
            self.wmu = np.array([1.0])
        else:
            raise ValueError('Unsupported Nrays=%d' % Nrays)

        self.muy = np.zeros_like(x)
        self.mux = np.sqrt(1.0 - x**2)




