from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence, TYPE_CHECKING, Optional
import numpy as np
from witt import witt
import Constants as Const
from scipy.interpolate import interp1d
from numpy.polynomial.legendre import leggauss
from Utils import ConvergenceError

class ScaleType(Enum):
    Geometric = 0
    ColumnMass = auto()
    Tau500 = auto()

class BoundaryCondition(Enum):
    Zero = auto()
    Thermalised = auto()

def get_top_pressure(eos : witt, temp, ne=None, rho=None):
    if ne is not None:
        pe = ne * Const.CM_TO_M**3 * eos.BK * temp
        return eos.pg_from_pe(temp, pe)
    elif rho is not None:
        return eos.pg_from_rho(temp, rho)
    
    pgasCgs = np.array([0.70575286, 0.59018545, 0.51286639, 0.43719268, 0.37731009,
        0.33516886, 0.31342915, 0.30604891, 0.30059491, 0.29207645,
        0.2859011 , 0.28119224, 0.27893046, 0.27949676, 0.28299726,
        0.28644693, 0.28825946, 0.29061192, 0.29340255, 0.29563072,
        0.29864548, 0.30776456, 0.31825915, 0.32137574, 0.3239401 ,
        0.32622212, 0.32792196, 0.3292243 , 0.33025437, 0.33146736,
        0.3319676 , 0.33217821, 0.3322355 , 0.33217166, 0.33210297,
        0.33203833, 0.33198508])
    tempCoord = np.array([  7600.,   7780.,   7970.,   8273.,   8635.,   8988.,   9228.,
        9358.,   9458.,   9587.,   9735.,   9983.,  10340.,  10850.,
        11440.,  12190.,  13080.,  14520.,  16280.,  17930.,  20420.,
        24060.,  27970.,  32150.,  36590.,  41180.,  45420.,  49390.,
        53280.,  60170.,  66150.,  71340.,  75930.,  83890.,  90820.,
        95600., 100000.])

    ptop = interp1d(tempCoord, pgasCgs, bounds_error=False, fill_value=(pgasCgs[0], pgasCgs[-1]))(temp)
    return ptop
    
@dataclass
class Atmosphere:
    scale: ScaleType
    depthScale: np.ndarray
    temperature: np.ndarray
    vlos: np.ndarray
    vturb: np.ndarray
    ne: Optional[np.ndarray] = None
    hydrogenPops: Optional[np.ndarray] = None
    B: Optional[np.ndarray] = None
    gammaB: Optional[np.ndarray] = None
    chiB: Optional[np.ndarray] = None
    nHTot: Optional[np.ndarray] = None
    Nspace: int = field(init=False)
    Nrays: int = field(init=False)
    lowerBc: BoundaryCondition = field(default=BoundaryCondition.Thermalised)
    upperBc: BoundaryCondition = field(default=BoundaryCondition.Zero)

    def __post_init__(self):
        self.Nspace = self.depthScale.shape[0]
        if self.hydrogenPops is not None:
            self.nHTot = np.sum(self.hydrogenPops, axis=0)

    def convert_scales(self, atomicTable, logG=2.44, Ptop=None, PeTop=None):
        if np.any(self.temperature < 2500):
            raise ValueError('Extremely low temperature (< 2500 K)')

        eos = witt()

        if self.nHTot is None and self.ne is not None:
            pe = self.ne * Const.CM_TO_M**3 * eos.BK * self.temperature
            rho = np.zeros(self.Nspace)
            for k in range(self.Nspace):
                rho[k] = eos.rho_from_pe(self.temperature[k], pe[k])
            self.nHTot = np.copy(rho / (Const.CM_TO_M**3 / Const.G_TO_KG) / (Const.AMU * atomicTable.weightPerH))
        elif self.ne is None and self.nHTot is not None:
            rho = Const.AMU * atomicTable.weightPerH * self.nHTot * Const.CM_TO_M**3 / Const.G_TO_KG
            pe = np.zeros(self.Nspace)
            for k in range(self.Nspace):
                pe[k] = eos.pe_from_rho(self.temperature[k], rho[k])
            self.ne = np.copy(pe / (eos.BK * self.temperature) / Const.CM_TO_M**3)
        elif self.ne is None and self.nHTot is None:
            # Have to do Hydrostatic Eq. based here on NICOLE implementation
            gravAcc = 10**logG / Const.CM_TO_M
            Avog = 6.022045e23 # Avogadro's Number
            if Ptop is None and PeTop is not None:
                PeTop *= (Const.CM_TO_M**3 / Const.G_TO_KG)
                Ptop = eos.pg_from_pe(self.temperature[0], PeTop)
            elif Ptop is not None and PeTop is None:
                Ptop *= (Const.CM_TO_M**3 / Const.G_TO_KG)
                PeTop = eos.pe_from_pg(self.temperature[0], Ptop)
            elif Ptop is None and PeTop is None:
                Ptop = get_top_pressure(eos, self.temperature[0])
                PeTop = eos.pe_from_pg(self.temperature[0], Ptop)
            else:
                raise ValueError("Cannot set both Ptop and PeTop")

            if self.scale == ScaleType.Tau500:
                tau = self.depthScale
            elif self.scale == ScaleType.Geometric:
                height = self.depthScale / Const.CM_TO_M
            else:
                cmass = self.depthScale / Const.G_TO_KG * Const.CM_TO_M**2

            rho = np.zeros(self.Nspace)
            chi_c = np.zeros(self.Nspace)
            pgas = np.zeros(self.Nspace)
            pe = np.zeros(self.Nspace)
            pgas[0] = Ptop
            pe[0] = PeTop
            chi_c[0] = eos.contOpacity(self.temperature[0], pgas[0], pe[0], np.array([5000.0]))
            avg_mol_weight = lambda k: atomicTable.weightPerH / (atomicTable.totalAbundance + pe[k] / pgas[k])
            rho[0] = Ptop * avg_mol_weight(0) / Avog / eos.BK / self.temperature[0]
            chi_c[0] /= rho[0]

            for k in range(1, self.Nspace):
                chi_c[k] = chi_c[k-1]
                rho[k] = rho[k-1]
                for it in range(200):
                    if self.scale == ScaleType.Tau500:
                        dtau = tau[k] - tau[k-1]
                        pgas[k] = pgas[k-1] + gravAcc * dtau / (0.5 * (chi_c[k-1] + chi_c[k]))
                    elif self.scale == ScaleType.Geometric:
                        pgas[k] = pgas[k-1] * np.exp(-gravAcc / Avog / eos.BK * avg_mol_weight(k-1) * 0.5 * (1.0 / self.temperature[k-1] + 1.0 / self.temperature[k]) * (height[k] - height[k-1]))
                    else:
                        pgas[k] = gravAcc * cmass[k]

                    pe[k] = eos.pe_from_pg(self.temperature[k], pgas[k])
                    prevChi = chi_c[k]
                    chi_c[k] = eos.contOpacity(self.temperature[k], pgas[k], pe[k], np.array([5000.0]))
                    rho[k] = pgas[k] * avg_mol_weight(k) / Avog / eos.BK / self.temperature[k]
                    chi_c[k] /= rho[k]

                    change = np.abs(prevChi - chi_c[k]) / (prevChi + chi_c[k])
                    print(change)
                    if change < 1e-5:
                        break
                else:
                    raise ConvergenceError('No convergence in HSE at depth point %d, last change %2.4e' % (k, change))

            # Filled in rho, pgas, and pe, based on EOS
            # Need to fill in ne and nHTot -- based on RH method
            self.ne = np.copy(pe / (eos.BK * self.temperature) / Const.CM_TO_M**3) 
            turbPe = 0.5 * Const.M_ELECTRON * self.vturb**2
            turbPg = 0.5 * atomicTable.avgMolWeight * Const.AMU * self.vturb**2
            self.nHTot = ((pgas - pe) / (eos.BK * self.temperature) / Const.CM_TO_M**3 - self.ne * turbPe) / (atomicTable.totalAbundance * (1.0 + turbPg / (Const.KBOLTZMANN * self.temperature)))
            if np.any(self.ne < 0) or np.any(self.nHTot < 0):
                raise ConvergenceError('HSE iterations produced negative density')


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

        # self.pgas = pgas
        # self.pe = pe

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
        elif self.scale == ScaleType.Geometric:
            cmass = np.zeros(self.Nspace)
            tau_ref = np.zeros(self.Nspace)
            height = self.depthScale

            cmass[0] = (self.nHTot[0] * atomicTable.weightPerH + self.ne[0]) * (Const.KBOLTZMANN * self.temperature[0] / 10**logG)
            tau_ref[0] = 0.5 * chi_c[0] * (height[0] - height[1])
            if tau_ref[0] > 1.0:
                tau_ref[0] = 0.0

            for k in range(1, self.Nspace):
                cmass[k] = cmass[k-1] + 0.5 * (rhoSI[k-1] + rhoSI[k]) * (height[k-1] - height[k])
                tau_ref[k] = tau_ref[k-1] + 0.5 * (chi_c[k-1] + chi_c[k]) * (height[k-1] - height[k])
            self.cmass = cmass
            self.height = height
            self.tau_ref = tau_ref
        elif self.scale == ScaleType.Tau500:
            cmass = np.zeros(self.Nspace)
            height = np.zeros(self.Nspace)
            tau_ref = self.depthScale

            cmass[0] = (tau_ref[0] / chi_c[0]) * rhoSI[0]
            for k in range(1, self.Nspace):
                height[k] = height[k-1] - 2.0 * (tau_ref[k] - tau_ref[k-1]) / (chi_c[k-1] + chi_c[k])
                cmass[k] = cmass[k-1] + 0.5 * (chi_c[k-1] + chi_c[k]) * (height[k-1] - height[k])

            print(tau_ref, height)
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




