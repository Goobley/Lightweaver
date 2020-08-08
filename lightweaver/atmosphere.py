from dataclasses import dataclass, field, asdict
from copy import copy
from enum import Enum, auto
from typing import Sequence, TYPE_CHECKING, Optional, Union, TypeVar, Type
import numpy as np
from .witt import witt
import lightweaver.constants as Const
from numpy.polynomial.legendre import leggauss
from .utils import ConvergenceError, view_flatten, get_data_path
from .atomic_table import PeriodicTable, AtomicAbundance, DefaultAtomicAbundance
import astropy.units as u
import pickle

if TYPE_CHECKING:
    from .LwCompiled import LwSpectrum

class ScaleType(Enum):
    Geometric = 0
    ColumnMass = auto()
    Tau500 = auto()

class BoundaryCondition:
    def compute_bc(self, atmos: 'Atmosphere', spect: 'LwSpectrum') -> np.ndarray:
        raise NotImplementedError

class NoBc(BoundaryCondition):
    pass

class ZeroRadiation(BoundaryCondition):
    pass

class ThermalisedRadiation(BoundaryCondition):
    pass

class PeriodicRadiation(BoundaryCondition):
    pass

def get_top_pressure(eos: witt, temp, ne=None, rho=None):
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

    ptop = np.interp(temp, tempCoord, pgasCgs)
    return ptop

@dataclass
class Stratifications:
    cmass: np.ndarray
    tauRef: np.ndarray

    def dimensioned_view(self, shape) -> 'Stratifications':
        strat = copy(self)
        strat.cmass = self.cmass.reshape(shape)
        strat.tauRef = self.tauRef.reshape(shape)
        return strat

    def unit_view(self) -> 'Stratifications':
        strat = copy(self)
        strat.cmass = self.cmass << u.kg / u.m**2
        strat.tauRef = self.tauRef << u.dimensionless_unscaled
        return strat

    def dimensioned_unit_view(self, shape) -> 'Stratifications':
        strat = self.dimensioned_view(shape)
        return strat.unit_view()

@dataclass
class Layout:
    Ndim: int
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    xLowerBc: BoundaryCondition
    xUpperBc: BoundaryCondition
    yLowerBc: BoundaryCondition
    yUpperBc: BoundaryCondition
    zLowerBc: BoundaryCondition
    zUpperBc: BoundaryCondition
    stratifications: Optional[Stratifications] = None

    @classmethod
    def make_1d(cls, z: np.ndarray, vz: np.ndarray,
                lowerBc: BoundaryCondition, upperBc: BoundaryCondition,
                stratifications: Optional[Stratifications]=None) -> 'Layout':

        return cls(Ndim=1, x=np.array(()), y=np.array(()),
                   z=z, vx=np.array(()), vy=np.array(()),
                   vz=vz, xLowerBc=NoBc(), xUpperBc=NoBc(),
                   yLowerBc=NoBc(), yUpperBc=NoBc(),
                   zLowerBc=lowerBc, zUpperBc=upperBc,
                   stratifications=stratifications)

    @classmethod
    def make_2d(cls, x: np.ndarray, z: np.ndarray,
                vx: np.ndarray, vz: np.ndarray,
                xLowerBc: BoundaryCondition, xUpperBc: BoundaryCondition,
                zLowerBc: BoundaryCondition, zUpperBc: BoundaryCondition,
                stratifications: Optional[Stratifications]=None) -> 'Layout':

        Bc = BoundaryCondition
        return cls(Ndim=2, x=x, y=np.array(()), z=z,
                   vx=vx, vy=np.array(()), vz=vz,
                   xLowerBc=xLowerBc, xUpperBc=xUpperBc,
                   yLowerBc=NoBc(), yUpperBc=NoBc(),
                   zLowerBc=zLowerBc, zUpperBc=zUpperBc,
                   stratifications=stratifications)

    @classmethod
    def make_3d(cls, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                xLowerBc: BoundaryCondition, xUpperBc: BoundaryCondition,
                yLowerBc: BoundaryCondition, yUpperBc: BoundaryCondition,
                zLowerBc: BoundaryCondition, zUpperBc: BoundaryCondition,
                stratifications: Optional[Stratifications]=None) -> 'Layout':

        return cls(Ndim=3, x=x, y=y, z=z,
                   vx=vx, vy=vy, vz=vz,
                   xLowerBc=xLowerBc, xUpperBc=xUpperBc,
                   yLowerBc=yLowerBc, yUpperBc=yUpperBc,
                   zLowerBc=zLowerBc, zUpperBc=zUpperBc,
                   stratifications=stratifications)

    @property
    def Nx(self) -> int:
        return self.x.shape[0]

    @property
    def Ny(self) -> int:
        return self.y.shape[0]

    @property
    def Nz(self) -> int:
        return self.z.shape[0]

    @property
    def Noutgoing(self) -> int:
        return max(1, self.Nx, self.Nx * self.Ny)

    @property
    def vlos(self) -> np.ndarray:
        if self.Ndim > 1:
            raise ValueError('vlos is ambiguous when Ndim > 1, use vx, vy, or vz instead.')
        return self.vz

    @property
    def Nspace(self) -> int:
        if self.Ndim == 1:
            return self.Nz
        elif self.Ndim == 2:
            return self.Nx * self.Nz
        elif self.Ndim == 3:
            return self.Nx * self.Ny * self.Nz
        else:
            raise ValueError('Invalid Ndim: %d, check geometry initialisation' % self.Ndim)

    @property
    def tauRef(self):
        if self.stratifications is not None:
            return self.stratifications.tauRef
        else:
            raise ValueError('tauRef not computed for this Atmosphere')

    @property
    def cmass(self):
        if self.stratifications is not None:
            return self.stratifications.cmass
        else:
            raise ValueError('tauRef not computed for this Atmosphere')

    @property
    def dimensioned_shape(self):
        if self.Ndim == 1:
            shape = (self.Nz,)
        elif self.Ndim == 2:
            shape = (self.Nz, self.Nx)
        elif self.Ndim == 3:
            shape = (self.Nz, self.Ny, self.Nx)
        else:
            raise ValueError('Unreasonable Ndim (%d)' % self.Ndim)
        return shape

    def dimensioned_view(self) ->  'Layout':
        layout = copy(self)
        shape = self.dimensioned_shape
        if self.stratifications is not None:
            layout.stratifications = self.stratifications.dimensioned_view(shape)
        if self.vx.size > 0:
            layout.vx = self.vx.reshape(shape)
        if self.vy.size > 0:
            layout.vy = self.vy.reshape(shape)
        if self.vz.size > 0:
            layout.vz = self.vz.reshape(shape)
        return layout

    def unit_view(self) -> 'Layout':
        layout = copy(self)
        layout.x = self.x << u.m
        layout.y = self.y << u.m
        layout.z = self.z << u.m
        layout.vx = self.vx << u.m / u.s
        layout.vy = self.vy << u.m / u.s
        layout.vz = self.vz << u.m / u.s
        if self.stratifications is not None:
            layout.stratifications = self.stratifications.unit_view()
        return layout

    def dimensioned_unit_view(self) -> 'Layout':
        layout = self.dimensioned_view()
        return layout.unit_view()

@dataclass
class Atmosphere:
    structure: Layout
    temperature: np.ndarray
    vturb: np.ndarray
    ne: np.ndarray
    nHTot: np.ndarray
    B: Optional[np.ndarray] = None
    gammaB: Optional[np.ndarray] = None
    chiB: Optional[np.ndarray] = None

    @property
    def Ndim(self) -> int:
        return self.structure.Ndim

    @property
    def Nx(self) -> int:
        return self.structure.Nx

    @property
    def Ny(self) -> int:
        return self.structure.Ny

    @property
    def Nz(self) -> int:
        return self.structure.Nz

    @property
    def Noutgoing(self) -> int:
        return self.structure.Noutgoing

    @property
    def vx(self) -> np.ndarray:
        return self.structure.vx

    @property
    def vy(self) -> np.ndarray:
        return self.structure.vz

    @property
    def vz(self) -> np.ndarray:
        return self.structure.vz

    @property
    def vlos(self) -> np.ndarray:
        return self.structure.vlos

    @property
    def cmass(self) -> np.ndarray:
        return self.structure.cmass

    @property
    def tauRef(self) -> np.ndarray:
        return self.structure.tauRef

    @property
    def height(self) -> np.ndarray:
        return self.structure.z

    @property
    def x(self) -> np.ndarray:
        return self.structure.x

    @property
    def y(self) -> np.ndarray:
        return self.structure.y

    @property
    def z(self) -> np.ndarray:
        return self.structure.z

    @property
    def Nspace(self):
        return self.structure.Nspace

    @property
    def Nrays(self):
        if self.muz is None:
            raise AttributeError('Nrays not set, call atmos.rays or .quadrature first')

        return self.muz.shape[0]

    def dimensioned_view(self):
        shape = self.structure.dimensioned_shape
        atmos = copy(self)
        atmos.structure = self.structure.dimensioned_view()
        atmos.temperature = self.temperature.reshape(shape)
        atmos.vturb = self.vturb.reshape(shape)
        atmos.ne = self.ne.reshape(shape)
        atmos.nHTot = self.nHTot.reshape(shape)
        if self.B is not None:
            atmos.B = self.B.reshape(shape)
            atmos.chiB = self.chiB.reshape(shape)
            atmos.gammaB = self.gammaB.reshape(shape)
        return atmos

    def unit_view(self):
        atmos = copy(self)
        atmos.structure = self.structure.unit_view()
        atmos.temperature = self.temperature << u.K
        atmos.vturb = self.vturb << u.m / u.s
        atmos.ne = self.ne << u.m**(-3)
        atmos.nHTot = self.nHTot << u.m**(-3)
        if self.B is not None:
            atmos.B = self.B << u.T
            atmos.chiB = self.chiB << u.rad
            atmos.gammaB = self.gammaB << u.rad
        return atmos

    def dimensioned_unit_view(self):
        atmos = self.dimensioned_view()
        return atmos.unit_view()

    @classmethod
    def make_1d(cls, scale: ScaleType, depthScale: np.ndarray,
                temperature: np.ndarray, vlos: np.ndarray,
                vturb: np.ndarray, ne: Optional[np.ndarray]=None,
                hydrogenPops: Optional[np.ndarray]=None,
                nHTot: Optional[np.ndarray]=None,
                B: Optional[np.ndarray]=None,
                gammaB: Optional[np.ndarray]=None,
                chiB: Optional[np.ndarray]=None,
                lowerBc: Optional[BoundaryCondition]=None,
                upperBc: Optional[BoundaryCondition]=None,
                convertScales: bool=True,
                abundance: Optional[AtomicAbundance]=None,
                logG: float=2.44,
                Pgas: Optional[np.ndarray]=None,
                Pe: Optional[np.ndarray]=None,
                Ptop: Optional[float]=None,
                PeTop: Optional[float]=None,
                verbose: bool=False):
        if scale == ScaleType.Geometric:
            depthScale = (depthScale << u.m).value
        elif scale == ScaleType.ColumnMass:
            depthScale = (depthScale << u.kg / u.m**2).value
        temperature = (temperature << u.K).value
        vlos = (vlos << u.m / u.s).value
        vturb = (vturb << u.m / u.s).value
        if ne is not None:
            ne = (ne << u.m**(-3)).value
        if hydrogenPops is not None:
            hydrogenPops = (hydrogenPops << u.m**(-3)).value
        if nHTot is not None:
            nHTot = (nHTot << u.m(-3)).value
        if B is not None:
            B = (B << u.T).value
        if gammaB is not None:
            gammaB = (gammaB << u.rad).value
        if chiB is not None:
            chiB = (chiB << u.rad).value

        if lowerBc is None:
            lowerBc = ThermalisedRadiation()
        elif isinstance(lowerBc, PeriodicRadiation):
            raise ValueError('Cannot set periodic boundary conditions for 1D atmosphere')
        if upperBc is None:
            upperBc = ZeroRadiation()
        elif isinstance(upperBc, PeriodicRadiation):
            raise ValueError('Cannot set periodic boundary conditions for 1D atmosphere')

        if scale != ScaleType.Geometric and not convertScales:
            raise ValueError('Height scale must be provided if scale conversion is not applied')

        if nHTot is None and hydrogenPops is not None:
            nHTot = np.sum(hydrogenPops, axis=0)

        if np.any(temperature < 2000):
            # NOTE(cmo): Minimum value was decreased in NICOLE so should be safe
            raise ValueError('Minimum temperature too low for EOS (< 2000 K)')

        if abundance is None:
            abundance = DefaultAtomicAbundance

        wittAbundances = np.array([abundance[e] for e in PeriodicTable.elements])
        eos = witt(abund_init=wittAbundances)

        Nspace = depthScale.shape[0]
        if nHTot is None and ne is not None:
            if verbose:
                print('Setting nHTot from electron pressure.')
            pe = ne * Const.CM_TO_M**3 * eos.BK * temperature
            rho = np.zeros(Nspace)
            for k in range(Nspace):
                rho[k] = eos.rho_from_pe(temperature[k], pe[k])
            nHTot = np.copy(rho / (Const.CM_TO_M**3 / Const.G_TO_KG) / (Const.Amu * abundance.massPerH))
        elif ne is None and nHTot is not None:
            if verbose:
                print('Setting ne from mass density.')
            rho = Const.Amu * abundance.massPerH * nHTot * Const.CM_TO_M**3 / Const.G_TO_KG
            pe = np.zeros(Nspace)
            for k in range(Nspace):
                pe[k] = eos.pe_from_rho(temperature[k], rho[k])
            ne = np.copy(pe / (eos.BK * temperature) / Const.CM_TO_M**3)
        elif ne is None and nHTot is None:
            if Pgas is not None and Pgas.shape[0] != Nspace:
                raise ValueError('Dimensions of Pgas do not match atmospheric depth')
            if Pe is not None and Pe.shape[0] != Nspace:
                raise ValueError('Dimensions of Pe do not match atmospheric depth')

            if Pgas is not None and Pe is None:
                if verbose:
                    print('Setting ne, nHTot from provided gas pressure.')
                # Convert to cgs for eos
                pgas = Pgas * (Const.CM_TO_M**2 / Const.G_TO_KG)
                pe = np.zeros(Nspace)
                for k in range(Nspace):
                    pe[k] = eos.pe_from_pg(temperature[k], pgas[k])
            elif Pe is not None and Pgas is None:
                if verbose:
                    print('Setting ne, nHTot from provided electron pressure.')
                # Convert to cgs for eos
                pe = Pe * (Const.CM_TO_M**2 / Const.G_TO_KG)
                pgas = np.zeros(Nspace)
                for k in range(Nspace):
                    pgas[k] = eos.pg_from_pe(temperature[k], pe[k])
            elif Pgas is None and Pe is None:
                # Doing Hydrostatic Eq. based here on NICOLE implementation
                gravAcc = 10**logG / Const.CM_TO_M
                Avog = 6.022045e23 # Avogadro's Number
                if Ptop is None and PeTop is not None:
                    if verbose:
                        print('Setting ne, nHTot to hydrostatic equilibrium (logG=%f) from provided top electron pressure.' % logG)
                    PeTop *= (Const.CM_TO_M**2 / Const.G_TO_KG)
                    Ptop = eos.pg_from_pe(temperature[0], PeTop)
                elif Ptop is not None and PeTop is None:
                    if verbose:
                        print('Setting ne, nHTot to hydrostatic equilibrium (logG=%f) from provided top gas pressure.' % logG)
                    Ptop *= (Const.CM_TO_M**2 / Const.G_TO_KG)
                    PeTop = eos.pe_from_pg(temperature[0], Ptop)
                elif Ptop is None and PeTop is None:
                    if verbose:
                        print('Setting ne, nHTot to hydrostatic equilibrium (logG=%f) from FALC gas pressure at upper boundary temperature.' % logG)
                    Ptop = get_top_pressure(eos, temperature[0])
                    PeTop = eos.pe_from_pg(temperature[0], Ptop)
                else:
                    raise ValueError("Cannot set both Ptop and PeTop")

                if scale == ScaleType.Tau500:
                    tau = depthScale
                elif scale == ScaleType.Geometric:
                    height = depthScale / Const.CM_TO_M
                else:
                    cmass = depthScale / Const.G_TO_KG * Const.CM_TO_M**2

                # NOTE(cmo): Compute HSE following the NICOLE method.
                rho = np.zeros(Nspace)
                chi_c = np.zeros(Nspace)
                pgas = np.zeros(Nspace)
                pe = np.zeros(Nspace)
                pgas[0] = Ptop
                pe[0] = PeTop
                chi_c[0] = eos.contOpacity(temperature[0], pgas[0], pe[0], np.array([5000.0]))
                avg_mol_weight = lambda k: abundance.massPerH / (abundance.totalAbundance + pe[k] / pgas[k])
                rho[0] = Ptop * avg_mol_weight(0) / Avog / eos.BK / temperature[0]
                chi_c[0] /= rho[0]

                for k in range(1, Nspace):
                    chi_c[k] = chi_c[k-1]
                    rho[k] = rho[k-1]
                    for it in range(200):
                        if scale == ScaleType.Tau500:
                            dtau = tau[k] - tau[k-1]
                            pgas[k] = pgas[k-1] + gravAcc * dtau / (0.5 * (chi_c[k-1] + chi_c[k]))
                        elif scale == ScaleType.Geometric:
                            pgas[k] = pgas[k-1] * np.exp(-gravAcc / Avog / eos.BK * avg_mol_weight(k-1) * 0.5 * (1.0 / temperature[k-1] + 1.0 / temperature[k]) * (height[k] - height[k-1]))
                        else:
                            pgas[k] = gravAcc * cmass[k]

                        pe[k] = eos.pe_from_pg(temperature[k], pgas[k])
                        prevChi = chi_c[k]
                        chi_c[k] = eos.contOpacity(temperature[k], pgas[k], pe[k], np.array([5000.0]))
                        rho[k] = pgas[k] * avg_mol_weight(k) / Avog / eos.BK / temperature[k]
                        chi_c[k] /= rho[k]

                        change = np.abs(prevChi - chi_c[k]) / (prevChi + chi_c[k])
                        if change < 1e-5:
                            break
                    else:
                        raise ConvergenceError('No convergence in HSE at depth point %d, last change %2.4e' % (k, change))

        # NOTE(cmo): Compute final pgas, pe from EOS that will be used for
        # background opacity.
        rhoSI = Const.Amu * abundance.massPerH * nHTot
        rho = Const.Amu * abundance.massPerH * nHTot * Const.CM_TO_M**3 / Const.G_TO_KG
        pgas = np.zeros_like(depthScale)
        pe = np.zeros_like(depthScale)
        for k in range(Nspace):
            pgas[k] = eos.pg_from_rho(temperature[k], rho[k])
            pe[k] = eos.pe_from_rho(temperature[k], rho[k])

        chi_c = np.zeros_like(depthScale)
        for k in range(depthScale.shape[0]):
            chi_c[k] = eos.contOpacity(temperature[k], pgas[k], pe[k], np.array([5000.0])) / Const.CM_TO_M

        # NOTE(cmo): We should now have a uniform minimum set of data (other
        # than the scale type), allowing us to simply convert between the
        # scales we do have!
        if convertScales:
            if scale == ScaleType.ColumnMass:
                height = np.zeros_like(depthScale)
                tau_ref = np.zeros_like(depthScale)
                cmass = depthScale

                height[0] = 0.0
                tau_ref[0] = chi_c[0] / rhoSI[0] * cmass[0]
                for k in range(1, cmass.shape[0]):
                    height[k] = height[k-1] - 2.0 * (cmass[k] - cmass[k-1]) / (rhoSI[k-1] + rhoSI[k])
                    tau_ref[k] = tau_ref[k-1] + 0.5 * (chi_c[k-1] + chi_c[k]) * (height[k-1] - height[k])

                hTau1 = np.interp(1.0, tau_ref, height)
                height -= hTau1
            elif scale == ScaleType.Geometric:
                cmass = np.zeros(Nspace)
                tau_ref = np.zeros(Nspace)
                height = depthScale

                cmass[0] = (nHTot[0] * abundance.massPerH + ne[0]) * (Const.KBoltzmann * temperature[0] / 10**logG)
                tau_ref[0] = 0.5 * chi_c[0] * (height[0] - height[1])
                if tau_ref[0] > 1.0:
                    tau_ref[0] = 0.0

                for k in range(1, Nspace):
                    cmass[k] = cmass[k-1] + 0.5 * (rhoSI[k-1] + rhoSI[k]) * (height[k-1] - height[k])
                    tau_ref[k] = tau_ref[k-1] + 0.5 * (chi_c[k-1] + chi_c[k]) * (height[k-1] - height[k])
            elif scale == ScaleType.Tau500:
                cmass = np.zeros(Nspace)
                height = np.zeros(Nspace)
                tau_ref = depthScale

                cmass[0] = (tau_ref[0] / chi_c[0]) * rhoSI[0]
                for k in range(1, Nspace):
                    height[k] = height[k-1] - 2.0 * (tau_ref[k] - tau_ref[k-1]) / (chi_c[k-1] + chi_c[k])
                    cmass[k] = cmass[k-1] + 0.5 * (chi_c[k-1] + chi_c[k]) * (height[k-1] - height[k])

                hTau1 = np.interp(1.0, tau_ref, height)
                height -= hTau1
            else:
                raise ValueError("Other scales not handled yet")

            stratifications: Optional[Stratifications] = Stratifications(cmass=cmass, tauRef=tau_ref)

        else:
            stratifications = None

        layout = Layout.make_1d(z=height, vz=vlos,
                                lowerBc=lowerBc, upperBc=upperBc,
                                stratifications=stratifications)
        atmos = cls(structure=layout, temperature=temperature, vturb=vturb,
                    ne=ne, nHTot=nHTot, B=B, gammaB=gammaB, chiB=chiB)

        return atmos

    @classmethod
    def make_2d(cls, height: np.ndarray, x: np.ndarray,
                temperature: np.ndarray, vx: np.ndarray,
                vz: np.ndarray, vturb: np.ndarray,
                ne: Optional[np.ndarray]=None,
                nHTot: Optional[np.ndarray]=None,
                B: Optional[np.ndarray]=None,
                gammaB: Optional[np.ndarray]=None,
                chiB: Optional[np.ndarray]=None,
                xUpperBc: Optional[BoundaryCondition]=None,
                xLowerBc: Optional[BoundaryCondition]=None,
                zUpperBc: Optional[BoundaryCondition]=None,
                zLowerBc: Optional[BoundaryCondition]=None,
                abundance: Optional[AtomicAbundance]=None,
                verbose=False):

        x = (x << u.m).value
        height = (height << u.m).value
        temperature = (temperature << u.K).value
        vx = (vx << u.m / u.s).value
        vz = (vz << u.m / u.s).value
        vturb = (vturb << u.m / u.s).value
        if ne is not None:
            ne = (ne << u.m**(-3)).value
        if nHTot is not None:
            nHTot = (nHTot << u.m**(-3)).value
        if B is not None:
            B = (B << u.T).value
            flatB = view_flatten(B)
        else:
            flatB = None

        if gammaB is not None:
            gammaB = (gammaB << u.rad).value
            flatGammaB = view_flatten(gammaB)
        else:
            flatGammaB = None

        if chiB is not None:
            chiB = (chiB << u.rad).value
            flatChiB = view_flatten(chiB)
        else:
            flatChiB = None

        if zLowerBc is None:
            zLowerBc = ThermalisedRadiation()
        elif isinstance(zLowerBc, PeriodicRadiation):
            raise ValueError('Cannot set periodic boundary conditions for z-axis.')
        if zUpperBc is None:
            zUpperBc = ZeroRadiation()
        elif isinstance(zUpperBc, PeriodicRadiation):
            raise ValueError('Cannot set periodic boundary conditions for z-axis.')
        if xUpperBc is None:
            xUpperBc = PeriodicRadiation()
        if xLowerBc is None:
            xLowerBc = PeriodicRadiation()
        if abundance is None:

            abundance = DefaultAtomicAbundance

        wittAbundances = np.array([abundance[e] for e in PeriodicTable.elements])
        eos = witt(abund_init=wittAbundances)

        flatHeight = view_flatten(height)
        flatTemperature = view_flatten(temperature)
        Nspace = flatHeight.shape[0]
        if nHTot is None and ne is not None:
            if verbose:
                print('Setting nHTot from electron pressure.')
            flatNe = view_flatten(ne)
            pe = flatNe * Const.CM_TO_M**3 * eos.BK * flatTemperature
            rho = np.zeros(Nspace)
            for k in range(Nspace):
                rho[k] = eos.rho_from_pe(flatTemperature[k], pe[k])
            nHTot = np.copy(rho / (Const.CM_TO_M**3 / Const.G_TO_KG) / (Const.Amu * abundance.massPerH))
        elif ne is None and nHTot is not None:
            if verbose:
                print('Setting ne from mass density.')
            flatNHTot = view_flatten(nHTot)
            rho = Const.Amu * abundance.massPerH * flatNHTot * Const.CM_TO_M**3 / Const.G_TO_KG
            pe = np.zeros(Nspace)
            for k in range(Nspace):
                pe[k] = eos.pe_from_rho(flatTemperature[k], rho[k])
            ne = np.copy(pe / (eos.BK * flatTemperature) / Const.CM_TO_M**3)
        elif ne is None and nHTot is None:
            raise ValueError('Cannot omit both ne and nHTot (currently).')
        flatX = view_flatten(x)
        flatNHTot = view_flatten(nHTot)
        flatNe = view_flatten(ne)
        flatVx = view_flatten(vx)
        flatVz = view_flatten(vz)
        flatVturb = view_flatten(vturb)

        layout = Layout.make_2d(x=flatX, z=flatHeight, vx=flatVx, vz=flatVz,
                                xLowerBc=xLowerBc, xUpperBc=xUpperBc,
                                zLowerBc=zLowerBc, zUpperBc=zUpperBc,
                                stratifications=None)

        atmos = cls(structure=layout, temperature=flatTemperature,
                    vturb=flatVturb, ne=flatNe, nHTot=flatNHTot, B=flatB,
                    gammaB=flatGammaB, chiB=flatChiB)
        return atmos

    def quadrature(self, Nrays: Optional[int]=None,
                   mu: Optional[Sequence[float]]=None,
                   wmu: Optional[Sequence[float]]=None):

        if self.Ndim == 1:
            if Nrays is not None and mu is None:
                if Nrays >= 1:
                    x, w = leggauss(Nrays)
                    mid, halfWidth = 0.5, 0.5
                    x = mid + halfWidth * x
                    w *= halfWidth

                    self.muz = x
                    self.wmu = w
                else:
                    raise ValueError('Unsupported Nrays=%d' % Nrays)
            elif Nrays is not None and mu is not None:
                if wmu is None:
                    raise ValueError('Must provide wmu when providing mu')
                if Nrays != len(mu):
                    raise ValueError('mu must be Nrays long if Nrays is provided')
                if len(mu) != len(wmu):
                    raise ValueError('mu and wmu must be the same shape')

                self.muz = np.array(mu)
                self.wmu = np.array(wmu)

            self.muy = np.zeros_like(self.muz)
            self.mux = np.sqrt(1.0 - self.muz**2)
        else:
            with open(get_data_path() + 'Quadratures.pickle', 'rb') as pkl:
                quads = pickle.load(pkl)

            rays = {int(q.split('n')[1]): q for q in quads}
            if Nrays not in rays:
                raise ValueError('For multidimensional cases Nrays must be in %s' % repr(rays))

            quad = quads[rays[Nrays]]

            if self.Ndim == 2:
                Nrays *= 2
                theta = np.deg2rad(quad[:, 1])
                chi = np.deg2rad(quad[:, 2])
                # polar coords:
                # x = sin theta cos chi
                # y = sin theta sin chi
                # z = cos theta
                self.mux = np.zeros(Nrays)
                self.mux[:Nrays // 2] = np.sin(theta) * np.cos(chi)
                self.mux[Nrays // 2:] = -np.sin(theta) * np.cos(chi)
                self.muz = np.zeros(Nrays)
                self.muz[:Nrays // 2] = np.cos(theta)
                self.muz[Nrays // 2:] = np.cos(theta)
                self.wmu = np.zeros(Nrays)
                self.wmu[:Nrays // 2] = quad[:, 0]
                self.wmu[Nrays // 2:] = quad[:, 0]
                self.wmu /= np.sum(self.wmu)
                self.muy = np.sqrt(1.0 - (self.mux**2 + self.muz**2))

            else:
                raise NotImplementedError()



    def angle_set_a4(self):
        mux_A4 = [0.88191710, 0.33333333, 0.33333333]
        muy_A4 = [0.33333333, 0.88191710, 0.33333333]
        wmu_A4 = [0.33333333, 0.33333333, 0.33333333]

        mux_A8 = [0.95118973, 0.78679579, 0.57735027,
                  0.21821789, 0.78679579, 0.57735027,
                  0.21821789, 0.57735027, 0.21821789, 0.21821789]
        muy_A8 = [0.21821789, 0.57735027, 0.78679579,
                  0.95118973, 0.21821789, 0.57735027,
                  0.78679579, 0.21821789, 0.57735027, 0.21821789]
        wmu_A8 = [0.12698138, 0.09138353, 0.09138353,
                  0.12698138, 0.09138353, 0.07075469,
                  0.09138353, 0.09138353, 0.09138353, 0.12698138]

        mux_test = [1.0 / np.sqrt(3)]
        muy_test = [1.0 / np.sqrt(3)]
        wmu_test = [1.0]

        mux = mux_A8
        muy = muy_A8
        wmu = wmu_A8

        Nrays = len(mux) * 2;
        self.mux = np.zeros(Nrays)
        self.mux[:Nrays // 2] = mux
        self.muy = np.zeros(Nrays)
        self.muy[:Nrays // 2] = muy
        self.wmu = np.zeros(Nrays)
        wnorm = 0.5 / sum(wmu)
        self.wmu[:Nrays // 2] = wmu
        self.wmu[Nrays // 2:] = wmu
        self.wmu *= wnorm

        for mu in range(Nrays // 2):
            self.mux[Nrays // 2 + mu] = -self.mux[mu]
            self.muy[Nrays // 2 + mu] = self.muy[mu]

        self.muz = np.sqrt(1.0 - (self.mux**2 + self.muy**2))
        # self.wmu = np.concatenate((self.wmu, [1e-20]))
        # self.muz = np.concatenate((self.muz, [1.0]))
        # self.mux = np.concatenate((self.mux, [0.0]))
        # self.muy = np.concatenate((self.muy, [0.0]))


    def rays(self, mu: Union[float, Sequence[float]]):
        if isinstance(mu, float):
            mu = [mu]

        self.muz = np.array(mu)
        self.wmu = np.zeros_like(self.muz)
        self.muy = np.zeros_like(self.muz)
        self.mux = np.sqrt(1.0 - self.muz**2)


# @dataclass
# class MagneticAtmosphere(Atmosphere):
#     B: np.ndarray
#     gammaB: np.ndarray
#     chiB: np.ndarray

#     @classmethod
#     def from_atmos(cls, atmos: Atmosphere, B: np.ndarray, gammaB: np.ndarray, chiB: np.ndarray):
#         return cls(**asdict(atmos), B=B, gammaB=gammaB, chiB=chiB)

