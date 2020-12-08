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
    '''
    Atmospheric scales used in the definition of 1D atmospheres to allow the
    correct conversion to a height based system.
    Options:

        - `Geometric`

        - `ColumnMass`

        - `Tau500`

    '''
    Geometric = 0
    ColumnMass = auto()
    Tau500 = auto()

class BoundaryCondition:
    '''
    Base class for boundary conditions.

    Defines the interface; do not use directly.
    '''
    def compute_bc(self, atmos: 'Atmosphere', spect: 'LwSpectrum') -> np.ndarray:
        '''
        Called when the radiation boundary condition is needed by the backend.

        Parameters
        ----------
        atmos : Atmosphere
            The atmospheric object in which to compute the radiation.
        spect : LwSpectrum
            The computational spectrum object provided by the Context.

        Returns
        -------
        result : np.ndarray
            This function needs to return a contiguous array of shape [Nwave,
            Nrays, Nbc], where Nwave is the number of wavelengths in the
            wavelength grid, Nrays is the number of rays in the angular
            quadrature (also including up/down directions) ordered as per
            [mu[0] down, mu[0] up, mu[1] down...], Nbc is the number of
            spatial positions the boundary condition needs to be defined at
            ordered in a flattened [Nz, Ny, Nx] fashion. (dtype: <f8)

        '''
        raise NotImplementedError

class NoBc(BoundaryCondition):
    '''
    Indicates no boundary condition on the axis because it is invalid for the
    current simulation.
    Used only by the backend.
    '''
    pass

class ZeroRadiation(BoundaryCondition):
    '''
    Zero radiation boundary condition.
    Commonly used for coronal situations.
    '''
    pass

class ThermalisedRadiation(BoundaryCondition):
    '''
    Thermalised radiation (blackbody) boundary condition.
    Commonly used for photospheric situations.
    '''
    pass

class PeriodicRadiation(BoundaryCondition):
    '''
    Periodic boundary condition.
    Commonly used on the x-axis in 2D simulations.
    '''
    pass

def get_top_pressure(eos: witt, temp, ne=None, rho=None):
    '''
    Return a pressure for the top of atmosphere.
    For internal use.

    In order this is deduced from:
        - the electron density `ne`, if provided
        - the mass density `rho`, if provided
        - the electron pressure present in FALC

    Returns
    -------
    pressure : float
        pressure IN CGS [dyn cm-2]

    '''
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
    '''
    Stores the optional derived z-stratifications of an atmospheric model.

    Attributes
    ----------
    cmass : np.ndarray
        Column mass [kg m-2].
    tauRef : np.ndarray
        Reference optical depth at 500 nm.
    '''
    cmass: np.ndarray
    tauRef: np.ndarray

    def dimensioned_view(self, shape) -> 'Stratifications':
        '''
        Makes an instance of `Stratifications` reshaped to the provided
        shape for multi-dimensional atmospheres.
        For internal use.

        Parameters
        ----------
        shape : tuple
            Shape to reform the stratifications, provided by
            `Layout.dimensioned_shape`.

        Returns
        -------
        stratifications : Stratifications
            Reshaped stratifications.
        '''
        strat = copy(self)
        strat.cmass = self.cmass.reshape(shape)
        strat.tauRef = self.tauRef.reshape(shape)
        return strat

    def unit_view(self) -> 'Stratifications':
        '''
        Makes an instance of `Stratifications`  with the correct `astropy.units`
        For internal use.

        Returns
        -------
        stratifications : Stratifications
            The same data with units applied.
        '''
        strat = copy(self)
        strat.cmass = self.cmass << u.kg / u.m**2
        strat.tauRef = self.tauRef << u.dimensionless_unscaled
        return strat

    def dimensioned_unit_view(self, shape) -> 'Stratifications':
        '''
        Makes an instance of `Stratifications` reshaped to the provided shape
        with the correct `astropy.units` for multi-dimensional atmospheres.
        For internal use.

        Parameters
        ----------
        shape : tuple
            Shape to reform the stratifications, provided by
            `Layout.dimensioned_shape`.

        Returns
        -------
        stratifications : Stratifications
            Reshaped stratifications with units.
        '''
        strat = self.dimensioned_view(shape)
        return strat.unit_view()

@dataclass
class Layout:
    '''
    Storage for basic atmospheric parameters whose presence is determined by problem dimensionality, boundary conditions and optional stratifications.

    Attributes
    ----------
    Ndim : int
        Number of dimensions in model.

    x : np.ndarray
        Ordinates of grid points along the x-axis (present for Ndim >= 2) [m].
    y : np.ndarray
        Ordinates of grid points along the y-axis (present for Ndim == 3) [m].
    z : np.ndarray
        Ordinates of grid points along the z-axis (present for all Ndim) [m].
    vx : np.ndarray
        x component of plasma velocity (present for Ndim >= 2) [m/s].
    vy : np.ndarray
        y component of plasma velocity (present for Ndim == 3) [m/s].
    vz : np.ndarray
        z component of plasma velocity (present for all Ndim) [m/s]. Aliased to `vlos` when `Ndim==1`
    xLowerBc : BoundaryCondition
        Boundary condition for the plane of minimal x-coordinate.
    xUpperBc : BoundaryCondition
        Boundary condition for the plane of maximal x-coordinate.
    yLowerBc : BoundaryCondition
        Boundary condition for the plane of minimal y-coordinate.
    yUpperBc : BoundaryCondition
        Boundary condition for the plane of maximal y-coordinate.
    zLowerBc : BoundaryCondition
        Boundary condition for the plane of minimal z-coordinate.
    zUpperBc : BoundaryCondition
        Boundary condition for the plane of maximal z-coordinate.
    '''

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
        '''
        Construct 1D Layout.
        '''

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
        '''
        Construct 2D Layout.
        '''

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
        '''
        Construct 3D Layout.
        '''

        return cls(Ndim=3, x=x, y=y, z=z,
                   vx=vx, vy=vy, vz=vz,
                   xLowerBc=xLowerBc, xUpperBc=xUpperBc,
                   yLowerBc=yLowerBc, yUpperBc=yUpperBc,
                   zLowerBc=zLowerBc, zUpperBc=zUpperBc,
                   stratifications=stratifications)

    @property
    def Nx(self) -> int:
        '''
        Number of grid points along the x-axis.
        '''
        return self.x.shape[0]

    @property
    def Ny(self) -> int:
        '''
        Number of grid points along the y-axis.
        '''
        return self.y.shape[0]

    @property
    def Nz(self) -> int:
        '''
        Number of grid points along the z-axis.
        '''
        return self.z.shape[0]

    @property
    def Noutgoing(self) -> int:
        '''
        Number of grid points at which the outgoing radiation is computed.
        '''
        return max(1, self.Nx, self.Nx * self.Ny)

    @property
    def vlos(self) -> np.ndarray:
        if self.Ndim > 1:
            raise ValueError('vlos is ambiguous when Ndim > 1, use vx, vy, or vz instead.')
        return self.vz

    @property
    def Nspace(self) -> int:
        '''
        Number of spatial points present in the grid.
        '''
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
        '''
        Alias to `self.stratifications.tauRef`, if computed.
        '''
        if self.stratifications is not None:
            return self.stratifications.tauRef
        else:
            raise ValueError('tauRef not computed for this Atmosphere')

    @property
    def cmass(self):
        '''
        Alias to `self.stratifications.cmass`, if computed.
        '''
        if self.stratifications is not None:
            return self.stratifications.cmass
        else:
            raise ValueError('tauRef not computed for this Atmosphere')

    @property
    def dimensioned_shape(self):
        '''
        Tuple defining the shape to which the arrays of atmospheric paramters
        can be reshaped to be indexed in a 1/2/3D fashion.
        '''
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
        '''
        Returns a view over the contents of Layout reshaped so all data has
        the correct (1/2/3D) dimensionality for the atmospheric model, as
        these are all stored under a flat scheme.
        '''
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
        '''
        Returns a view over the contents of the Layout with the correct
        `astropy.units`.
        '''
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
        '''
        Returns a view over the contents of Layout reshaped so all data has
        the correct (1/2/3D) dimensionality for the atmospheric model, and
        the correct `astropy.units`.
        '''
        layout = self.dimensioned_view()
        return layout.unit_view()

@dataclass
class Atmosphere:
    '''
    Storage for all atmospheric data. These arrays will be shared directly
    with the backend, so a modification here also modifies the data seen by
    the backend. Be careful to modify these arrays *in place*, as their data
    is shared by direct memory reference. Use the class methods to construct
    atmospheres of different dimensionality.

    Attributes
    ----------
    structure : Layout
        A layout structure holding the atmospheric stratification, and
        velocity description.
    temperature : np.ndarray
        The atmospheric temperature structure.
    vturb : np.ndarray
        The atmospheric microturbulent velocity structure.
    ne : np.ndarray
        The electron density structure in the atmosphere.
    nHTot : np.ndarray
        The total hydrogen number density distribution throughout the
        atmosphere.
    B : np.ndarray, optional
        The magnitude of the stratified magnetic field throughout the
        atmosphere (Tesla).
    gammaB : np.ndarray, optional
        Co-altitude of magnetic field vector (radians) throughout the
        atmosphere from the local vertical.
    chiB : np.ndarray, optional
        Azimuth of magnetic field vector (radians) in the x-y plane, measured
        from the x-axis.
    '''

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
        '''
        Ndim : int
            The dimensionality (1, 2, or 3) of the atmospheric model.
        '''
        return self.structure.Ndim

    @property
    def Nx(self) -> int:
        '''
        Nx : int
            The number of points in the x-direction discretisation.
        '''
        return self.structure.Nx

    @property
    def Ny(self) -> int:
        '''
        Ny : int
            The number of points in the y-direction discretisation.
        '''
        return self.structure.Ny

    @property
    def Nz(self) -> int:
        '''
        Nz : int
            The number of points in the y-direction discretisation.
        '''
        return self.structure.Nz

    @property
    def Noutgoing(self) -> int:
        '''
        Noutgoing : int
            The number of cells at the top of the atmosphere (that each produce a
            spectrum).
        '''
        return self.structure.Noutgoing

    @property
    def vx(self) -> np.ndarray:
        '''
        vx : np.ndarray
            x component of plasma velocity (present for Ndim >= 2) [m/s].
        '''
        return self.structure.vx

    @property
    def vy(self) -> np.ndarray:
        '''
        vy : np.ndarray
            y component of plasma velocity (present for Ndim == 3) [m/s].
        '''
        return self.structure.vy

    @property
    def vz(self) -> np.ndarray:
        '''
        vz : np.ndarray
            z component of plasma velocity (present for all Ndim) [m/s]. Aliased
            to `vlos` when `Ndim==1`
        '''
        return self.structure.vz

    @property
    def vlos(self) -> np.ndarray:
        '''
        vz : np.ndarray
            z component of plasma velocity (present for all Ndim) [m/s]. Only
            available when Ndim==1`.
        '''
        return self.structure.vlos

    @property
    def cmass(self) -> np.ndarray:
        '''
        cmass : np.ndarray
            Column mass [kg m-2].
        '''
        return self.structure.cmass

    @property
    def tauRef(self) -> np.ndarray:
        '''
        tauRef : np.ndarray
            Reference optical depth at 500 nm.
        '''
        return self.structure.tauRef

    @property
    def height(self) -> np.ndarray:
        return self.structure.z

    @property
    def x(self) -> np.ndarray:
        '''
        x : np.ndarray
            Ordinates of grid points along the x-axis (present for Ndim >= 2) [m].
        '''
        return self.structure.x

    @property
    def y(self) -> np.ndarray:
        '''
        y : np.ndarray
            Ordinates of grid points along the y-axis (present for Ndim == 3) [m].
        '''
        return self.structure.y

    @property
    def z(self) -> np.ndarray:
        '''
        z : np.ndarray
            Ordinates of grid points along the z-axis (present for all Ndim) [m].
        '''
        return self.structure.z

    @property
    def zLowerBc(self) -> BoundaryCondition:
        '''
        zLowerBc : BoundaryCondition
            Boundary condition for the plane of minimal z-coordinate.
        '''
        return self.structure.zLowerBc

    @property
    def zUpperBc(self) -> BoundaryCondition:
        '''
        zUpperBc : BoundaryCondition
            Boundary condition for the plane of maximal z-coordinate.
        '''
        return self.structure.zUpperBc

    @property
    def yLowerBc(self) -> BoundaryCondition:
        '''
        yLowerBc : BoundaryCondition
            Boundary condition for the plane of minimal y-coordinate.
        '''
        return self.structure.yLowerBc

    @property
    def yUpperBc(self) -> BoundaryCondition:
        '''
        yUpperBc : BoundaryCondition
            Boundary condition for the plane of maximal y-coordinate.
        '''
        return self.structure.yUpperBc

    @property
    def xLowerBc(self) -> BoundaryCondition:
        '''
        xLowerBc : BoundaryCondition
            Boundary condition for the plane of minimal x-coordinate.
        '''
        return self.structure.xLowerBc

    @property
    def xUpperBc(self) -> BoundaryCondition:
        '''
        xUpperBc : BoundaryCondition
            Boundary condition for the plane of maximal x-coordinate.
        '''
        return self.structure.xUpperBc

    @property
    def Nspace(self):
        '''
        Nspace : int
            Total number of points in the atmospheric spatial discretistaion.
        '''
        return self.structure.Nspace

    @property
    def Nrays(self):
        '''
        Nrays : int
            Number of rays in angular discretisation used.
        '''
        if self.muz is None:
            raise AttributeError('Nrays not set, call atmos.rays or .quadrature first')

        return self.muz.shape[0]

    def dimensioned_view(self):
        '''
        Returns a view over the contents of Layout reshaped so all data has
        the correct (1/2/3D) dimensionality for the atmospheric model, as
        these are all stored under a flat scheme.
        '''
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
        '''
        Returns a view over the contents of the Layout with the correct
        `astropy.units`.
        '''
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
        '''
        Returns a view over the contents of Layout reshaped so all data has
        the correct (1/2/3D) dimensionality for the atmospheric model, and
        the correct `astropy.units`.
        '''
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
        '''
        Constructor for 1D Atmosphere objects. Optionally will use an
        equation of state (EOS) to estimate missing parameters.

        If sufficient information is provided (i.e. all required parameters
        and ne and (hydrogenPops or nHTot)) then the EOS is not invoked to
        estimate any thermodynamic properties. If both of nHTot and
        hydrogenPops are omitted, then the electron pressure will be used
        with the Wittmann equation of state to estimate the mass density, and
        the hydrogen number density will be inferred from this and the
        abundances. If, instead, ne is omitted, then the mass density will be
        used with the Wittmann EOS to estimate the electron pressure.
        If both of these are omitted then the EOS will be used to estimate
        both. If:

            - Pgas is provided, then this gas pressure will define the
              atmospheric stratification and will be used with the EOS.

            - Pe is provided, then this electron pressure will define the
              atmospheric stratification and will be used with the EOS.

            - Ptop is provided, then this gas pressure at the top of the
              atmosphere will be used with the log gravitational acceleration
              logG, and the EOS to estimate the missing parameters assuming
              hydrostatic equilibrium.

            - PeTop is provided, then this electron pressure at the top of
              the atmosphere will be used with the log gravitational
              acceleration logG, and the EOS to estimate the missing parameters
              assuming hydrostatic equilibrium.

            - If all of Pgas, Pe, Ptop, PeTop are omitted then Ptop will be
              estimated from the gas pressure in the FALC model at the
              temperature at the top boundary. The hydrostatic reconstruction
              will then continue as usual.

        convertScales will substantially slow down this function due to the
        slow calculation of background opacities used to compute tauRef. If
        an atmosphere is constructed with a Geometric stratification, and an
        estimate of tauRef is not required before running the main RT module,
        then this can be set to False.
        All of these parameters can be provided as astropy Quantities, and
        will be converted in the constructor.

        Parameters
        ----------
        scale : ScaleType
            The type of stratification used along the z-axis.
        depthScale : np.ndarray
            The z-coordinates used along the chosen stratification. The
            stratification is expected to start at the top of the atmosphere
            (closest to the observer), and descend along the observer's line
            of sight.
        temperature : np.ndarray
            Temperature structure of the atmosphere [K].
        vlos : np.ndarray
            Velocity structure of the atmosphere along z [m/s].
        vturb : np.ndarray
            Microturbulent velocity structure of the atmosphere [m/s].
        ne : np.ndarray
            Electron density structure of the atmosphere [m-3].
        hydrogenPops : np.ndarray, optional
            Detailed (per level) hydrogen number density structure of the
            atmosphere [m-3], 2D array [Nlevel, Nspace].
        nHTot : np.ndarray, optional
            Total hydrogen number density structure of the atmosphere [m-3]
        B : np.ndarray, optional.
            Magnetic field strength [T].
        gammaB : np.ndarray, optional
            Co-altitude of magnetic field vector [radians].
        chiB : np.ndarray, optional
            Azimuth of magnetic field vector (in x-y plane, from x) [radians].
        lowerBc : BoundaryCondition, optional
            Boundary condition for incoming radiation at the minimal z
            coordinate (default: ThermalisedRadiation).
        upperBc : BoundaryCondition, optional
            Boundary condition for incoming radiation at the maximal z
            coordinate (default: ZeroRadiation).
        convertScales : bool, optional
            Whether to automatically compute tauRef and cmass for an
            atmosphere given in a stratification of m (default: True).
        abundance: AtomicAbundance, optional
            An instance of AtomicAbundance giving the abundances of each
            atomic species in the given atmosphere, only used if the EOS is
            invoked. (default: DefaultAtomicAbundance)
        logG: float, optional
            The log10 of the magnitude of gravitational acceleration [m/s2]
            (default: 2.44).
        Pgas: np.ndarray, optional
            The gas pressure stratification of the atmosphere [Pa],
            optionally used by the EOS.
        Pe: np.ndarray, optional
            The electron pressure stratification of the atmosphere [Pa],
            optionally used by the EOS.
        Ptop: np.ndarray, optional
            The gas pressure at the top of the atmosphere [Pa], optionally
            used by the EOS for a hydrostatic reconstruction.
        Petop: np.ndarray, optional
            The electron pressure at the top of the atmosphere [Pa],
            optionally used by the EOS for a hydrostatic reconstruction.
        verbose: bool, optional
            Explain decisions made with the EOS to estimate missing
            parameters (if invoked) through print calls (default: False).

        Raises
        ------
        ValueError
            if incorrect arguments or unable to construct estimate missing
            parameters.
        '''
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
            nHTot = (nHTot << u.m**(-3)).value
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
            nHTot = np.copy(rho / (Const.CM_TO_M**3 / Const.G_TO_KG) / (Const.Amu * abundance.massPerH))
            ne = np.copy(pe / (eos.BK * temperature) / Const.CM_TO_M**3)

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
        '''
        Constructor for 2D Atmosphere objects.

        No provision for estimating parameters using hydrostatic equilibrium
        is provided, but one of ne, or nHTot can be omitted and inferred by
        use of the Wittmann equation of state.
        The atmosphere must be defined on a geometric stratification.
        All atmospheric parameters are expected in a 2D [z, x] array.

        Parameters
        ----------
        height : np.ndarray
            The z-coordinates of the atmospheric grid. The stratification is
            expected to start at the top of the atmosphere (closest to the
            observer), and descend along the observer's line of sight.
        x : np.ndarray
            The (horizontal) x-coordinates of the atmospheric grid.
        temperature : np.ndarray
            Temperature structure of the atmosphere [K].
        vx : np.ndarray
            x-component of the atmospheric velocity [m/s].
        vz : np.ndarray
            z-component of the atmospheric velocity [m/s].
        vturb : np.ndarray
            Microturbulent velocity structure [m/s].
        ne : np.ndarray
            Electron density structure of the atmosphere [m-3].
        nHTot : np.ndarray, optional
            Total hydrogen number density structure of the atmosphere [m-3].
        B : np.ndarray, optional.
            Magnetic field strength [T].
        gammaB : np.ndarray, optional
            Co-altitude of magnetic field vector [radians].
        chiB : np.ndarray, optional
            Azimuth of magnetic field vector (in x-y plane, from x) [radians].
        xLowerBc : BoundaryCondition, optional
            Boundary condition for incoming radiation at the minimal x
            coordinate (default: PeriodicRadiation).
        xUpperBc : BoundaryCondition, optional
            Boundary condition for incoming radiation at the maximal x
            coordinate (default: PeriodicRadiation).
        zLowerBc : BoundaryCondition, optional
            Boundary condition for incoming radiation at the minimal z
            coordinate (default: ThermalisedRadiation).
        zUpperBc : BoundaryCondition, optional
            Boundary condition for incoming radiation at the maximal z
            coordinate (default: ZeroRadiation).
        convertScales : bool, optional
            Whether to automatically compute tauRef and cmass for an
            atmosphere given in a stratification of m (default: True).
        abundance: AtomicAbundance, optional
            An instance of AtomicAbundance giving the abundances of each
            atomic species in the given atmosphere, only used if the EOS is
            invoked. (default: DefaultAtomicAbundance)
        verbose: bool, optional
            Explain decisions made with the EOS to estimate missing
            parameters (if invoked) through print calls (default: False).

        Raises
        ------
        ValueError
            if incorrect arguments or unable to construct estimate missing
            parameters.
        '''

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
        '''
        Compute the angular quadrature for solving the RTE and Kinetic
        Equilibrium in a given atmosphere.

        Procedure varies with dimensionality.

        1D:
            If a number of rays is given (typically 3 or 5), then the
            Gauss-Legendre quadrature for this set is used.
            If mu and wmu are instead given then these will be validated and
            used.

        2+D:
            If the number of rays selected is in the list of near optimal
            quadratures for unpolarised radiation provided by Stepan et al
            2020 (A&A, 646 A24), then this is used. Otherwise an exception is
            raised.

            The available quadratures are:

            +--------+-------+
            | Points | Order |
            +========+=======+
            |   1    |  3    |
            +--------+-------+
            |   3    |  7    |
            +--------+-------+
            |   6    |  9    |
            +--------+-------+
            |   7    |  11   |
            +--------+-------+
            |   10   |  13   |
            +--------+-------+
            |   11   |  15   |
            +--------+-------+

        Parameters
        ----------
        Nrays : int, optional
            The number of rays to use in the quadrature. See notes above.
        mu : sequence of float, optional
            The cosine of the angle made between the between each of the set
            of rays and the z axis, only used in 1D.
        wmu : sequence of float, optional
            The integration weights for each mu, must be provided if mu is provided.

        Raises
        ------
        ValueError
            on incorrect input.
        '''

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


    def rays(self, muz: Union[float, Sequence[float]],
             mux: Optional[Union[float, Sequence[float]]]=None,
             muy: Optional[Union[float, Sequence[float]]]=None,
             wmu: Optional[Union[float, Sequence[float]]]=None):
        '''
        Set up the rays on the Atmosphere for computing the intensity in a
        particular direction (or set of directions).

        If only the z angle is set then the ray is assumed in the x-z plane.
        If either muz or muy is omitted then this angle is inferred by
        normalisation of the projection.

        Parameters
        ----------
        muz : float or sequence of float, optional
            The angular projections along the z axis.
        mux : float or sequence of float, optional
            The angular projections along the x axis.
        muy : float or sequence of float, optional
            The angular projections along the y axis.
        wmu : float or sequence of float, optional
            The integration weights for the given ray if J is to be
            integrated for angle set.

        Raises
        ------
        ValueError
            if the angular projections or integration weights are incorrectly
            normalised.
        '''

        if isinstance(muz, float):
            muz = [muz]
        if isinstance(mux, float):
            mux = [mux]
        if isinstance(muy, float):
            muy = [muy]

        if mux is None and muy is None:
            self.muz = np.array(muz)
            self.wmu = np.zeros_like(self.muz)
            self.muy = np.zeros_like(self.muz)
            self.mux = np.sqrt(1.0 - self.muz**2)
        elif muy is None:
            self.muz = np.array(muz)
            self.wmu = np.zeros_like(self.muz)
            self.mux = np.array(mux)
            self.muy = np.sqrt(1.0 - (self.muz**2 + self.mux**2))
        elif mux is None:
            self.muz = np.array(muz)
            self.wmu = np.zeros_like(self.muz)
            self.muy = np.array(muy)
            self.mux = np.sqrt(1.0 - (self.muz**2 + self.muy**2))
        else:
            self.muz = np.array(muz)
            self.mux = np.array(mux)
            self.muy = np.array(muy)
            self.wmu = np.zeros_like(muz)

            if not np.allclose(self.muz**2 + self.mux**2 + self.muy**2, 1):
                raise ValueError('mux**2 + muy**2 + muz**2 != 1.0')

        if wmu is not None:
            self.wmu = np.array(wmu)

            if not np.isclose(self.wmu.sum(), 1.0):
                raise ValueError('sum of wmus is not 1.0')


# @dataclass
# class MagneticAtmosphere(Atmosphere):
#     B: np.ndarray
#     gammaB: np.ndarray
#     chiB: np.ndarray

#     @classmethod
#     def from_atmos(cls, atmos: Atmosphere, B: np.ndarray, gammaB: np.ndarray, chiB: np.ndarray):
#         return cls(**asdict(atmos), B=B, gammaB=gammaB, chiB=chiB)

