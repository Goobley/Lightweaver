import numbers
import pickle
from copy import copy
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional, Sequence, Union, cast

import astropy.units as u
import numpy as np
from numpy.polynomial.legendre import leggauss

import lightweaver.constants as Const

from .atomic_table import AtomicAbundance, DefaultAtomicAbundance, PeriodicTable
from .deprecation import deprecated_names
from .utils import ConvergenceError, check_shape_exception, get_data_path, view_flatten
from .wittmann import Wittmann, cgs

if TYPE_CHECKING:
    from .LwCompiled import LwSpectrum


class ScaleType(Enum):
    """
    Atmospheric scales used in the definition of 1D atmospheres to allow the
    correct conversion to a height based system.
    Options:

        - `Geometric`

        - `ColumnMass`

        - `Tau500`

    """

    Geometric = 0
    ColumnMass = auto()
    Tau500 = auto()


@deprecated_names(attrs=('index_vector',))
class BoundaryCondition:
    """
    Base class for boundary conditions.

    Defines the interface; do not use directly.

    Attributes
    ----------
    These attributes are only available after set_required_angles has been called.

    mux : np.ndarray
        The mu_x to return from compute_bc (in order).
    muy : np.ndarray
        The mu_y to return from compute_bc (in order).
    muz : np.ndarray
        The mu_z to return from compute_bc (in order).
    index_vector : np.ndarray
        A 2D array of integer shape (mu, to_obs) - where mu is the mu index on
        the associated atmosphere - relating each index of the second (Nrays)
        axis of a pair of (mu, to_obs). Used to construct and destructure this
        array.

    """

    def compute_bc(self, atmos: 'Atmosphere', spect: 'LwSpectrum') -> np.ndarray:
        """
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
            quadrature (also including up/down directions) ordered as
            specified by the mux/y/z and index_vector variables on this
            object, Nbc is the number of spatial positions the boundary
            condition needs to be defined at ordered in a flattened [Nz, Ny,
            Nx] fashion. (dtype: <f8)

        """
        raise NotImplementedError

    def set_required_angles(self, mux, muy, muz, index_vector):
        """
        The angles (and their ordering) to be used for this boundary
        condition (in the case of a callable)
        """
        self.mux = mux
        self.muy = muy
        self.muz = muz
        self.index_vector = index_vector


class NoBc(BoundaryCondition):
    """
    Indicates no boundary condition on the axis because it is invalid for the
    current simulation.
    Used only by the backend.
    """

    pass


class ZeroRadiation(BoundaryCondition):
    """
    Zero radiation boundary condition.
    Commonly used for coronal situations.
    """

    pass


class ThermalisedRadiation(BoundaryCondition):
    """
    Thermalised radiation (blackbody) boundary condition.
    Commonly used for photospheric situations.
    """

    pass


class PeriodicRadiation(BoundaryCondition):
    """
    Periodic boundary condition.
    Commonly used on the x-axis in 2D simulations.
    """

    pass


def get_top_pressure(eos: Wittmann, temp, ne=None, rho=None):
    """
    Return a pressure for the top of atmosphere.
    For internal use.

    In order this is deduced from:
        - the electron density `ne` [m-3], if provided
        - the mass density `rho` [kg m-3], if provided
        - the electron pressure present in FALC

    Returns
    -------
    pressure : float
        pressure IN CGS [dyn cm-2]

    """
    if ne is not None:
        pe = (ne << u.Unit('m-3')).to('cm-3').value * cgs.BK * temp
        return eos.pg_from_pe(temp, pe)
    elif rho is not None:
        return eos.pg_from_rho(temp, (rho << u.Unit('kg m-3')).to('g cm-3').value)

    pgasCgs = np.array(
        [
            0.70575286,
            0.59018545,
            0.51286639,
            0.43719268,
            0.37731009,
            0.33516886,
            0.31342915,
            0.30604891,
            0.30059491,
            0.29207645,
            0.2859011,
            0.28119224,
            0.27893046,
            0.27949676,
            0.28299726,
            0.28644693,
            0.28825946,
            0.29061192,
            0.29340255,
            0.29563072,
            0.29864548,
            0.30776456,
            0.31825915,
            0.32137574,
            0.3239401,
            0.32622212,
            0.32792196,
            0.3292243,
            0.33025437,
            0.33146736,
            0.3319676,
            0.33217821,
            0.3322355,
            0.33217166,
            0.33210297,
            0.33203833,
            0.33198508,
        ]
    )
    tempCoord = np.array(
        [
            7600.0,
            7780.0,
            7970.0,
            8273.0,
            8635.0,
            8988.0,
            9228.0,
            9358.0,
            9458.0,
            9587.0,
            9735.0,
            9983.0,
            10340.0,
            10850.0,
            11440.0,
            12190.0,
            13080.0,
            14520.0,
            16280.0,
            17930.0,
            20420.0,
            24060.0,
            27970.0,
            32150.0,
            36590.0,
            41180.0,
            45420.0,
            49390.0,
            53280.0,
            60170.0,
            66150.0,
            71340.0,
            75930.0,
            83890.0,
            90820.0,
            95600.0,
            100000.0,
        ]
    )

    ptop = np.interp(temp, tempCoord, pgasCgs)
    return ptop


@deprecated_names
@dataclass
class Stratifications:
    """
    Stores the optional derived z-stratifications of an atmospheric model.

    Attributes
    ----------
    cmass : np.ndarray
        Column mass [kg m-2].
    tau_ref : np.ndarray
        Reference optical depth at 500 nm.
    """

    cmass: np.ndarray
    tau_ref: np.ndarray

    def dimensioned_view(self, shape) -> 'Stratifications':
        """
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
        """
        strat = copy(self)
        strat.cmass = self.cmass.reshape(shape)
        strat.tau_ref = self.tau_ref.reshape(shape)
        return strat

    def unit_view(self) -> 'Stratifications':
        """
        Makes an instance of `Stratifications`  with the correct `astropy.units`
        For internal use.

        Returns
        -------
        stratifications : Stratifications
            The same data with units applied.
        """
        strat = copy(self)
        strat.cmass = self.cmass << u.kg / u.m**2
        strat.tau_ref = self.tau_ref << u.dimensionless_unscaled
        return strat

    def dimensioned_unit_view(self, shape) -> 'Stratifications':
        """
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
        """
        strat = self.dimensioned_view(shape)
        return strat.unit_view()


@deprecated_names
@dataclass
class Layout:
    """
    Storage for basic atmospheric parameters whose presence is determined by
    problem dimensionality, boundary conditions and optional stratifications.

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
        x component of plasma velocity (optional for Ndim < 2) [m/s].
    vy : np.ndarray
        y component of plasma velocity (optional for Ndim < 3) [m/s].
    vz : np.ndarray
        z component of plasma velocity (present for all Ndim) [m/s]. Aliased to
        `vlos` when `Ndim==1`
    x_lower_bc : BoundaryCondition
        Boundary condition for the plane of minimal x-coordinate.
    x_upper_bc : BoundaryCondition
        Boundary condition for the plane of maximal x-coordinate.
    y_lower_bc : BoundaryCondition
        Boundary condition for the plane of minimal y-coordinate.
    y_upper_bc : BoundaryCondition
        Boundary condition for the plane of maximal y-coordinate.
    z_lower_bc : BoundaryCondition
        Boundary condition for the plane of minimal z-coordinate.
    z_upper_bc : BoundaryCondition
        Boundary condition for the plane of maximal z-coordinate.
    """

    Ndim: int
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    x_lower_bc: BoundaryCondition
    x_upper_bc: BoundaryCondition
    y_lower_bc: BoundaryCondition
    y_upper_bc: BoundaryCondition
    z_lower_bc: BoundaryCondition
    z_upper_bc: BoundaryCondition
    stratifications: Optional[Stratifications] = None

    @classmethod
    def make_1d(
        cls,
        z: np.ndarray,
        vz: np.ndarray,
        lower_bc: BoundaryCondition,
        upper_bc: BoundaryCondition,
        stratifications: Optional[Stratifications] = None,
        vx: Optional[np.ndarray] = None,
        vy: Optional[np.ndarray] = None,
    ) -> 'Layout':
        """
        Construct 1D Layout.
        """

        if vx is None:
            vx = np.array(())
        if vy is None:
            vy = np.array(())

        return cls(
            Ndim=1,
            x=np.array(()),
            y=np.array(()),
            z=z,
            vx=vx,
            vy=vy,
            vz=vz,
            x_lower_bc=NoBc(),
            x_upper_bc=NoBc(),
            y_lower_bc=NoBc(),
            y_upper_bc=NoBc(),
            z_lower_bc=lower_bc,
            z_upper_bc=upper_bc,
            stratifications=stratifications,
        )

    @classmethod
    def make_2d(
        cls,
        x: np.ndarray,
        z: np.ndarray,
        vx: np.ndarray,
        vz: np.ndarray,
        x_lower_bc: BoundaryCondition,
        x_upper_bc: BoundaryCondition,
        z_lower_bc: BoundaryCondition,
        z_upper_bc: BoundaryCondition,
        stratifications: Optional[Stratifications] = None,
        vy: Optional[np.ndarray] = None,
    ) -> 'Layout':
        """
        Construct 2D Layout.
        """
        if vy is None:
            vy = np.array(())

        return cls(
            Ndim=2,
            x=x,
            y=np.array(()),
            z=z,
            vx=vx,
            vy=vy,
            vz=vz,
            x_lower_bc=x_lower_bc,
            x_upper_bc=x_upper_bc,
            y_lower_bc=NoBc(),
            y_upper_bc=NoBc(),
            z_lower_bc=z_lower_bc,
            z_upper_bc=z_upper_bc,
            stratifications=stratifications,
        )

    @classmethod
    def make_3d(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        vx: np.ndarray,
        vy: np.ndarray,
        vz: np.ndarray,
        x_lower_bc: BoundaryCondition,
        x_upper_bc: BoundaryCondition,
        y_lower_bc: BoundaryCondition,
        y_upper_bc: BoundaryCondition,
        z_lower_bc: BoundaryCondition,
        z_upper_bc: BoundaryCondition,
        stratifications: Optional[Stratifications] = None,
    ) -> 'Layout':
        """
        Construct 3D Layout.
        """

        return cls(
            Ndim=3,
            x=x,
            y=y,
            z=z,
            vx=vx,
            vy=vy,
            vz=vz,
            x_lower_bc=x_lower_bc,
            x_upper_bc=x_upper_bc,
            y_lower_bc=y_lower_bc,
            y_upper_bc=y_upper_bc,
            z_lower_bc=z_lower_bc,
            z_upper_bc=z_upper_bc,
            stratifications=stratifications,
        )

    @property
    def Nx(self) -> int:
        """
        Number of grid points along the x-axis.
        """
        return self.x.shape[0]

    @property
    def Ny(self) -> int:
        """
        Number of grid points along the y-axis.
        """
        return self.y.shape[0]

    @property
    def Nz(self) -> int:
        """
        Number of grid points along the z-axis.
        """
        return self.z.shape[0]

    @property
    def Noutgoing(self) -> int:
        """
        Number of grid points at which the outgoing radiation is computed.
        """
        return max(1, self.Nx, self.Nx * self.Ny)

    @property
    def vlos(self) -> np.ndarray:
        if self.Ndim > 1:
            raise ValueError('vlos is ambiguous when Ndim > 1, use vx, vy, or vz instead.')
        return self.vz

    @property
    def Nspace(self) -> int:
        """
        Number of spatial points present in the grid.
        """
        if self.Ndim == 1:
            return self.Nz
        elif self.Ndim == 2:
            return self.Nx * self.Nz
        elif self.Ndim == 3:
            return self.Nx * self.Ny * self.Nz
        else:
            raise ValueError('Invalid Ndim: %d, check geometry initialisation' % self.Ndim)

    @property
    def tau_ref(self):
        """
        Alias to `self.stratifications.tau_ref`, if computed.
        """
        if self.stratifications is not None:
            return self.stratifications.tau_ref
        else:
            raise ValueError('tau_ref not computed for this Atmosphere')

    @property
    def cmass(self):
        """
        Alias to `self.stratifications.cmass`, if computed.
        """
        if self.stratifications is not None:
            return self.stratifications.cmass
        else:
            raise ValueError('tau_ref not computed for this Atmosphere')

    @property
    def dimensioned_shape(self):
        """
        Tuple defining the shape to which the arrays of atmospheric paramters
        can be reshaped to be indexed in a 1/2/3D fashion.
        """
        if self.Ndim == 1:
            shape = (self.Nz,)
        elif self.Ndim == 2:
            shape = (self.Nz, self.Nx)
        elif self.Ndim == 3:
            shape = (self.Nz, self.Ny, self.Nx)
        else:
            raise ValueError('Unreasonable Ndim (%d)' % self.Ndim)
        return shape

    def dimensioned_view(self) -> 'Layout':
        """
        Returns a view over the contents of Layout reshaped so all data has
        the correct (1/2/3D) dimensionality for the atmospheric model, as
        these are all stored under a flat scheme.
        """
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
        """
        Returns a view over the contents of the Layout with the correct
        `astropy.units`.
        """
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
        """
        Returns a view over the contents of Layout reshaped so all data has
        the correct (1/2/3D) dimensionality for the atmospheric model, and
        the correct `astropy.units`.
        """
        layout = self.dimensioned_view()
        return layout.unit_view()


@deprecated_names
@dataclass
class Atmosphere:
    """
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
    nh_tot : np.ndarray
        The total hydrogen number density distribution throughout the
        atmosphere.
    B : np.ndarray, optional
        The magnitude of the stratified magnetic field throughout the
        atmosphere (Tesla).
    gamma_B : np.ndarray, optional
        Co-altitude (latitude) of magnetic field vector (radians) throughout the
        atmosphere from the local vertical.
    chi_B : np.ndarray, optional
        Azimuth of magnetic field vector (radians) in the x-y plane, measured
        from the x-axis.
    """

    structure: Layout
    temperature: np.ndarray
    vturb: np.ndarray
    ne: np.ndarray
    nh_tot: np.ndarray
    B: Optional[np.ndarray] = None
    gamma_B: Optional[np.ndarray] = None
    chi_B: Optional[np.ndarray] = None

    @property
    def Ndim(self) -> int:
        """
        Ndim : int
            The dimensionality (1, 2, or 3) of the atmospheric model.
        """
        return self.structure.Ndim

    @property
    def Nx(self) -> int:
        """
        Nx : int
            The number of points in the x-direction discretisation.
        """
        return self.structure.Nx

    @property
    def Ny(self) -> int:
        """
        Ny : int
            The number of points in the y-direction discretisation.
        """
        return self.structure.Ny

    @property
    def Nz(self) -> int:
        """
        Nz : int
            The number of points in the y-direction discretisation.
        """
        return self.structure.Nz

    @property
    def Noutgoing(self) -> int:
        """
        Noutgoing : int
            The number of cells at the top of the atmosphere (that each produce a
            spectrum).
        """
        return self.structure.Noutgoing

    @property
    def vx(self) -> np.ndarray:
        """
        vx : np.ndarray
            x component of plasma velocity (present for Ndim >= 2) [m/s].
        """
        return self.structure.vx

    @property
    def vy(self) -> np.ndarray:
        """
        vy : np.ndarray
            y component of plasma velocity (present for Ndim == 3) [m/s].
        """
        return self.structure.vy

    @property
    def vz(self) -> np.ndarray:
        """
        vz : np.ndarray
            z component of plasma velocity (present for all Ndim) [m/s]. Aliased
            to `vlos` when `Ndim==1`
        """
        return self.structure.vz

    @property
    def vlos(self) -> np.ndarray:
        """
        vz : np.ndarray
            z component of plasma velocity (present for all Ndim) [m/s]. Only
            available when Ndim==1`.
        """
        return self.structure.vlos

    @property
    def cmass(self) -> np.ndarray:
        """
        cmass : np.ndarray
            Column mass [kg m-2].
        """
        return self.structure.cmass

    @property
    def tau_ref(self) -> np.ndarray:
        """
        tau_ref : np.ndarray
            Reference optical depth at 500 nm.
        """
        return self.structure.tau_ref

    @property
    def height(self) -> np.ndarray:
        return self.structure.z

    @property
    def x(self) -> np.ndarray:
        """
        x : np.ndarray
            Ordinates of grid points along the x-axis (present for Ndim >= 2) [m].
        """
        return self.structure.x

    @property
    def y(self) -> np.ndarray:
        """
        y : np.ndarray
            Ordinates of grid points along the y-axis (present for Ndim == 3) [m].
        """
        return self.structure.y

    @property
    def z(self) -> np.ndarray:
        """
        z : np.ndarray
            Ordinates of grid points along the z-axis (present for all Ndim) [m].
        """
        return self.structure.z

    @property
    def z_lower_bc(self) -> BoundaryCondition:
        """
        z_lower_bc : BoundaryCondition
            Boundary condition for the plane of minimal z-coordinate.
        """
        return self.structure.z_lower_bc

    @property
    def z_upper_bc(self) -> BoundaryCondition:
        """
        z_upper_bc : BoundaryCondition
            Boundary condition for the plane of maximal z-coordinate.
        """
        return self.structure.z_upper_bc

    @property
    def y_lower_bc(self) -> BoundaryCondition:
        """
        y_lower_bc : BoundaryCondition
            Boundary condition for the plane of minimal y-coordinate.
        """
        return self.structure.y_lower_bc

    @property
    def y_upper_bc(self) -> BoundaryCondition:
        """
        y_upper_bc : BoundaryCondition
            Boundary condition for the plane of maximal y-coordinate.
        """
        return self.structure.y_upper_bc

    @property
    def x_lower_bc(self) -> BoundaryCondition:
        """
        x_lower_bc : BoundaryCondition
            Boundary condition for the plane of minimal x-coordinate.
        """
        return self.structure.x_lower_bc

    @property
    def x_upper_bc(self) -> BoundaryCondition:
        """
        x_upper_bc : BoundaryCondition
            Boundary condition for the plane of maximal x-coordinate.
        """
        return self.structure.x_upper_bc

    @property
    def Nspace(self):
        """
        Nspace : int
            Total number of points in the atmospheric spatial discretistaion.
        """
        return self.structure.Nspace

    @property
    def Nrays(self):
        """
        Nrays : int
            Number of rays in angular discretisation used.
        """
        try:
            if self.muz is None:
                raise AttributeError('Nrays not set, call atmos.rays or .quadrature first')
        except AttributeError:
            raise AttributeError('Nrays not set, call atmos.rays or .quadrature first')

        return self.muz.shape[0]

    def dimensioned_view(self):
        """
        Returns a view over the contents of Layout reshaped so all data has
        the correct (1/2/3D) dimensionality for the atmospheric model, as
        these are all stored under a flat scheme.
        """
        shape = self.structure.dimensioned_shape
        atmos = copy(self)
        atmos.structure = self.structure.dimensioned_view()
        atmos.temperature = self.temperature.reshape(shape)
        atmos.vturb = self.vturb.reshape(shape)
        atmos.ne = self.ne.reshape(shape)
        atmos.nh_tot = self.nh_tot.reshape(shape)
        if self.B is not None:
            atmos.B = self.B.reshape(shape)
            atmos.chi_B = self.chi_B.reshape(shape)
            atmos.gamma_B = self.gamma_B.reshape(shape)
        return atmos

    def unit_view(self):
        """
        Returns a view over the contents of the Layout with the correct
        `astropy.units`.
        """
        atmos = copy(self)
        atmos.structure = self.structure.unit_view()
        atmos.temperature = self.temperature << u.K
        atmos.vturb = self.vturb << u.m / u.s
        atmos.ne = self.ne << u.m ** (-3)
        atmos.nh_tot = self.nh_tot << u.m ** (-3)
        if self.B is not None:
            atmos.B = self.B << u.T
            atmos.chi_B = self.chi_B << u.rad
            atmos.gamma_B = self.gamma_B << u.rad
        return atmos

    def dimensioned_unit_view(self):
        """
        Returns a view over the contents of Layout reshaped so all data has
        the correct (1/2/3D) dimensionality for the atmospheric model, and
        the correct `astropy.units`.
        """
        atmos = self.dimensioned_view()
        return atmos.unit_view()

    @classmethod
    def make_1d(
        cls,
        scale: ScaleType,
        depth_scale: np.ndarray,
        temperature: np.ndarray,
        vlos: Optional[np.ndarray] = None,
        vturb: Optional[np.ndarray] = None,
        ne: Optional[np.ndarray] = None,
        hydrogen_pops: Optional[np.ndarray] = None,
        nh_tot: Optional[np.ndarray] = None,
        vx: Optional[np.ndarray] = None,
        vy: Optional[np.ndarray] = None,
        vz: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        gamma_B: Optional[np.ndarray] = None,
        chi_B: Optional[np.ndarray] = None,
        lower_bc: Optional[BoundaryCondition] = None,
        upper_bc: Optional[BoundaryCondition] = None,
        convert_scales: bool = True,
        abundance: Optional[AtomicAbundance] = None,
        log_g: float = 2.44,
        Pgas: Optional[np.ndarray] = None,
        Pe: Optional[np.ndarray] = None,
        Ptop: Optional[float] = None,
        Pe_top: Optional[float] = None,
        verbose: bool = False,
    ):
        """
        Constructor for 1D Atmosphere objects. Optionally will use an
        equation of state (EOS) to estimate missing parameters.

        If sufficient information is provided (i.e. all required parameters
        and ne and (hydrogen_pops or nh_tot)) then the EOS is not invoked to
        estimate any thermodynamic properties. If both of nh_tot and
        hydrogen_pops are omitted, then the electron pressure will be used
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
              log_g, and the EOS to estimate the missing parameters assuming
              hydrostatic equilibrium.

            - Pe_top is provided, then this electron pressure at the top of
              the atmosphere will be used with the log gravitational
              acceleration log_g, and the EOS to estimate the missing parameters
              assuming hydrostatic equilibrium.

            - If all of Pgas, Pe, Ptop, Pe_top are omitted then Ptop will be
              estimated from the gas pressure in the FALC model at the
              temperature at the top boundary. The hydrostatic reconstruction
              will then continue as usual.

        convert_scales will substantially slow down this function due to the
        slow calculation of background opacities used to compute tau_ref. If
        an atmosphere is constructed with a Geometric stratification, and an
        estimate of tau_ref is not required before running the main RT module,
        then this can be set to False.
        All of these parameters can be provided as astropy Quantities, and
        will be converted in the constructor.

        Parameters
        ----------
        scale : ScaleType
            The type of stratification used along the z-axis.
        depth_scale : np.ndarray
            The z-coordinates used along the chosen stratification. The
            stratification is expected to start at the top of the atmosphere
            (closest to the observer), and descend along the observer's line
            of sight.
        temperature : np.ndarray
            Temperature structure of the atmosphere [K].
        vlos : np.ndarray, optional
            Alias for vz
            Velocity structure of the atmosphere along z [m/s]. Default: 0 m/s everywhere
        vturb : np.ndarray
            Microturbulent velocity structure of the atmosphere [m/s]. Default: 0 m/s everywhere.
        ne : np.ndarray
            Electron density structure of the atmosphere [m-3].
        hydrogen_pops : np.ndarray, optional
            Detailed (per level) hydrogen number density structure of the
            atmosphere [m-3], 2D array [Nlevel, Nspace].
        nh_tot : np.ndarray, optional
            Total hydrogen number density structure of the atmosphere [m-3]
        vx : np.ndarray, optional
            x-component of atmospheric velocity [m/s]. If specifying vx/vy then a 3D
            quadrature will be needed.
        vy : np.ndarray, optional
            y-component of atmospheric velocity [m/s]. If specifying vx/vy then a 3D
            quadrature will be needed.
        vz : np.ndarray, optional
            alias for vlos. [m/s]
        B : np.ndarray, optional.
            Magnetic field strength [T].
        gamma_B : np.ndarray, optional
            Co-altitude of magnetic field vector [radians].
        chi_B : np.ndarray, optional
            Azimuth of magnetic field vector (in x-y plane, from x) [radians].
        lower_bc : BoundaryCondition, optional
            Boundary condition for incoming radiation at the minimal z
            coordinate (default: ThermalisedRadiation).
        upper_bc : BoundaryCondition, optional
            Boundary condition for incoming radiation at the maximal z
            coordinate (default: ZeroRadiation).
        convert_scales : bool, optional
            Whether to automatically compute tau_ref and cmass for an
            atmosphere given in a stratification of m (default: True).
        abundance: AtomicAbundance, optional
            An instance of AtomicAbundance giving the abundances of each
            atomic species in the given atmosphere, only used if the EOS is
            invoked. (default: DefaultAtomicAbundance)
        log_g: float, optional
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
        """
        if scale == ScaleType.Geometric:
            depth_scale = (depth_scale << u.m).value
            if np.any((depth_scale[:-1] - depth_scale[1:]) < 0.0):
                raise ValueError('Geometric depth scale should be provided in decreasing height.')
        elif scale == ScaleType.ColumnMass:
            depth_scale = (depth_scale << u.kg / u.m**2).value
            if np.any((depth_scale[1:] - depth_scale[:-1]) < 0.0):
                raise ValueError(
                    'Column mass depth scale should be provided in increasing column mass.'
                )

        def check_shape(x, x_name):
            return check_shape_exception(x, depth_scale.shape[0], 1, x_name)

        temperature = (temperature << u.K).value
        check_shape(temperature, 'temperature')
        if vlos is None:
            if vz is None:
                vlos = np.zeros_like(temperature)
            else:
                vlos = vz
        elif vz is not None:
            raise ValueError('Cannot set both vlos and vz (they are aliases).')
        vlos = (vlos << u.m / u.s).value
        check_shape(vlos, 'vlos')
        vz = vlos
        if vturb is None:
            vturb = np.zeros_like(temperature)
        vturb = (vturb << u.m / u.s).value
        check_shape(vturb, 'vturb')
        if ne is not None:
            ne = (ne << u.m ** (-3)).value
            check_shape(ne, 'ne')
        if hydrogen_pops is not None:
            hydrogen_pops = (hydrogen_pops << u.m ** (-3)).value
            hydrogen_pops = cast(np.ndarray, hydrogen_pops)
            if hydrogen_pops.shape[1] != depth_scale.shape[0]:
                raise ValueError(
                    f'Array hydrogen_pops does not have the expected'
                    f' second dimension: {depth_scale.shape[0]}'
                    f' (got: {hydrogen_pops.shape[1]}).'
                )
        if nh_tot is not None:
            nh_tot = (nh_tot << u.m ** (-3)).value
            check_shape(nh_tot, 'nh_tot')
        if vx is not None:
            if vy is None:
                raise ValueError('vx is set, vy must be also.')
            vx = (vx << u.m / u.s).value
            check_shape(vx, 'vx')
        if vy is not None:
            if vx is None:
                raise ValueError('vy is set, vx must be also.')
            vy = (vy << u.m / u.s).value
            check_shape(vy, 'vy')

        if B is not None:
            B = (B << u.T).value
            check_shape(B, 'B')
            if gamma_B is None or chi_B is None:
                raise ValueError('B is set, both gamma_B and chi_B must be also.')
        if gamma_B is not None:
            gamma_B = (gamma_B << u.rad).value
            check_shape(gamma_B, 'gamma_B')
            if B is None or chi_B is None:
                raise ValueError('gamma_B is set, both B and chi_B must be also.')
        if chi_B is not None:
            chi_B = (chi_B << u.rad).value
            check_shape(chi_B, 'chi_B')
            if gamma_B is None or B is None:
                raise ValueError('chi_B is set, both B and gamma_B must be also.')

        if lower_bc is None:
            lower_bc = ThermalisedRadiation()
        elif isinstance(lower_bc, PeriodicRadiation):
            raise ValueError('Cannot set periodic boundary conditions for 1D atmosphere')
        if upper_bc is None:
            upper_bc = ZeroRadiation()
        elif isinstance(upper_bc, PeriodicRadiation):
            raise ValueError('Cannot set periodic boundary conditions for 1D atmosphere')

        if scale != ScaleType.Geometric and not convert_scales:
            raise ValueError('Height scale must be provided if scale conversion is not applied')

        if nh_tot is None and hydrogen_pops is not None:
            nh_tot = np.sum(hydrogen_pops, axis=0)

        if np.any(temperature < 2000):
            # NOTE(cmo): Minimum value was decreased in NICOLE so should be safe
            raise ValueError('Minimum temperature too low for EOS (< 2000 K)')

        if abundance is None:
            abundance = DefaultAtomicAbundance

        wittAbundances = np.array([abundance[e] for e in PeriodicTable.elements])
        eos = Wittmann(abund_init=wittAbundances)

        Nspace = depth_scale.shape[0]
        if nh_tot is None and ne is not None:
            if verbose:
                print('Setting nh_tot from electron pressure.')
            pe = (ne << u.Unit('m-3')).to('cm-3').value * cgs.BK * temperature
            rho = np.zeros(Nspace)
            for k in range(Nspace):
                rho[k] = eos.rho_from_pe(temperature[k], pe[k])
            nh_tot = np.copy(
                (rho << u.Unit('g cm-3')).to('kg m-3').value / (Const.Amu * abundance.mass_per_h)
            )
        elif ne is None and nh_tot is not None:
            if verbose:
                print('Setting ne from mass density.')
            rho = (
                ((Const.Amu * abundance.mass_per_h * nh_tot) << u.Unit('kg m-3')).to('g cm-3').value
            )
            pe = np.zeros(Nspace)
            for k in range(Nspace):
                pe[k] = eos.pe_from_rho(temperature[k], rho[k])
            ne = np.copy(((pe / (cgs.BK * temperature)) << u.Unit('cm-3')).to('m-3').value)
        elif ne is None and nh_tot is None:
            if Pgas is not None and Pgas.shape[0] != Nspace:
                raise ValueError('Dimensions of Pgas do not match atmospheric depth')
            if Pe is not None and Pe.shape[0] != Nspace:
                raise ValueError('Dimensions of Pe do not match atmospheric depth')

            if Pgas is not None and Pe is None:
                if verbose:
                    print('Setting ne, nh_tot from provided gas pressure.')
                # Convert to cgs for eos
                pgas = (Pgas << u.Unit('Pa')).to('dyn cm-2').value
                pe = np.zeros(Nspace)
                rho = np.zeros(Nspace)
                for k in range(Nspace):
                    pe[k] = eos.pe_from_pg(temperature[k], pgas[k])
                    rho[k] = eos.rho_from_pg(temperature[k], pgas[k])
            elif Pe is not None and Pgas is None:
                if verbose:
                    print('Setting ne, nh_tot from provided electron pressure.')
                # Convert to cgs for eos
                pe = (Pe << u.Unit('Pa')).to('dyn cm-2').value
                pgas = np.zeros(Nspace)
                rho = np.zeros(Nspace)
                for k in range(Nspace):
                    pgas[k] = eos.pg_from_pe(temperature[k], pe[k])
                    rho[k] = eos.rho_from_pe(temperature[k], pe[k])
            elif Pgas is None and Pe is None:
                # Doing Hydrostatic Eq. based here on NICOLE implementation
                gravAcc = ((10**log_g) << u.Unit('m s-2')).to('cm s-2').value
                Avog = 6.022045e23  # Avogadro's Number
                if Ptop is None and Pe_top is not None:
                    if verbose:
                        print(
                            (
                                'Setting ne, nh_tot to hydrostatic equilibrium (log_g=%f)'
                                ' from provided top electron pressure.'
                            )
                            % log_g
                        )
                    Pe_top = (Pe_top << u.Unit('Pa')).to('dyn cm-2').value
                    Ptop = eos.pg_from_pe(temperature[0], Pe_top)
                elif Ptop is not None and Pe_top is None:
                    if verbose:
                        print(
                            (
                                'Setting ne, nh_tot to hydrostatic equilibrium (log_g=%f)'
                                ' from provided top gas pressure.'
                            )
                            % log_g
                        )
                    Ptop = (Ptop << u.Unit('Pa')).to('dyn cm-2').value
                    Pe_top = eos.pe_from_pg(temperature[0], Ptop)
                elif Ptop is None and Pe_top is None:
                    if verbose:
                        print(
                            (
                                'Setting ne, nh_tot to hydrostatic equilibrium (log_g=%f)'
                                ' from FALC gas pressure at upper boundary temperature.'
                            )
                            % log_g
                        )
                    Ptop = get_top_pressure(eos, temperature[0])
                    Pe_top = eos.pe_from_pg(temperature[0], Ptop)
                else:
                    raise ValueError('Cannot set both Ptop and Pe_top')

                if scale == ScaleType.Tau500:
                    tau = depth_scale
                elif scale == ScaleType.Geometric:
                    height = (depth_scale << u.Unit('m')).to('cm').value
                else:
                    cmass = (depth_scale << u.Unit('kg m-2')).to('g cm-2').value

                # NOTE(cmo): Compute HSE following the NICOLE method.
                rho = np.zeros(Nspace)
                chi_c = np.zeros(Nspace)
                pgas = np.zeros(Nspace)
                pe = np.zeros(Nspace)
                pgas[0] = Ptop
                pe[0] = Pe_top
                chi_c[0] = eos.cont_opacity(
                    temperature[0], pgas[0], pe[0], np.array([5000.0])
                ).item()

                def avg_mol_weight(k):
                    return abundance.mass_per_h / (abundance.total_abundance + pe[k] / pgas[k])

                rho[0] = Ptop * avg_mol_weight(0) / Avog / cgs.BK / temperature[0]
                chi_c[0] /= rho[0]

                for k in range(1, Nspace):
                    chi_c[k] = chi_c[k - 1]
                    rho[k] = rho[k - 1]
                    for it in range(200):
                        if scale == ScaleType.Tau500:
                            dtau = tau[k] - tau[k - 1]
                            pgas[k] = pgas[k - 1] + gravAcc * dtau / (
                                0.5 * (chi_c[k - 1] + chi_c[k])
                            )
                        elif scale == ScaleType.Geometric:
                            pgas[k] = pgas[k - 1] * np.exp(
                                -gravAcc
                                / Avog
                                / cgs.BK
                                * avg_mol_weight(k - 1)
                                * 0.5
                                * (1.0 / temperature[k - 1] + 1.0 / temperature[k])
                                * (height[k] - height[k - 1])
                            )
                        else:
                            pgas[k] = gravAcc * cmass[k]

                        pe[k] = eos.pe_from_pg(temperature[k], pgas[k])
                        prevChi = chi_c[k]
                        chi_c[k] = eos.cont_opacity(
                            temperature[k], pgas[k], pe[k], np.array([5000.0])
                        ).item()
                        rho[k] = pgas[k] * avg_mol_weight(k) / Avog / cgs.BK / temperature[k]
                        chi_c[k] /= rho[k]

                        change = np.abs(prevChi - chi_c[k]) / (prevChi + chi_c[k])
                        if change < 1e-5:
                            break
                    else:
                        raise ConvergenceError(
                            ('No convergence in HSE at depth point %d, last change %2.4e')
                            % (k, change)
                        )
            nh_tot = np.copy(
                (rho << u.Unit('g cm-3')).to('kg m-3').value / (Const.Amu * abundance.mass_per_h)
            )
            ne = np.copy(((pe / (cgs.BK * temperature)) << u.Unit('cm-3')).to('m-3').value)

        # NOTE(cmo): Compute final pgas, pe from EOS that will be used for
        # background opacity.
        rhoSI = Const.Amu * abundance.mass_per_h * nh_tot
        rho = (rhoSI << u.Unit('kg m-3')).to('g cm-3').value
        pgas = np.zeros_like(depth_scale)
        pe = np.zeros_like(depth_scale)
        for k in range(Nspace):
            pgas[k] = eos.pg_from_rho(temperature[k], rho[k])
            pe[k] = eos.pe_from_rho(temperature[k], rho[k])

        chi_c = np.zeros_like(depth_scale)
        for k in range(depth_scale.shape[0]):
            chi_c[k] = eos.cont_opacity(temperature[k], pgas[k], pe[k], np.array([5000.0])).item()
        chi_c = (chi_c << u.Unit('cm-1')).to('m-1').value

        # NOTE(cmo): We should now have a uniform minimum set of data (other
        # than the scale type), allowing us to simply convert between the
        # scales we do have!
        if convert_scales:
            if scale == ScaleType.ColumnMass:
                height = np.zeros_like(depth_scale)
                tau_ref = np.zeros_like(depth_scale)
                cmass = depth_scale

                height[0] = 0.0
                tau_ref[0] = chi_c[0] / rhoSI[0] * cmass[0]
                for k in range(1, cmass.shape[0]):
                    height[k] = height[k - 1] - 2.0 * (
                        (cmass[k] - cmass[k - 1]) / (rhoSI[k - 1] + rhoSI[k])
                    )
                    tau_ref[k] = tau_ref[k - 1] + 0.5 * (
                        (chi_c[k - 1] + chi_c[k]) * (height[k - 1] - height[k])
                    )

                hTau1 = np.interp(1.0, tau_ref, height)
                height -= hTau1
            elif scale == ScaleType.Geometric:
                cmass = np.zeros(Nspace)
                tau_ref = np.zeros(Nspace)
                height = depth_scale
                nh_tot = cast(np.ndarray, nh_tot)
                ne = cast(np.ndarray, ne)

                cmass[0] = (nh_tot[0] * abundance.total_abundance + ne[0]) * (
                    Const.KBoltzmann * temperature[0] / 10**log_g
                )
                tau_ref[0] = 0.5 * chi_c[0] * (height[0] - height[1])
                if tau_ref[0] > 1.0:
                    tau_ref[0] = 0.0

                for k in range(1, Nspace):
                    cmass[k] = cmass[k - 1] + 0.5 * (
                        (rhoSI[k - 1] + rhoSI[k]) * (height[k - 1] - height[k])
                    )
                    tau_ref[k] = tau_ref[k - 1] + 0.5 * (
                        (chi_c[k - 1] + chi_c[k]) * (height[k - 1] - height[k])
                    )
            elif scale == ScaleType.Tau500:
                cmass = np.zeros(Nspace)
                height = np.zeros(Nspace)
                tau_ref = depth_scale

                cmass[0] = (tau_ref[0] / chi_c[0]) * rhoSI[0]
                for k in range(1, Nspace):
                    height[k] = height[k - 1] - 2.0 * (
                        (tau_ref[k] - tau_ref[k - 1]) / (chi_c[k - 1] + chi_c[k])
                    )
                    cmass[k] = cmass[k - 1] + 0.5 * (
                        (rhoSI[k - 1] + rhoSI[k]) * (height[k - 1] - height[k])
                    )

                hTau1 = np.interp(1.0, tau_ref, height)
                height -= hTau1
            else:
                raise ValueError('Other scales not handled yet')

            stratifications: Optional[Stratifications] = Stratifications(
                cmass=cmass, tau_ref=tau_ref
            )

        else:
            stratifications = None
            height = depth_scale

        layout = Layout.make_1d(
            z=height,
            vx=vx,
            vy=vy,
            vz=vz,
            lower_bc=lower_bc,
            upper_bc=upper_bc,
            stratifications=stratifications,
        )
        ne = cast(np.ndarray, ne)
        nh_tot = cast(np.ndarray, nh_tot)
        atmos = cls(
            structure=layout,
            temperature=temperature,
            vturb=vturb,
            ne=ne,
            nh_tot=nh_tot,
            B=B,
            gamma_B=gamma_B,
            chi_B=chi_B,
        )

        return atmos

    @classmethod
    def make_2d(
        cls,
        height: np.ndarray,
        x: np.ndarray,
        temperature: np.ndarray,
        vx: Optional[np.ndarray] = None,
        vy: Optional[np.ndarray] = None,
        vz: Optional[np.ndarray] = None,
        vturb: Optional[np.ndarray] = None,
        ne: Optional[np.ndarray] = None,
        nh_tot: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        gamma_B: Optional[np.ndarray] = None,
        chi_B: Optional[np.ndarray] = None,
        x_upper_bc: Optional[BoundaryCondition] = None,
        x_lower_bc: Optional[BoundaryCondition] = None,
        z_upper_bc: Optional[BoundaryCondition] = None,
        z_lower_bc: Optional[BoundaryCondition] = None,
        abundance: Optional[AtomicAbundance] = None,
        verbose=False,
    ):
        """
        Constructor for 2D Atmosphere objects.

        No provision for estimating parameters using hydrostatic equilibrium
        is provided, but one of ne, or nh_tot can be omitted and inferred by
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
        vx : np.ndarray, optional.
            x-component of the atmospheric velocity [m/s]. Default: 0 m/s.
        vy : np.ndarray, optional.
            y-component of the atmospheric velocity [m/s]. Not used by default,
            use a 3D quadrature if you need it.
        vz : np.ndarray, optional
            z-component of the atmospheric velocity [m/s]. Default: 0 m/s.
        vturb : np.ndarray
            Microturbulent velocity structure [m/s].
        ne : np.ndarray
            Electron density structure of the atmosphere [m-3].
        nh_tot : np.ndarray, optional
            Total hydrogen number density structure of the atmosphere [m-3].
        B : np.ndarray, optional.
            Magnetic field strength [T].
        gamma_B : np.ndarray, optional
            Inclination (co-altitude) of magnetic field vector to the z-axis
            [radians].
        chi_B : np.ndarray, optional
            Azimuth of magnetic field vector (in x-y plane, from x) [radians].
        x_lower_bc : BoundaryCondition, optional
            Boundary condition for incoming radiation at the minimal x
            coordinate (default: PeriodicRadiation).
        x_upper_bc : BoundaryCondition, optional
            Boundary condition for incoming radiation at the maximal x
            coordinate (default: PeriodicRadiation).
        z_lower_bc : BoundaryCondition, optional
            Boundary condition for incoming radiation at the minimal z
            coordinate (default: ThermalisedRadiation).
        z_upper_bc : BoundaryCondition, optional
            Boundary condition for incoming radiation at the maximal z
            coordinate (default: ZeroRadiation).
        convert_scales : bool, optional
            Whether to automatically compute tau_ref and cmass for an
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
        """

        x = (x << u.m).value
        if np.any((x[1:] - x[:-1]) < 0.0):
            raise ValueError('x should be increasing with index (left -> right).')
        height = (height << u.m).value
        if np.any((height[:-1] - height[1:]) < 0.0):
            raise ValueError('Height should be decreasing with index (top -> bottom).')
        temperature = (temperature << u.K).value
        if vx is None:
            vx = np.zeros_like(temperature)
        vx = (vx << u.m / u.s).value
        if vy is not None:
            vy = (vy << u.m / u.s).value
        if vz is None:
            vz = np.zeros_like(temperature)
        vz = (vz << u.m / u.s).value
        if vturb is None:
            vturb = np.zeros_like(temperature)
        vturb = (vturb << u.m / u.s).value
        if ne is not None:
            ne = (ne << u.m ** (-3)).value
        if nh_tot is not None:
            nh_tot = (nh_tot << u.m ** (-3)).value
        if B is not None:
            B = (B << u.T).value
            B = cast(np.ndarray, B)
            flatB = view_flatten(B)
        else:
            flatB = None

        if gamma_B is not None:
            gamma_B = (gamma_B << u.rad).value
            gamma_B = cast(np.ndarray, gamma_B)
            flatGammaB = view_flatten(gamma_B)
        else:
            flatGammaB = None

        if chi_B is not None:
            chi_B = (chi_B << u.rad).value
            chi_B = cast(np.ndarray, chi_B)
            flatChiB = view_flatten(chi_B)
        else:
            flatChiB = None

        if z_lower_bc is None:
            z_lower_bc = ThermalisedRadiation()
        elif isinstance(z_lower_bc, PeriodicRadiation):
            raise ValueError('Cannot set periodic boundary conditions for z-axis.')
        if z_upper_bc is None:
            z_upper_bc = ZeroRadiation()
        elif isinstance(z_upper_bc, PeriodicRadiation):
            raise ValueError('Cannot set periodic boundary conditions for z-axis.')
        if x_upper_bc is None:
            x_upper_bc = PeriodicRadiation()
        if x_lower_bc is None:
            x_lower_bc = PeriodicRadiation()
        if abundance is None:
            abundance = DefaultAtomicAbundance

        wittAbundances = np.array([abundance[e] for e in PeriodicTable.elements])
        eos = Wittmann(abund_init=wittAbundances)

        flatHeight = view_flatten(height)
        flatTemperature = view_flatten(temperature)
        Nspace = flatTemperature.shape[0]
        if nh_tot is None and ne is not None:
            if verbose:
                print('Setting nh_tot from electron pressure.')
            flatNe = view_flatten(ne)
            pe = (flatNe << u.Unit('m-3')).to('cm-3').value * cgs.BK * flatTemperature
            rho = np.zeros(Nspace)
            for k in range(Nspace):
                rho[k] = eos.rho_from_pe(flatTemperature[k], pe[k])
            nh_tot = np.ascontiguousarray(
                (rho << u.Unit('g cm-3')).to('kg m-3').value / (Const.Amu * abundance.mass_per_h)
            )
        elif ne is None and nh_tot is not None:
            if verbose:
                print('Setting ne from mass density.')
            flatNHTot = view_flatten(nh_tot)
            rho = (
                ((Const.Amu * abundance.mass_per_h * flatNHTot) << u.Unit('kg m-3'))
                .to('g cm-3')
                .value
            )
            pe = np.zeros(Nspace)
            for k in range(Nspace):
                pe[k] = eos.pe_from_rho(flatTemperature[k], rho[k])
            ne = np.ascontiguousarray(
                ((pe / (cgs.BK * flatTemperature)) << u.Unit('cm-3')).to('m-3').value
            )
        elif ne is None and nh_tot is None:
            raise ValueError('Cannot omit both ne and nh_tot (currently).')
        flatX = view_flatten(x)
        nh_tot = cast(np.ndarray, nh_tot)
        flatNHTot = view_flatten(nh_tot)
        ne = cast(np.ndarray, ne)
        flatNe = view_flatten(ne)
        flatVx = view_flatten(vx)
        flatVy = None if vy is None else view_flatten(vy)
        flatVz = view_flatten(vz)
        flatVturb = view_flatten(vturb)

        layout = Layout.make_2d(
            x=flatX,
            z=flatHeight,
            vx=flatVx,
            vy=flatVy,
            vz=flatVz,
            x_lower_bc=x_lower_bc,
            x_upper_bc=x_upper_bc,
            z_lower_bc=z_lower_bc,
            z_upper_bc=z_upper_bc,
            stratifications=None,
        )

        atmos = cls(
            structure=layout,
            temperature=flatTemperature,
            vturb=flatVturb,
            ne=flatNe,
            nh_tot=flatNHTot,
            B=flatB,
            gamma_B=flatGammaB,
            chi_B=flatChiB,
        )
        return atmos

    def quadrature(
        self,
        Nrays: Optional[int] = None,
        mu: Optional[Sequence[float]] = None,
        wmu: Optional[Sequence[float]] = None,
        force3d: bool = False,
    ):
        """
        Compute the angular quadrature for solving the RTE and Kinetic
        Equilibrium in a given atmosphere.

        Procedure varies with dimensionality.

        By convention muz is always positive, as the direction on this axis
        is determined by the to_obs term that is used internally to the formal
        solver.

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
            The number of rays to use in the quadrature (per octant). See notes
            above.
        mu : sequence of float, optional
            The cosine of the angle made between the between each of the set
            of rays and the z axis, only used in 1D.
        wmu : sequence of float, optional
            The integration weights for each mu, must be provided if mu is provided.
        force3d : bool, optional
            Force the use of a 3D quadrature. Default: False.

        Raises
        ------
        ValueError
            on incorrect input.
        """

        n_dim_effective = self.Ndim if not force3d else 3
        # NOTE(cmo): Catch the case where we need a 3d quadrature in a less than 3d atmosphere.
        if self.structure.vx.size > 0 and self.structure.vy.size > 0:
            n_dim_effective = 3
        if n_dim_effective == 1:
            if mu is not None:
                if wmu is None:
                    raise ValueError('Must provide wmu when providing mu')
                if Nrays is not None and Nrays != len(mu):
                    raise ValueError('mu must be Nrays long if Nrays is provided')
                if len(mu) != len(wmu):
                    raise ValueError('mu and wmu must be the same shape')

                self.muz = np.array(mu, dtype=np.float64)
                self.wmu = np.array(wmu, dtype=np.float64)
            elif Nrays is not None:
                if Nrays >= 1:
                    x, w = leggauss(Nrays)
                    mid, halfWidth = 0.5, 0.5
                    x = mid + halfWidth * x
                    w *= halfWidth

                    # NOTE(cmo): muz is in [0, 1], i.e. the upward directed
                    # rays. The downward rays are handled by up/down/
                    self.muz = x
                    self.wmu = w
                else:
                    raise ValueError('Unsupported Nrays=%d' % Nrays)
            else:
                raise ValueError('Must provide either Nrays, or mu and wmu')

            self.muy = np.zeros_like(self.muz)
            self.mux = np.sqrt(1.0 - self.muz**2)
        else:
            with open(get_data_path() + 'Quadratures.pickle', 'rb') as pkl:
                quads = pickle.load(pkl)

            rays = {int(q.split('n')[1]): q for q in quads}
            if Nrays not in rays:
                raise ValueError('For multidimensional cases Nrays must be in %s' % repr(rays))

            quad = quads[rays[Nrays]]

            if n_dim_effective == 2:
                Nrays *= 2
                theta = np.deg2rad(quad[:, 1])
                chi = np.deg2rad(quad[:, 2])
                # polar coords:
                # x = sin theta cos chi
                # y = sin theta sin chi
                # z = cos theta
                # Fill first half then flip in x-z plane. This solves for 2 octants.
                # As always, we keep muz always positive in the atmosphere definition.
                self.mux = np.zeros(Nrays)
                self.mux[: Nrays // 2] = np.sin(theta) * np.cos(chi)
                self.mux[Nrays // 2 :] = -np.sin(theta) * np.cos(chi)
                self.muz = np.zeros(Nrays)
                self.muz[: Nrays // 2] = np.cos(theta)
                self.muz[Nrays // 2 :] = np.cos(theta)
                self.wmu = np.zeros(Nrays)
                self.wmu[: Nrays // 2] = quad[:, 0]
                self.wmu[Nrays // 2 :] = quad[:, 0]
                self.wmu /= np.sum(self.wmu)
                self.muy = np.sqrt(1.0 - (self.mux**2 + self.muz**2))
            else:
                # n_dim_effective == 3
                Nrays *= 4
                theta = np.deg2rad(quad[:, 1])
                chi = np.deg2rad(quad[:, 2])
                # polar coords:
                # x = sin theta cos chi
                # y = sin theta sin chi
                # z = cos theta
                # Fill first quarter then flip in x-z plane for second quarter.
                # Then flip in y-z plane for second half. This solves for 4 octants.
                # As always, we keep muz always positive in the atmosphere definition.
                self.mux = np.zeros(Nrays)
                self.mux[: Nrays // 4] = np.sin(theta) * np.cos(chi)
                self.mux[Nrays // 4 : Nrays // 2] = -np.sin(theta) * np.cos(chi)
                self.mux[Nrays // 2 :] = -self.mux[: Nrays // 2]
                self.muy = np.zeros(Nrays)
                self.muy[: Nrays // 4] = np.sin(theta) * np.sin(chi)
                self.muy[Nrays // 4 : Nrays // 2] = np.sin(theta) * np.sin(chi)
                self.muy[Nrays // 2 :] = -self.muy[: Nrays // 2]

                self.wmu = np.zeros(Nrays)
                self.wmu[: Nrays // 4] = quad[:, 0]
                self.wmu[Nrays // 4 : Nrays // 2] = quad[:, 0]
                self.wmu[Nrays // 2 :] = self.wmu[: Nrays // 2]

                self.wmu /= np.sum(self.wmu)
                self.muz = np.sqrt(1.0 - (self.mux**2 + self.muy**2))

        self.configure_bcs()

    def rays(
        self,
        muz: Union[float, Sequence[float]],
        mux: Optional[Union[float, Sequence[float]]] = None,
        muy: Optional[Union[float, Sequence[float]]] = None,
        wmu: Optional[Union[float, Sequence[float]]] = None,
        up_only: bool = False,
    ):
        """
        Set up the rays on the Atmosphere for computing the intensity in a
        particular direction (or set of directions).

        If only the z angle is set then the ray is assumed in the x-z plane.
        If either muz or muy is omitted then this angle is inferred by
        normalisation of the projection.

        By convention muz is always positive, as the direction on this axis
        is determined by the to_obs term that is used internally to the formal
        solver.

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
        up_only : bool, optional
            Whether to only configure boundary conditions for up-only rays.
            (default: False)

        Raises
        ------
        ValueError
            if the angular projections or integration weights are incorrectly
            normalised.
        """

        if isinstance(muz, numbers.Real):
            muz = [float(muz)]
        if isinstance(mux, numbers.Real):
            mux = [float(mux)]
        if isinstance(muy, numbers.Real):
            muy = [float(muy)]
        if isinstance(wmu, numbers.Real):
            wmu = [float(wmu)]

        if mux is None and muy is None:
            self.muz = np.array(muz, dtype=np.float64)
            self.wmu = np.zeros_like(self.muz)
            self.muy = np.zeros_like(self.muz)
            self.mux = np.sqrt(1.0 - self.muz**2)
        elif muy is None:
            self.muz = np.array(muz, dtype=np.float64)
            self.wmu = np.zeros_like(self.muz)
            self.mux = np.array(mux, dtype=np.float64)
            self.muy = np.sqrt(1.0 - (self.muz**2 + self.mux**2))
        elif mux is None:
            self.muz = np.array(muz, dtype=np.float64)
            self.wmu = np.zeros_like(self.muz)
            self.muy = np.array(muy, dtype=np.float64)
            self.mux = np.sqrt(1.0 - (self.muz**2 + self.muy**2))
        else:
            self.muz = np.array(muz, dtype=np.float64)
            self.mux = np.array(mux, dtype=np.float64)
            self.muy = np.array(muy, dtype=np.float64)
            self.wmu = np.zeros_like(muz)

            if not np.allclose(self.muz**2 + self.mux**2 + self.muy**2, 1):
                raise ValueError('mux**2 + muy**2 + muz**2 != 1.0')

        if not np.all(self.muz > 0):
            raise ValueError('muz must be > 0')

        if wmu is not None:
            self.wmu = np.array(wmu, dtype=np.float64)

            if not np.isclose(self.wmu.sum(), 1.0):
                raise ValueError('sum of wmus is not 1.0')

        self.configure_bcs(up_only=up_only)

    def configure_bcs(self, up_only: bool = False):
        """
        Configure the required angular information for all boundary
        conditions on the model.

        Parameters
        ----------
        up_only : bool, optional
            Whether to only configure boundary conditions for up-going rays.
            (default: False)
        """

        # NOTE(cmo): We always have z-bcs
        # For z_lower_bc, muz is positive, and we have all mux, muz
        mux, muy, muz = self.mux, self.muy, self.muz
        # NOTE(cmo): index_vector is of shape (mu, to_obs) to allow the core to
        # easily destructure the blob that will be handed to it from
        # compute_bc.
        index_vector = np.ones((self.mux.shape[0], 2), dtype=np.int32) * -1
        index_vector[:, 1] = np.arange(mux.shape[0])
        self.z_lower_bc.set_required_angles(mux, muy, muz, index_vector)

        index_vector = np.ones((mux.shape[0], 2), dtype=np.int32) * -1
        if not up_only:
            index_vector[:, 0] = np.arange(mux.shape[0])
        self.z_upper_bc.set_required_angles(-mux, -muy, -muz, index_vector)

        toObsRange = [0, 1]
        if up_only:
            toObsRange = [1]

        # NOTE(cmo): If 2+D we have x-bcs too
        # x_lower_bc has all muz and all mux > 0
        mux, muy, muz = [], [], []
        index_vector = np.ones((self.mux.shape[0], 2), dtype=np.int32) * -1
        count = 0
        musDone = np.zeros(self.muz.shape[0], dtype=np.bool_)
        for mu in range(self.muz.shape[0]):
            for equalMu in np.argwhere(np.abs(self.muz) == self.muz[mu]).reshape(-1)[::-1]:
                if musDone[equalMu]:
                    continue
                musDone[equalMu] = True

                for toObsI in toObsRange:
                    sign = [-1, 1][toObsI]
                    sMux = sign * self.mux[equalMu]
                    if sMux > 0:
                        mux.append(sMux)
                        muy.append(sign * self.muy[equalMu])
                        muz.append(sign * self.muz[equalMu])
                        index_vector[equalMu, toObsI] = count
                        count += 1
            if np.all(musDone):
                break

        mux = np.array(mux)
        muy = np.array(muy)
        muz = np.array(muz)
        self.x_lower_bc.set_required_angles(mux, muy, muz, index_vector)

        mux, muy, muz = [], [], []
        index_vector = np.ones((self.mux.shape[0], 2), dtype=np.int32) * -1
        count = 0
        musDone = np.zeros(self.muz.shape[0], dtype=np.bool_)
        for mu in range(self.muz.shape[0]):
            for equalMu in np.argwhere(np.abs(self.muz) == self.muz[mu]).reshape(-1):
                if musDone[equalMu]:
                    continue
                musDone[equalMu] = True

                for toObsI in toObsRange:
                    sign = [-1, 1][toObsI]
                    sMux = sign * self.mux[equalMu]
                    if sMux < 0:
                        mux.append(sMux)
                        muy.append(sign * self.muy[equalMu])
                        muz.append(sign * self.muz[equalMu])
                        index_vector[equalMu, toObsI] = count
                        count += 1
            if np.all(musDone):
                break

        mux = np.array(mux)
        muy = np.array(muy)
        muz = np.array(muz)
        self.x_upper_bc.set_required_angles(mux, muy, muz, index_vector)

        self.y_lower_bc.set_required_angles(
            np.zeros((0)),
            np.zeros((0)),
            np.zeros((0)),
            np.ones((self.mux.shape[0], 2), dtype=np.int32) * -1,
        )
        self.y_upper_bc.set_required_angles(
            np.zeros((0)),
            np.zeros((0)),
            np.zeros((0)),
            np.ones((self.mux.shape[0], 2), dtype=np.int32) * -1,
        )

        if self.Ndim > 2:
            raise ValueError('Only <= 2D atmospheres supported currently.')
