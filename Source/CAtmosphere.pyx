import numpy as np
cimport numpy as np
from CmoArray cimport *
# from CmoArrayHelper cimport *
from libcpp cimport bool as bool_t
from libcpp.vector cimport vector
from libc.math cimport sqrt, exp, copysign
from .atmosphere import BoundaryCondition, ZeroRadiation, ThermalisedRadiation, PeriodicRadiation, NoBc
from .atomic_model import AtomicLine, LineType, LineProfileState
from .utils import InitialSolution, ExplodingMatrixError, UnityCrswIterator
from weno4 import weno4
import lightweaver.constants as Const
import time
from enum import Enum, auto
from copy import copy, deepcopy

include 'CmoArrayHelper.pyx'

# NOTE(cmo): Some late binding stuff to be able to use numpy C API
np.import_array()

ctypedef np.int8_t i8
# ctypedef Array1NonOwn[np.int32_t] I32View
ctypedef Array1NonOwn[np.int32_t] I32View
ctypedef Array1NonOwn[bool_t] BoolView

cdef extern from "LwFormalInterface.hpp":
    cdef cppclass FormalSolver:
        int Ndim
        int width
        const char* name;

    cdef cppclass FormalSolverManager:
        vector[FormalSolver] formalSolvers;
        bool_t load_fs_from_path(const char* path)

    cdef cppclass InterpFn:
        int Ndim
        const char* name
        InterpFn()

    cdef cppclass InterpFnManager:
        vector[InterpFn] fns
        bool_t load_fn_from_path(const char* path)

cdef extern from "Lightweaver.hpp":
    cdef enum RadiationBc:
        UNINITIALISED
        ZERO
        THERMALISED
        PERIODIC
        CALLABLE

    cdef cppclass AtmosphericBoundaryCondition:
        RadiationBc type
        F64Arr2D bcData

        AtmosphericBoundaryCondition()
        AtmosphericBoundaryCondition(RadiationBc typ, int Nwave, int Nspace)
        void set_bc_data(F64View2D data)

    cdef cppclass Atmosphere:
        int Nspace
        int Nrays
        int Ndim
        int Nx
        int Ny
        int Nz
        F64View x
        F64View y
        F64View z
        F64View height
        F64View temperature
        F64View ne
        F64View vx
        F64View vy
        F64View vz
        F64View2D vlosMu
        F64View B
        F64View gammaB
        F64View chiB
        F64View2D cosGamma
        F64View2D cos2chi
        F64View2D sin2chi
        F64View vturb
        F64View nHTot
        F64View muz
        F64View muy
        F64View mux
        F64View wmu

        AtmosphericBoundaryCondition xLowerBc
        AtmosphericBoundaryCondition xUpperBc
        AtmosphericBoundaryCondition yLowerBc
        AtmosphericBoundaryCondition yUpperBc
        AtmosphericBoundaryCondition zLowerBc
        AtmosphericBoundaryCondition zUpperBc

        void update_projections()
    cdef void build_intersection_list(Atmosphere* atmos)

cdef extern from "Lightweaver.hpp" namespace "PrdCores":
    cdef int max_fine_grid_size()

cdef extern from "Lightweaver.hpp" namespace "Prd":
    cdef cppclass PrdStorage:
        F64Arr3D gII

cdef extern from "Background.hpp":
    cdef cppclass BackgroundData:
        F64View chPops
        F64View ohPops
        F64View h2Pops
        F64View hMinusPops
        F64View2D hPops

        F64View wavelength
        F64View2D chi
        F64View2D eta
        F64View2D scatt

    cdef void basic_background(BackgroundData* bg, Atmosphere* atmos)
    cdef f64 Gaunt_bf(f64, f64, int)

cdef extern from "Ng.hpp":
    cdef cppclass Ng:
        int Norder
        int Nperiod
        int Ndelay
        bool_t init
        Ng()
        Ng(int nOrder, int nPeriod, int nDelay, F64View sol)
        bool_t accelerate(F64View sol)
        f64 max_change()
        void clear()

cdef extern from "Lightweaver.hpp":
    cdef cppclass Background:
        F64View2D chi
        F64View2D eta
        F64View2D sca

    cdef cppclass Spectrum:
        F64View wavelength
        F64View3D I
        F64View4D Quv
        F64View2D J
        F64Arr2D JRest

    cdef cppclass ZeemanComponents:
        I32View alpha
        F64View shift
        F64View strength

    cdef enum TransitionType:
        LINE
        CONTINUUM

    cdef cppclass Transition:
        TransitionType type
        f64 Aji
        f64 Bji
        f64 Bij
        f64 lambda0
        f64 dopplerWidth
        int Nblue
        int i
        int j
        F64View wavelength
        F64View gij
        F64View alpha
        F64View4D phi
        F64View wphi
        bool_t polarised
        F64View4D phiQ
        F64View4D phiU
        F64View4D phiV
        F64View4D psiQ
        F64View4D psiU
        F64View4D psiV
        F64View Qelast
        F64View aDamp
        BoolView active

        F64View Rij
        F64View Rji
        F64View2D rhoPrd
        F64View3D gII
        PrdStorage prdStorage

        void uv(int la, int mu, bool_t toObs, F64View Uji, F64View Vij, F64View Vji)
        void compute_phi(const Atmosphere& atmos, F64View aDamp, F64View vBroad)
        void compute_wphi(const Atmosphere& atmos)
        void compute_polarised_profiles(const Atmosphere& atmos, F64View aDamp, F64View vBroad, const ZeemanComponents& z) except +

    cdef cppclass Atom:
        Atmosphere* atmos
        F64View2D n
        F64View2D nStar
        F64View vBroad
        F64View nTotal
        F64View stages

        F64View3D Gamma
        F64View3D C

        F64View eta
        F64View2D gij
        F64View2D wla
        F64View2D U
        F64View2D chi

        vector[Transition*] trans
        Ng ng

        int Nlevel
        int Ntrans
        void setup_wavelength(int la)

    cdef cppclass DepthData:
        bool_t fill
        F64View4D chi
        F64View4D eta

    cdef cppclass Context:
        Atmosphere* atmos
        Spectrum* spect
        vector[Atom*] activeAtoms
        vector[Atom*] detailedAtoms
        Background* background
        DepthData* depthData
        int Nthreads
        FormalSolver formalSolver
        InterpFn interpFn
        void initialise_threads()
        void update_threads()

    cdef cppclass PrdIterData:
        int iter
        f64 dRho

    cdef f64 formal_sol_gamma_matrices(Context& ctx)
    cdef f64 formal_sol_gamma_matrices(Context& ctx, bool_t lambdaIterate)
    cdef f64 formal_sol_update_rates(Context& ctx)
    cdef f64 formal_sol_update_rates_fixed_J(Context& ctx)
    cdef f64 formal_sol(Context& ctx)
    cdef f64 formal_sol_full_stokes(Context& ctx) except +
    cdef f64 formal_sol_full_stokes(Context& ctx, bool_t updateJ) except +
    cdef PrdIterData redistribute_prd_lines(Context& ctx, int maxIter, f64 tol)
    cdef void stat_eq(Atom* atom) except +
    cdef void time_dependent_update(Atom* atomIn, F64View2D nOld, f64 dt) except +
    cdef void configure_hprd_coeffs(Context& ctx)

cdef extern from "Lightweaver.hpp" namespace "EscapeProbability":
    cdef void gamma_matrices_escape_prob(Atom* a, Background& background, const Atmosphere& atmos)

cdef class LwDepthData:
    cdef object shape
    cdef DepthData depthData
    cdef f64[:,:,:,::1] chi
    cdef f64[:,:,:,::1] eta

    def __init__(self, Nlambda, Nmu, Nspace):
        self.shape = (Nlambda, Nmu, 2, Nspace)
        self.depthData.fill = 0

    def __getstate__(self):
        s = {}
        s['shape'] = self.shape
        s['fill'] = bool(self.fill)
        try:
            s['chi'] = np.copy(np.asarray(self.chi))
            s['eta'] = np.copy(np.asarray(self.eta))
        except AttributeError:
            s['chi'] = None
            s['eta'] = None

        return s

    def __setstate__(self, s):
        self.shape = s['shape']
        self.depthData.fill = int(s['fill'])
        if s['chi'] is not None:
            self.chi = s['chi']
            self.depthData.chi = f64_view_4(self.chi)
            self.eta = s['eta']
            self.depthData.eta = f64_view_4(self.eta)

    @property
    def fill(self):
        return bool(self.depthData.fill)

    @fill.setter
    def fill(self, value):
        try:
            self.depthData.fill = int(value)
            if value:
                self.chi
        except AttributeError:
            self.chi = np.zeros(self.shape)
            self.depthData.chi = f64_view_4(self.chi)
            self.eta = np.zeros(self.shape)
            self.depthData.eta = f64_view_4(self.eta)

    @property
    def chi(self):
        return np.asarray(self.chi)

    @property
    def eta(self):
        return np.asarray(self.eta)

def BC_to_enum(bc):
    if isinstance(bc, ZeroRadiation):
        return ZERO
    elif isinstance(bc, ThermalisedRadiation):
        return THERMALISED
    elif isinstance(bc, PeriodicRadiation):
        return PERIODIC
    elif isinstance(bc, NoBc):
        return UNINITIALISED
    elif isinstance(bc, BoundaryCondition):
        return CALLABLE
    else:
        raise ValueError('Argument is not a BoundaryCondition.')

cdef verify_bc_array_sizes(AtmosphericBoundaryCondition* abc, f64[:,::1] pyArr, str location):
    cdef int dim0 = abc.bcData.shape(0)
    cdef int dim1 = abc.bcData.shape(1)
    if dim0 != pyArr.shape[0] or dim1 != pyArr.shape[1]:
        raise ValueError('BC returned from python does not match expected shape for %s (%d, %d)' % (location, dim0, dim1))

cdef class LwAtmosphere:
    cdef Atmosphere atmos
    cdef f64[::1] x
    cdef f64[::1] y
    cdef f64[::1] z
    cdef f64[::1] temperature
    cdef f64[::1] ne
    cdef f64[::1] vx
    cdef f64[::1] vy
    cdef f64[::1] vz
    cdef f64[:,::1] vlosMu
    cdef f64[::1] B
    cdef f64[::1] gammaB
    cdef f64[::1] chiB
    cdef f64[:,::1] cosGamma
    cdef f64[:,::1] cos2chi
    cdef f64[:,::1] sin2chi
    cdef f64[::1] vturb
    cdef f64[::1] nHTot
    cdef f64[::1] muz
    cdef f64[::1] muy
    cdef f64[::1] mux
    cdef f64[::1] wmu
    # TODO(cmo): I don't really like storing Nwave here, but I don't know how
    # much of a choice we have.
    cdef int Nwave

    cdef public object pyAtmos

    def __init__(self, atmos, Nwavelengths):
        cdef int Nwave = Nwavelengths
        self.Nwave = Nwave
        self.pyAtmos = atmos

        self.x = atmos.x
        self.y = atmos.y
        self.z = atmos.z

        self.temperature = atmos.temperature
        self.ne = atmos.ne

        self.vz = atmos.vz
        self.vx = atmos.vx
        self.vy = atmos.vy

        self.vturb = atmos.vturb
        self.nHTot = atmos.nHTot
        self.muz = atmos.muz
        self.muy = atmos.muy
        self.mux = atmos.mux
        self.wmu = atmos.wmu
        self.atmos.z = f64_view(self.z)
        self.atmos.height = f64_view(self.z)
        self.atmos.x = f64_view(self.x)
        self.atmos.y = f64_view(self.y)
        self.atmos.temperature = f64_view(self.temperature)
        self.atmos.ne = f64_view(self.ne)
        self.atmos.vx = f64_view(self.vx)
        self.atmos.vy = f64_view(self.vy)
        self.atmos.vz = f64_view(self.vz)
        self.atmos.vturb = f64_view(self.vturb)
        self.atmos.nHTot = f64_view(self.nHTot)
        self.atmos.muz = f64_view(self.muz)
        self.atmos.muy = f64_view(self.muy)
        self.atmos.mux = f64_view(self.mux)
        self.atmos.wmu = f64_view(self.wmu)

        cdef int Nspace = atmos.Nspace
        self.atmos.Nspace = Nspace
        cdef int Nrays = atmos.Nrays
        self.atmos.Nrays = Nrays

        cdef int Ndim = atmos.Ndim
        self.atmos.Ndim = Ndim
        cdef int Nx = atmos.Nx
        self.atmos.Nx = Nx
        cdef int Ny = atmos.Ny
        self.atmos.Ny = Ny
        cdef int Nz = atmos.Nz
        self.atmos.Nz = Nz

        if atmos.B is not None:
            self.B = atmos.B
            self.gammaB = atmos.gammaB
            self.chiB = atmos.chiB
            self.atmos.B = f64_view(self.B)
            self.atmos.gammaB = f64_view(self.gammaB)
            self.atmos.chiB = f64_view(self.chiB)
            self.cosGamma = np.zeros((Nrays, Nspace))
            self.atmos.cosGamma = f64_view_2(self.cosGamma)
            self.cos2chi = np.zeros((Nrays, Nspace))
            self.atmos.cos2chi = f64_view_2(self.cos2chi)
            self.sin2chi = np.zeros((Nrays, Nspace))
            self.atmos.sin2chi = f64_view_2(self.sin2chi)


        self.vlosMu = np.zeros((Nrays, Nspace))
        self.atmos.vlosMu = f64_view_2(self.vlosMu)

        self.configure_bcs(atmos)
        self.update_projections()

    def configure_bcs(self, atmos):
        cdef int Nx = max(atmos.Nx, 1)
        cdef int Ny = max(atmos.Ny, 1)
        cdef int Nz = atmos.Nz

        cdef int Nbcx = Nz * Ny
        cdef int Nbcy = Nz * Nx
        cdef int Nbcz = Nx * Ny
        s = atmos.structure
        self.atmos.xLowerBc = AtmosphericBoundaryCondition(BC_to_enum(s.xLowerBc), self.Nwave, Nbcx)
        self.atmos.xUpperBc = AtmosphericBoundaryCondition(BC_to_enum(s.xUpperBc), self.Nwave, Nbcx)
        self.atmos.yLowerBc = AtmosphericBoundaryCondition(BC_to_enum(s.yLowerBc), self.Nwave, Nbcy)
        self.atmos.yUpperBc = AtmosphericBoundaryCondition(BC_to_enum(s.yUpperBc), self.Nwave, Nbcy)
        self.atmos.zLowerBc = AtmosphericBoundaryCondition(BC_to_enum(s.zLowerBc), self.Nwave, Nbcz)
        self.atmos.zUpperBc = AtmosphericBoundaryCondition(BC_to_enum(s.zUpperBc), self.Nwave, Nbcz)

    def compute_bcs(self, LwSpectrum spect):
        cdef f64[:,::1] bc
        cdef int mu, la
        cdef F64View2D data
        if self.atmos.zLowerBc.type == CALLABLE:
            bc = self.pyAtmos.zLowerBc.compute_bc(self.pyAtmos, spect)
            verify_bc_array_sizes(&self.atmos.zLowerBc, bc, 'zLowerBc')
            data = f64_view_2(bc)
            self.atmos.zLowerBc.set_bc_data(data)

        if self.atmos.zUpperBc.type == CALLABLE:
            bc = self.pyAtmos.zUpperBc.compute_bc(self.pyAtmos, spect)
            verify_bc_array_sizes(&self.atmos.zUpperBc, bc, 'zUpperBc')
            data = f64_view_2(bc)
            self.atmos.zUpperBc.set_bc_data(data)

        if self.atmos.xLowerBc.type == CALLABLE:
            bc = self.pyAtmos.xLowerBc.compute_bc(self.pyAtmos, spect)
            verify_bc_array_sizes(&self.atmos.xLowerBc, bc, 'xLowerBc')
            data = f64_view_2(bc)
            self.atmos.xLowerBc.set_bc_data(data)

        if self.atmos.xUpperBc.type == CALLABLE:
            bc = self.pyAtmos.xUpperBc.compute_bc(self.pyAtmos, spect)
            verify_bc_array_sizes(&self.atmos.xUpperBc, bc, 'xUpperBc')
            data = f64_view_2(bc)
            self.atmos.xUpperBc.set_bc_data(data)

        if self.atmos.yLowerBc.type == CALLABLE:
            bc = self.pyAtmos.yLowerBc.compute_bc(self.pyAtmos, spect)
            verify_bc_array_sizes(&self.atmos.yLowerBc, bc, 'yLowerBc')
            data = f64_view_2(bc)
            self.atmos.yLowerBc.set_bc_data(data)

        if self.atmos.yUpperBc.type == CALLABLE:
            bc = self.pyAtmos.yUpperBc.compute_bc(self.pyAtmos, spect)
            verify_bc_array_sizes(&self.atmos.yUpperBc, bc, 'yUpperBc')
            data = f64_view_2(bc)
            self.atmos.yUpperBc.set_bc_data(data)

    def update_projections(self):
        self.atmos.update_projections()
        build_intersection_list(&self.atmos)

    def __getstate__(self):
        state = {}
        state['pyAtmos'] = self.pyAtmos
        state['x'] = self.pyAtmos.x
        state['y'] = self.pyAtmos.y
        state['z'] = self.pyAtmos.z
        state['temperature'] = self.pyAtmos.temperature
        state['ne'] = self.pyAtmos.ne
        state['vx'] = self.pyAtmos.vx
        state['vy'] = self.pyAtmos.vy
        state['vz'] = self.pyAtmos.vz
        state['vlosMu'] = np.asarray(self.vlosMu)
        try:
            state['B'] = self.pyAtmos.B
            state['gammaB'] = self.pyAtmos.gammaB
            state['chiB'] = self.pyAtmos.chiB
            state['cosGamma'] = np.asarray(self.cosGamma)
            state['cos2chi'] = np.asarray(self.cos2chi)
            state['sin2chi'] = np.asarray(self.sin2chi)
        except AttributeError:
            state['B'] = None
            state['gammaB'] = None
            state['chiB'] = None
            state['cosGamma'] = None
            state['cos2chi'] = None
            state['sin2chi'] = None
        state['vturb'] = self.pyAtmos.vturb
        state['nHTot'] = self.pyAtmos.nHTot
        state['muz'] = self.pyAtmos.muz
        state['muy'] = self.pyAtmos.muy
        state['mux'] = self.pyAtmos.mux
        state['wmu'] = self.pyAtmos.wmu
        state['Nwave'] = self.Nwave
        state['Ndim'] = self.Ndim
        state['Nx'] = self.Nx
        state['Ny'] = self.Ny
        state['Nz'] = self.Nz

        return state

    def __setstate__(self, state):
        self.pyAtmos = state['pyAtmos']
        self.x = state['x']
        self.atmos.x = f64_view(self.x)
        self.y = state['y']
        self.atmos.y = f64_view(self.y)
        self.z = state['z']
        self.atmos.z = f64_view(self.z)
        self.atmos.height = f64_view(self.z)
        self.temperature = state['temperature']
        self.atmos.temperature = f64_view(self.temperature)
        self.ne = state['ne']
        self.atmos.ne = f64_view(self.ne)
        self.vx = state['vx']
        self.atmos.vx = f64_view(self.vx)
        self.vy = state['vy']
        self.atmos.vy = f64_view(self.vy)
        self.vz = state['vz']
        self.atmos.vz = f64_view(self.vz)
        self.vlosMu = state['vlosMu']
        self.atmos.vlosMu = f64_view_2(self.vlosMu)
        if state['B'] is not None:
            self.B = state['B']
            self.atmos.B = f64_view(self.B)
            self.gammaB = state['gammaB']
            self.atmos.gammaB = f64_view(self.gammaB)
            self.chiB = state['chiB']
            self.atmos.chiB = f64_view(self.chiB)
            self.cosGamma = state['cosGamma']
            self.atmos.cosGamma = f64_view_2(self.cosGamma)
            self.cos2chi = state['cos2chi']
            self.atmos.cos2chi = f64_view_2(self.cos2chi)
            self.sin2chi = state['sin2chi']
            self.atmos.sin2chi = f64_view_2(self.sin2chi)
        self.vturb = state['vturb']
        self.atmos.vturb = f64_view(self.vturb)
        self.nHTot = state['nHTot']
        self.atmos.nHTot = f64_view(self.nHTot)
        self.muz = state['muz']
        self.atmos.muz = f64_view(self.muz)
        self.muy = state['muy']
        self.atmos.muy = f64_view(self.muy)
        self.mux = state['mux']
        self.atmos.mux = f64_view(self.mux)
        self.wmu = state['wmu']
        self.atmos.wmu = f64_view(self.wmu)

        cdef int Nspace = self.temperature.shape[0]
        self.atmos.Nspace = Nspace
        cdef int Nrays = self.vlosMu.shape[0]
        self.atmos.Nrays = Nrays
        cdef int Nwave = state['Nwave']
        self.Nwave = Nwave
        cdef int Ndim = state['Ndim']
        self.atmos.Ndim = Ndim
        cdef int Nx = state['Nx']
        self.atmos.Nx = Nx
        cdef int Ny = state['Ny']
        self.atmos.Ny = Ny
        cdef int Nz = state['Nz']
        self.atmos.Nz = Nz

        self.configure_bcs(self.pyAtmos)

    @property
    def Nspace(self):
        return self.atmos.Nspace

    @property
    def Nrays(self):
        return self.atmos.Nrays

    @property
    def Ndim(self):
        return self.atmos.Ndim

    @property
    def Nx(self):
        return self.atmos.Nx

    @property
    def Ny(self):
        return self.atmos.Ny

    @property
    def Nz(self):
        return self.atmos.Nz

    @property
    def x(self):
        return np.asarray(self.x)

    @property
    def y(self):
        return np.asarray(self.y)

    @property
    def z(self):
        return np.asarray(self.z)

    @property
    def height(self):
        return np.asarray(self.z)

    @property
    def temperature(self):
        return np.asarray(self.temperature)

    @property
    def ne(self):
        return np.asarray(self.ne)

    @property
    def vx(self):
        return np.asarray(self.vx)

    @property
    def vy(self):
        return np.asarray(self.vy)

    @property
    def vz(self):
        return np.asarray(self.vz)

    @property
    def vlos(self):
        if self.pyAtmos.Ndim > 1:
            raise ValueError('vlos is ambiguous when Ndim > 1, use vx, vy, or vz instead.')
        return np.asarray(self.vz)

    @property
    def vlosMu(self):
        return np.asarray(self.vlosMu)

    @property
    def B(self):
        return np.asarray(self.B)

    @property
    def gammaB(self):
        return np.asarray(self.gammaB)

    @property
    def chiB(self):
        return np.asarray(self.chiB)

    @property
    def cosGamma(self):
        return np.asarray(self.cosGamma)

    @property
    def cos2chi(self):
        return np.asarray(self.cos2chi)

    @property
    def sin2chi(self):
        return np.asarray(self.sin2chi)

    @property
    def vturb(self):
        return np.asarray(self.vturb)

    @property
    def nHTot(self):
        return np.asarray(self.nHTot)

    @property
    def muz(self):
        return np.asarray(self.muz)

    @property
    def muy(self):
        return np.asarray(self.muy)

    @property
    def mux(self):
        return np.asarray(self.mux)

    @property
    def wmu(self):
        return np.asarray(self.wmu)


cdef class BackgroundProvider:
    def __init__(self, eqPops, radSet, wavelength):
        pass

    # cpdef compute_background(self, LwAtmosphere atmos, f64[:,::1] chi, f64[:,::1] eta, f64[:,::1] sca):
    cpdef compute_background(self, LwAtmosphere atmos, chi, eta, sca):
        raise NotImplementedError

cdef class BasicBackground(BackgroundProvider):
    cdef BackgroundData bd
    cdef object eqPops
    cdef object radSet

    cdef f64[::1] chPops
    cdef f64[::1] ohPops
    cdef f64[::1] h2Pops
    cdef f64[::1] hMinusPops
    cdef f64[:,::1] hPops

    cdef f64[::1] wavelength

    def __init__(self, eqPops, radSet, wavelength):
        super().__init__(eqPops, radSet, wavelength)
        self.eqPops = eqPops
        self.radSet = radSet

        if 'CH' in eqPops:
            self.chPops = eqPops['CH']
            self.bd.chPops = f64_view(self.chPops)
        if 'OH' in eqPops:
            self.ohPops = eqPops['OH']
            self.bd.ohPops = f64_view(self.ohPops)
        if 'H2' in eqPops:
            self.h2Pops = eqPops['H2']
            self.bd.h2Pops = f64_view(self.h2Pops)

        self.hMinusPops = eqPops['H-']
        self.bd.hMinusPops = f64_view(self.hMinusPops)
        self.hPops = eqPops['H']
        self.bd.hPops = f64_view_2(self.hPops)

        self.wavelength = wavelength
        self.bd.wavelength = f64_view(self.wavelength)

    # cpdef compute_background(self, LwAtmosphere atmos, f64[:,::1] chi, f64[:,::1] eta, f64[:,::1] sca):
    cpdef compute_background(self, LwAtmosphere atmos, chiIn, etaIn, scaIn):
        cdef int Nlambda = self.wavelength.shape[0]
        cdef int Nspace = atmos.Nspace
        cdef f64[:,::1] chi = chiIn
        cdef f64[:,::1] eta = etaIn
        cdef f64[:,::1] sca = scaIn

        # NOTE(cmo): Update hPops in case it changed LTE<->NLTE
        self.hPops = self.eqPops['H']

        self.bd.chi = f64_view_2(chi)
        self.bd.eta = f64_view_2(eta)
        self.bd.scatt = f64_view_2(sca)

        basic_background(&self.bd, &atmos.atmos)
        self.rayleigh_scattering(atmos, sca)
        self.bf_opacities(atmos, chi, eta)

        cdef int la, k
        for la in range(Nlambda):
            for k in range(Nspace):
                chi[la, k] += sca[la, k]

    cpdef rayleigh_scattering(self, LwAtmosphere atmos, f64[:,::1] sca):
        cdef f64[::1] scaLine = np.zeros(atmos.Nspace)
        cdef int k, la
        cdef RayleighScatterer rayH, rayHe

        if 'H' in self.radSet:
            hPops = self.eqPops['H']
            rayH = RayleighScatterer(atmos, self.radSet['H'], hPops)
            for la in range(self.wavelength.shape[0]):
                if rayH.scatter(self.wavelength[la], scaLine):
                    for k in range(atmos.Nspace):
                        sca[la, k] += scaLine[k]

        if 'He' in self.radSet:
            hePops = self.eqPops['He']
            rayHe = RayleighScatterer(atmos, self.radSet['He'], hePops)
            for la in range(self.wavelength.shape[0]):
                if rayHe.scatter(self.wavelength[la], scaLine):
                    for k in range(atmos.Nspace):
                        sca[la, k] += scaLine[k]

    cpdef bf_opacities(self, LwAtmosphere atmos, f64[:,::1] chi, f64[:,::1] eta):
        atoms = self.radSet.passiveAtoms
        # print([a.name for a in atoms])
        if len(atoms) == 0:
            return

        continua = []
        cdef f64 sigma0 = 32.0 / (3.0 * sqrt(3.0)) * Const.QElectron**2 / (4.0 * np.pi * Const.Epsilon0) / (Const.MElectron * Const.CLight) * Const.HPlanck / (2.0 * Const.ERydberg)
        for a in atoms:
            for c in a.continua:
                continua.append(c)

        cdef f64[:, ::1] alpha = np.zeros((self.wavelength.shape[0], len(continua)))
        cdef int i, la, k, Z
        cdef f64 nEff, gbf_0, wav, edge, lambdaMin
        for i, c in enumerate(continua):
            alphaLa = c.alpha(np.asarray(self.wavelength))
            for la in range(self.wavelength.shape[0]):
                alpha[la, i] = alphaLa[la]

        cdef f64[:, ::1] expla = np.zeros((self.wavelength.shape[0], atmos.Nspace))
        cdef f64 hc_k = Const.HC / (Const.KBoltzmann * Const.NM_TO_M)
        cdef f64 twohc = (2.0 * Const.HC) / Const.NM_TO_M**3
        cdef f64 hc_kla
        for la in range(self.wavelength.shape[0]):
            hc_kla = hc_k / self.wavelength[la]
            for k in range(atmos.Nspace):
                expla[la, k] = exp(-hc_kla / atmos.temperature[k])

        cdef f64 twohnu3_c2
        cdef f64 gijk
        cdef int ci
        cdef int cj
        cdef f64[:,::1] nStar
        cdef f64[:,::1] n
        for i, c in enumerate(continua):
            nStar = self.eqPops.atomicPops[c.atom.element].nStar
            n = self.eqPops.atomicPops[c.atom.element].n

            ci = c.i
            cj = c.j
            for la in range(self.wavelength.shape[0]):
                twohnu3_c2 = twohc / self.wavelength[la]**3
                for k in range(atmos.Nspace):
                    gijk = nStar[ci, k] / nStar[cj, k] * expla[la, k]
                    chi[la, k] += alpha[la, i] * (1.0 - expla[la, k]) * n[ci, k]
                    eta[la, k] += twohnu3_c2 * gijk * alpha[la, i] * n[cj, k]

    def __getstate__(self):
        state = {}
        state['eqPops'] = self.eqPops
        state['radSet'] = self.radSet
        if 'CH' is self.eqPops:
            state['chPops'] = self.eqPops['CH']
        else:
            state['chPops'] = None

        if 'OH' in self.eqPops:
            state['ohPops'] = self.eqPops['OH']
        else:
            state['ohPops'] = None

        if 'H2' in self.eqPops:
            state['h2Pops'] = self.eqPops['H2']
        else:
            state['h2Pops'] = None

        state['hMinusPops'] = self.eqPops['H-']
        state['hPops'] = self.eqPops['H']
        state['wavelength'] = np.asarray(self.wavelength)

        return state

    def __setstate__(self, state):
        self.eqPops = state['eqPops']
        self.radSet = state['radSet']

        if state['chPops'] is not None:
            self.chPops = state['chPops']
            self.bd.chPops = f64_view(self.chPops)
        if state['ohPops'] is not None:
            self.ohPops = state['ohPops']
            self.bd.ohPops = f64_view(self.h2Pops)
        if state['h2Pops'] is not None:
            self.h2Pops = state['h2Pops']
            self.bd.h2Pops = f64_view(self.h2Pops)

        self.hMinusPops = state['hMinusPops']
        self.bd.hMinusPops = f64_view(self.hMinusPops)
        self.hPops = state['hPops']
        self.bd.hPops = f64_view_2(self.hPops)

        self.wavelength = state['wavelength']
        self.bd.wavelength = f64_view(self.wavelength)

    @classmethod
    def _reconstruct(cls, state):
        o = cls.__new__(cls)
        o.__setstate__(state)
        return o

    def __reduce__(self):
        return self._reconstruct, (self.__getstate__(),)


cdef class LwBackground:
    cdef Background background
    cdef object eqPops
    cdef object radSet

    cdef BackgroundProvider provider

    cdef f64[::1] wavelength
    cdef f64[:,::1] chi
    cdef f64[:,::1] eta
    cdef f64[:,::1] sca

    def __init__(self, atmosphere, eqPops, radSet, wavelength, provider=None):
        cdef LwAtmosphere atmos = atmosphere
        self.eqPops = eqPops
        self.radSet = radSet

        self.wavelength = wavelength

        cdef int Nlambda = self.wavelength.shape[0]
        cdef int Nspace = atmos.Nspace

        self.chi = np.zeros((Nlambda, Nspace))
        self.eta = np.zeros((Nlambda, Nspace))
        self.sca = np.zeros((Nlambda, Nspace))

        if provider is None:
            self.provider = BasicBackground(eqPops, radSet, wavelength)
        else:
            self.provider = provider(eqPops, radSet, wavelength)

        chiPy = np.asarray(self.chi)
        etaPy = np.asarray(self.eta)
        scaPy = np.asarray(self.sca)
        self.provider.compute_background(atmos, chiPy, etaPy, scaPy)

        self.background.chi = f64_view_2(self.chi)
        self.background.eta = f64_view_2(self.eta)
        self.background.sca = f64_view_2(self.sca)

    cpdef update_background(self, LwAtmosphere atmos):
        chiPy = np.asarray(self.chi)
        etaPy = np.asarray(self.eta)
        scaPy = np.asarray(self.sca)
        self.provider.compute_background(atmos, chiPy, etaPy, scaPy)

    def __getstate__(self):
        state = {}
        state['eqPops'] = self.eqPops
        state['radSet'] = self.radSet
        state['provider'] = self.provider
        state['wavelength'] = np.asarray(self.wavelength)
        state['chi'] = np.asarray(self.chi)
        state['eta'] = np.asarray(self.eta)
        state['sca'] = np.asarray(self.sca)

        return state

    def __setstate__(self, state):
        self.eqPops = state['eqPops']
        self.radSet = state['radSet']
        self.provider = state['provider']

        self.wavelength = state['wavelength']
        self.chi = state['chi']
        self.eta = state['eta']
        self.sca = state['sca']
        self.background.chi = f64_view_2(self.chi)
        self.background.eta = f64_view_2(self.eta)
        self.background.sca = f64_view_2(self.sca)

    @property
    def chi(self):
        return np.asarray(self.chi)

    @property
    def eta(self):
        return np.asarray(self.eta)

    @property
    def sca(self):
        return np.asarray(self.sca)


cdef class RayleighScatterer:
    cdef f64 lambdaLimit
    cdef LwAtmosphere atmos
    cdef f64 C
    cdef f64 sigmaE
    cdef f64[:,::1] pops
    cdef object atom
    cdef bool_t lines
    cdef list lambdaRed

    def __init__(self, atmos, atom, pops):
        if len(atom.lines) == 0:
            self.lines = False
            return

        self.lines = True
        self.lambdaRed = []
        cdef f64 lambdaLimit = 1e6
        cdef f64 lambdaRed
        for l in atom.lines:
            lambdaRed = l.wavelength()[-1]
            self.lambdaRed.append(lambdaRed)
            if l.i == 0:
                lambdaLimit = min(lambdaLimit, lambdaRed)

        self.lambdaLimit = lambdaLimit
        self.atom = atom
        self.atmos = atmos
        self.pops = pops

        C = Const
        self.C = 2.0 * np.pi * (C.QElectron / C.Epsilon0) * C.QElectron / C.MElectron / C.CLight
        self.sigmaE = 8.0 * np.pi / 3.0 * (C.QElectron / (np.sqrt(4.0 * np.pi * C.Epsilon0) * (np.sqrt(C.MElectron) * C.CLight)))**4

    cpdef scatter(self, f64 wavelength, f64[::1] sca):
        if wavelength <= self.lambdaLimit:
            return False
        if not self.lines:
            return False

        cdef f64 fomega = 0.0
        cdef f64 g0 = self.atom.levels[0].g
        cdef f64 lambdaRed
        cdef f64 f
        cdef int i
        for i, l in enumerate(self.atom.lines):
            if l.i != 0:
                continue

            lambdaRed = self.lambdaRed[i]
            if wavelength > lambdaRed:
                lambda2 = 1.0 / ((wavelength / l.lambda0)**2 - 1.0)
                f = l.Aji * (l.jLevel.g / g0) * (l.lambda0 * Const.NM_TO_M)**2 / self.C
                fomega += f * lambda2**2

        cdef f64 sigmaRayleigh = self.sigmaE * fomega

        cdef int k
        for k in range(sca.shape[0]):
            sca[k] = sigmaRayleigh * self.pops[0, k]

        return True

cdef gII_to_numpy(F64Arr3D gII):
    if gII.data() is NULL:
        raise AttributeError
    cdef np.npy_intp shape[3]
    shape[0] = <np.npy_intp> gII.shape(0)
    shape[1] = <np.npy_intp> gII.shape(1)
    shape[2] = <np.npy_intp> gII.shape(2)
    cdef f64[:,:,::1] ndarray = np.PyArray_SimpleNewFromData(3, &shape[0],
                                            np.NPY_FLOAT64, <void*>gII.data())
    return ndarray.copy()

cdef gII_from_numpy(Transition trans, f64[:,:,::1] gII):
    trans.prdStorage.gII = F64Arr3D(f64_view_3(gII))

cdef class LwTransition:
    cdef Transition trans
    cdef f64[:, :, :, ::1] phi
    cdef f64[:, :, :, ::1] phiQ
    cdef f64[:, :, :, ::1] phiU
    cdef f64[:, :, :, ::1] phiV
    cdef f64[:, :, :, ::1] psiQ
    cdef f64[:, :, :, ::1] psiU
    cdef f64[:, :, :, ::1] psiV
    cdef f64[::1] wphi
    cdef f64[::1] alpha
    cdef f64[::1] wavelength
    cdef i8[::1] active
    cdef f64[::1] Qelast
    cdef f64[::1] aDamp
    cdef f64[:, ::1] rhoPrd
    cdef f64[::1] Rij
    cdef f64[::1] Rji
    cdef public object transModel
    cdef LwAtmosphere atmos
    cdef object spect
    cdef public LwAtom atom

    def __init__(self, trans, compAtom, atmos, spect):
        self.transModel = trans
        cdef LwAtom atom = compAtom
        self.atom = atom
        cdef LwAtmosphere a = atmos
        self.atmos = a
        self.spect = spect
        transId = trans.transId
        self.wavelength = spect.transWavelengths[transId]
        self.trans.wavelength = f64_view(self.wavelength)
        self.trans.i = trans.i
        self.trans.j = trans.j
        self.trans.polarised = False
        self.trans.Nblue = spect.blueIdx[transId]
        cdef int Nlambda = self.wavelength.shape[0]
        cdef int Nspace = self.atmos.Nspace
        cdef int Nrays = self.atmos.Nrays

        if isinstance(trans, AtomicLine):
            self.trans.type = LINE
            self.trans.Aji = trans.Aji
            self.trans.Bji = trans.Bji
            self.trans.Bij = trans.Bij
            self.trans.lambda0 = trans.lambda0
            self.trans.dopplerWidth = Const.CLight / self.trans.lambda0
            self.Qelast = np.zeros(Nspace)
            self.aDamp = np.zeros(Nspace)
            self.trans.Qelast = f64_view(self.Qelast)
            self.trans.aDamp = f64_view(self.aDamp)
            self.phi = np.zeros((Nlambda, Nrays, 2, Nspace))
            self.wphi = np.zeros(Nspace)
            self.trans.phi = f64_view_4(self.phi)
            self.trans.wphi = f64_view(self.wphi)
            self.compute_phi()
            if trans.type == LineType.PRD:
                self.rhoPrd = np.ones((Nlambda, Nspace))
                self.trans.rhoPrd = f64_view_2(self.rhoPrd)
        else:
            self.trans.type = CONTINUUM
            self.alpha = trans.alpha(np.asarray(self.wavelength))
            self.trans.alpha = f64_view(self.alpha)
            self.trans.dopplerWidth = 1.0
            self.trans.lambda0 = trans.lambda0

        self.active = spect.activeWavelengths[transId].astype(np.int8)
        self.trans.active = BoolView(<bool_t*>&self.active[0], self.active.shape[0])

        atomicState = self.atom.modelPops
        self.Rij = atomicState.radiativeRates[(self.trans.i, self.trans.j)]
        self.Rji = atomicState.radiativeRates[(self.trans.j, self.trans.i)]
        self.trans.Rij = f64_view(self.Rij)
        self.trans.Rji = f64_view(self.Rji)

    def __getstate__(self):
        state = {}
        state['atmos'] = self.atmos
        state['atom'] = self.atom
        state['spect'] = self.spect
        state['transModel'] = self.transModel
        state['type'] = self.type
        state['Nblue'] = self.trans.Nblue
        transId = self.transModel.transId
        state['wavelength'] = self.spect.transWavelengths[transId]
        state['active'] = np.asarray(self.active)
        modelPops = self.atom.modelPops
        state['Rij'] = modelPops.radiativeRates[(self.trans.i, self.trans.j)]
        state['Rji'] = modelPops.radiativeRates[(self.trans.j, self.trans.i)]
        state['polarised'] = False
        if self.type == 'Line':
            state['phi'] = np.asarray(self.phi)
            try:
                state['phiQ'] = np.asarray(self.phiQ)
                state['phiU'] = np.asarray(self.phiU)
                state['phiV'] = np.asarray(self.phiV)
                state['psiQ'] = np.asarray(self.psiQ)
                state['psiU'] = np.asarray(self.psiU)
                state['psiV'] = np.asarray(self.psiV)
                state['polarised'] = True
            except AttributeError:
                state['phiQ'] = None
                state['phiU'] = None
                state['phiV'] = None
                state['psiQ'] = None
                state['psiU'] = None
                state['psiV'] = None

            state['wphi'] = np.asarray(self.wphi)
            state['Qelast'] = np.asarray(self.Qelast)
            state['aDamp'] = np.asarray(self.aDamp)
            try:
                state['rhoPrd'] = np.asarray(self.rhoPrd)
            except AttributeError:
                state['rhoPrd'] = None

            try:
                state['gII'] = np.asarray(gII_to_numpy(self.trans.prdStorage.gII))
            except AttributeError:
                state['gII'] = None
        else:
            state['alpha'] = np.asarray(self.alpha)
        return state

    def __setstate__(self, state):
        self.transModel = state['transModel']
        trans = self.transModel
        cdef LwAtmosphere a = state['atmos']
        self.atmos = a
        cdef LwAtom atom = state['atom']
        self.atom = atom
        self.spect = state['spect']
        self.wavelength = state['wavelength']
        self.trans.wavelength = f64_view(self.wavelength)
        self.trans.i = trans.i
        self.trans.j = trans.j
        self.trans.Nblue = state['Nblue']
        self.trans.polarised = state['polarised']

        if state['type'] == 'Line':
            self.trans.type = LINE
            self.trans.Aji = trans.Aji
            self.trans.Bji = trans.Bji
            self.trans.Bij = trans.Bij
            self.trans.lambda0 = trans.lambda0
            self.trans.dopplerWidth = Const.CLight / self.trans.lambda0
            self.Qelast = state['Qelast']
            self.aDamp = state['aDamp']
            self.trans.Qelast = f64_view(self.Qelast)
            self.trans.aDamp = f64_view(self.aDamp)
            self.phi = state['phi']
            self.wphi = state['wphi']
            self.trans.phi = f64_view_4(self.phi)
            self.trans.wphi = f64_view(self.wphi)
            if state['rhoPrd'] is not None:
                self.rhoPrd = state['rhoPrd']
                self.trans.rhoPrd = f64_view_2(self.rhoPrd)
            if state['gII'] is not None:
                gII_from_numpy(self.trans, state['gII'])
                self.trans.gII = self.trans.prdStorage.gII

            if state['polarised']:
                self.phiQ = state['phiQ']
                self.phiU = state['phiU']
                self.phiV = state['phiV']
                self.psiQ = state['psiQ']
                self.psiU = state['psiU']
                self.psiV = state['psiV']
                self.trans.phiQ = f64_view_4(self.phiQ)
                self.trans.phiU = f64_view_4(self.phiU)
                self.trans.phiV = f64_view_4(self.phiV)
                self.trans.psiQ = f64_view_4(self.psiQ)
                self.trans.psiU = f64_view_4(self.psiU)
                self.trans.psiV = f64_view_4(self.psiV)
        else:
            self.trans.type = CONTINUUM
            self.alpha = state['alpha']
            self.trans.alpha = f64_view(self.alpha)
            self.trans.dopplerWidth = 1.0
            self.trans.lambda0 = trans.lambda0

        self.active = state['active']
        self.trans.active = BoolView(<bool_t*>&self.active[0], self.active.shape[0])

        self.Rij = state['Rij']
        self.Rji = state['Rji']
        self.trans.Rij = f64_view(self.Rij)
        self.trans.Rji = f64_view(self.Rji)

    def load_rates_prd_from_state(self, prevState, preserveProfiles=True):

        np.asarray(self.Rij)[:] = prevState['Rij']
        np.asarray(self.Rji)[:] = prevState['Rji']

        if self.type == 'Continuum':
            return

        cdef int k
        if self.wavelength.shape == prevState['wavelength'].shape \
           and np.all(self.wavelength == prevState['wavelength']):
            if prevState['rhoPrd'] is not None:
                np.asarray(self.rhoPrd)[:] = prevState['rhoPrd']

            if prevState['gII'] is not None:
                gII_from_numpy(self.trans, prevState['gII'])
                self.trans.gII = self.trans.prdStorage.gII

            if preserveProfiles:
                np.asarray(self.phi)[:] = prevState['phi']
                if prevState['phiQ'] is not None:
                    np.asarray(self.phiQ)[:] = prevState['phiQ']
                    np.asarray(self.phiU)[:] = prevState['phiU']
                    np.asarray(self.phiV)[:] = prevState['phiV']
                    np.asarray(self.psiQ)[:] = prevState['psiQ']
                    np.asarray(self.psiU)[:] = prevState['psiU']
                    np.asarray(self.psiV)[:] = prevState['psiV']

        else:
            if prevState['rhoPrd'] is not None:
                for k in range(prevState['rhoPrd'].shape[1]):
                    np.asarray(self.rhoPrd)[:, k] = np.interp(self.wavelength, prevState['wavelength'], prevState['rhoPrd'][:, k])


    def compute_phi(self):
        if self.type == 'Continuum':
            return

        cdef Atmosphere* atmos = &self.atmos.atmos
        callbackUsed = False
        def default_voigt_callback(f64[::1] aDamp, f64[::1] vBroad):
            cdef F64View aDampView = f64_view(aDamp)
            cdef F64View vBroadView = f64_view(vBroad)
            self.trans.compute_phi(atmos[0], aDampView, vBroadView)
            nonlocal callbackUsed
            callbackUsed = True
            return np.asarray(self.phi)

        state = LineProfileState(wavelength=np.asarray(self.wavelength),
                                 vlosMu=np.asarray(self.atmos.vlosMu),
                                 atmos=self.atmos.pyAtmos,
                                 eqPops=self.atom.eqPops,
                                 default_voigt_callback=default_voigt_callback,
                                 vBroad=self.atom.vBroad)
        profile = self.transModel.compute_phi(state)

        cdef f64[:,:,:,::1] phi = profile.phi
        cdef f64[::1] Qelast = profile.Qelast
        cdef f64[::1] aDamp = profile.aDamp
        if not callbackUsed:
            self.phi[...] = phi
        self.Qelast[...] = Qelast
        self.aDamp[...] = aDamp

        self.trans.compute_wphi(self.atmos.atmos)

    cpdef compute_polarised_profiles(self):
        if self.type == 'Continuum':
            return

        if not self.transModel.polarisable:
            return

        cdef int Nlambda = self.wavelength.shape[0]
        cdef int Nrays = self.atmos.Nrays
        cdef int Nspace = self.atmos.Nspace
        try:
            self.phiQ
        except AttributeError:
            self.phiQ = np.zeros((Nlambda, Nrays, 2, Nspace))
            self.phiU = np.zeros((Nlambda, Nrays, 2, Nspace))
            self.phiV = np.zeros((Nlambda, Nrays, 2, Nspace))
            self.psiQ = np.zeros((Nlambda, Nrays, 2, Nspace))
            self.psiU = np.zeros((Nlambda, Nrays, 2, Nspace))
            self.psiV = np.zeros((Nlambda, Nrays, 2, Nspace))
            self.trans.phiQ = f64_view_4(self.phiQ)
            self.trans.phiU = f64_view_4(self.phiU)
            self.trans.phiV = f64_view_4(self.phiV)
            self.trans.psiQ = f64_view_4(self.psiQ)
            self.trans.psiU = f64_view_4(self.psiU)
            self.trans.psiV = f64_view_4(self.psiV)

        self.trans.polarised = True

        cdef LwAtom atom = self.atom
        aDamp, Qelast = self.transModel.damping(self.atmos.pyAtmos, atom.eqPops)

        cdef Atmosphere* atmos = &self.atmos.atmos
        cdef int i
        for i in range(self.Qelast.shape[0]):
            self.Qelast[i] = Qelast[i]
            self.aDamp[i] = aDamp[i]

        z = self.transModel.zeeman_components()
        cdef LwZeemanComponents zc = LwZeemanComponents(z)

        self.trans.compute_polarised_profiles(atmos[0], self.trans.aDamp, atom.atom.vBroad, zc.zc)

    def uv(self, int la, int mu, bool_t toObs, f64[::1] Uji not None, f64[::1] Vij not None, f64[::1] Vji not None):
        # TODO(cmo): Allow these to take None, and allocate if they are. Then
        # return in some UV datastruct
        cdef bint obs = toObs
        cdef F64View cUji = f64_view(Uji)
        cdef F64View cVij = f64_view(Vij)
        cdef F64View cVji = f64_view(Vji)

        self.trans.uv(la, mu, obs, cUji, cVij, cVji)

    @property
    def jLevel(self):
        return self.transModel.jLevel

    @property
    def iLevel(self):
        return self.transModel.iLevel

    @property
    def j(self):
        return self.transModel.j

    @property
    def i(self):
        return self.transModel.i

    @property
    def Aji(self):
        return self.trans.Aji

    @property
    def Bji(self):
        return self.trans.Bji

    @property
    def Bij(self):
        return self.trans.Bij

    @property
    def Nblue(self):
        return self.trans.Nblue

    @property
    def dopplerWidth(self):
        return self.trans.dopplerWidth

    @property
    def lambda0(self):
        return self.trans.lambda0

    @property
    def wphi(self):
        return np.asarray(self.wphi)

    @property
    def phi(self):
        return np.asarray(self.phi)

    @property
    def phiQ(self):
        return np.asarray(self.phiQ)

    @property
    def phiU(self):
        return np.asarray(self.phiU)

    @property
    def phiV(self):
        return np.asarray(self.phiV)

    @property
    def psiQ(self):
        return np.asarray(self.psiQ)

    @property
    def psiU(self):
        return np.asarray(self.psiU)

    @property
    def psiV(self):
        return np.asarray(self.psiV)

    @property
    def Rij(self):
        return np.asarray(self.Rij)

    @property
    def Rji(self):
        return np.asarray(self.Rji)

    @property
    def rhoPrd(self):
        return np.asarray(self.rhoPrd)

    @property
    def alpha(self):
        return np.asarray(self.alpha)

    @property
    def wavelength(self):
        return np.asarray(self.wavelength)

    @property
    def active(self):
        return np.asarray(self.active).astype(np.bool)

    @property
    def Qelast(self):
        return np.asarray(self.Qelast)

    @property
    def aDamp(self):
        return np.asarray(self.aDamp)

    @property
    def polarisable(self):
        return self.transModel.polarisable

    @property
    def type(self):
        if self.trans.type == LINE:
            return 'Line'
        else:
            return 'Continuum'

cdef class LwZeemanComponents:
    cdef ZeemanComponents zc
    cdef np.int32_t[::1] alpha
    cdef f64[::1] shift
    cdef f64[::1] strength

    def __init__(self, z):
        self.alpha = z.alpha
        self.shift = z.shift
        self.strength = z.strength

        self.zc.alpha = I32View(&self.alpha[0], self.alpha.shape[0])
        self.zc.shift = f64_view(self.shift)
        self.zc.strength = f64_view(self.strength)

cdef class LwAtom:
    cdef Atom atom
    cdef f64[::1] vBroad
    cdef f64[:,:,::1] Gamma
    cdef f64[:,:,::1] C
    cdef f64[::1] nTotal
    cdef f64[:,::1] nStar
    cdef f64[:,::1] n
    cdef f64[::1] eta
    cdef f64[:,::1] chi
    cdef f64[:,::1] U
    cdef f64[:,::1] gij
    cdef f64[:,::1] wla
    cdef f64[::1] stages
    cdef public object atomicModel
    cdef public object modelPops
    cdef LwAtmosphere atmos
    cdef object eqPops
    cdef list trans
    cdef bool_t detailed

    def __init__(self, atom, atmos, eqPops, spect, background, detailed=False, initSol=None, ngOptions=None, conserveCharge=False):
        self.atomicModel = atom
        self.detailed = detailed
        cdef LwAtmosphere a = atmos
        self.atmos = a
        self.atom.atmos = &a.atmos
        self.eqPops = eqPops
        modelPops = eqPops.atomicPops[atom.element]
        self.modelPops = modelPops

        self.vBroad = atom.vBroad(atmos)
        self.atom.vBroad = f64_view(self.vBroad)
        self.nTotal = modelPops.nTotal
        self.atom.nTotal = f64_view(self.nTotal)

        self.trans = []
        for t in atom.transitions:
            if spect.activeTrans[t.transId]:
                self.trans.append(LwTransition(t, self, atmos, spect))

        cdef LwTransition lt
        for lt in self.trans:
            self.atom.trans.push_back(&lt.trans)

        cdef int Nlevel = len(atom.levels)
        cdef int Ntrans = len(self.trans)
        self.atom.Nlevel = Nlevel
        self.atom.Ntrans = Ntrans

        if not self.detailed:
            self.Gamma = np.zeros((Nlevel, Nlevel, a.Nspace))
            self.atom.Gamma = f64_view_3(self.Gamma)

            self.C = np.zeros((Nlevel, Nlevel, atmos.Nspace))
            self.atom.C = f64_view_3(self.C)

        self.stages = np.array([l.stage for l in self.atomicModel.levels], dtype=np.float64)
        self.atom.stages = f64_view(self.stages)
        self.nStar = modelPops.nStar
        self.atom.nStar = f64_view_2(self.nStar)

        doInitSol = True
        self.n = modelPops.n
        self.atom.n = f64_view_2(self.n)

        if self.detailed:
            doInitSol = False
            ngOptions = None

        if Ntrans > 0:
            self.gij = np.zeros((Ntrans, atmos.Nspace))
            self.atom.gij = f64_view_2(self.gij)
            self.wla = np.zeros((Ntrans, atmos.Nspace))
            self.atom.wla = f64_view_2(self.wla)

        if not self.detailed:
            self.U = np.zeros((Nlevel, atmos.Nspace))
            self.atom.U = f64_view_2(self.U)

            self.eta = np.zeros(atmos.Nspace)
            self.atom.eta = f64_view(self.eta)
            self.chi = np.zeros((Nlevel, atmos.Nspace))
            self.atom.chi = f64_view_2(self.chi)

        if initSol is None:
            initSol = InitialSolution.Lte

        if doInitSol and initSol == InitialSolution.Zero:
            raise ValueError('Zero radiation InitialSolution not currently supported')

        if doInitSol and initSol == InitialSolution.EscapeProbability and Ntrans > 0:
            self.set_pops_escape_probability(self.atmos, background, conserveCharge=conserveCharge)

        if ngOptions is not None:
            self.atom.ng = Ng(ngOptions.Norder, ngOptions.Nperiod, ngOptions.Ndelay, self.atom.n.flatten())
        else:
            self.atom.ng = Ng(0,0,0, self.atom.n.flatten())

    def __getstate__(self):
        state = {}
        state['atomicModel'] = self.atomicModel
        state['modelPops'] = self.modelPops
        state['atmos'] = self.atmos
        state['eqPops'] = self.eqPops
        state['trans'] = self.trans
        state['detailed'] = self.detailed
        state['vBroad'] = np.asarray(self.vBroad)
        state['nTotal'] = self.modelPops.nTotal
        state['nStar'] = self.modelPops.nStar
        state['n'] = self.modelPops.n
        state['stages'] = np.asarray(self.stages)
        state['Ng'] = (self.atom.ng.Norder, self.atom.ng.Nperiod, self.atom.ng.Ndelay)
        if self.detailed:
            state['U'] = None
            state['eta'] = None
            state['chi'] = None
            state['Gamma'] = None
            state['C'] = None
        else:
            state['U'] = np.asarray(self.U)
            state['eta'] = np.asarray(self.eta)
            state['chi'] = np.asarray(self.chi)
            state['Gamma'] = np.asarray(self.Gamma)
            state['C'] = np.asarray(self.C)
        cdef int Ntrans = len(self.trans)
        if Ntrans > 0:
            state['gij'] = np.asarray(self.gij)
            state['wla'] = np.asarray(self.wla)
        else:
            state['gij'] = None
            state['wla'] = None

        return state

    def __setstate__(self, state):
        self.atomicModel = state['atomicModel']
        self.modelPops = state['modelPops']
        cdef LwAtmosphere a = state['atmos']
        self.atmos = a
        self.atom.atmos = &a.atmos
        self.eqPops = state['eqPops']

        self.detailed = state['detailed']

        self.vBroad = state['vBroad']
        self.atom.vBroad = f64_view(self.vBroad)
        self.nTotal = state['nTotal']
        self.atom.nTotal = f64_view(self.nTotal)

        self.trans = state['trans']
        cdef LwTransition lt
        for lt in self.trans:
            self.atom.trans.push_back(&lt.trans)

        cdef int Nlevel = len(self.atomicModel.levels)
        cdef int Ntrans = len(self.trans)
        self.atom.Nlevel = Nlevel
        self.atom.Ntrans = Ntrans

        if not self.detailed:
            self.Gamma = state['Gamma']
            self.atom.Gamma = f64_view_3(self.Gamma)

            self.C = state['C']
            self.atom.C = f64_view_3(self.C)

            self.U = state['U']
            self.atom.U = f64_view_2(self.U)

            self.eta = state['eta']
            self.atom.eta = f64_view(self.eta)
            self.chi = state['chi']
            self.atom.chi = f64_view_2(self.chi)


        self.stages = state['stages']
        self.atom.stages = f64_view(self.stages)
        self.nStar = state['nStar']
        self.atom.nStar = f64_view_2(self.nStar)
        self.n = state['n']
        self.atom.n = f64_view_2(self.n)

        if Ntrans > 0:
            self.gij = state['gij']
            self.atom.gij = f64_view_2(self.gij)
            self.wla = state['wla']
            self.atom.wla = f64_view_2(self.wla)

        ng = state['Ng']
        self.atom.ng = Ng(ng[0], ng[1], ng[2], self.atom.n.flatten())

    def load_pops_rates_prd_from_state(self, prevState, popsOnly=False, preserveProfiles=False):
        if not self.detailed:
            np.asarray(self.n)[:] = prevState['n']
            ng = prevState['Ng']
            self.atom.ng = Ng(ng[0], ng[1], ng[2], self.atom.n.flatten())

        if popsOnly:
            return

        cdef LwTransition t
        cdef int i
        for i, t in enumerate(self.trans):
            for st in prevState['trans']:
                if st.transModel.i == t.i and st.transModel.j == t.j:
                    t.load_rates_prd_from_state(st.__getstate__(), preserveProfiles=preserveProfiles)
                    break

    def compute_collisions(self, fillDiagonal=False):
        cdef np.ndarray[np.double_t, ndim=3] C = np.asarray(self.C)
        C.fill(0.0)
        for col in self.atomicModel.collisions:
            col.compute_rates(self.atmos.pyAtmos, self.eqPops, C)
        C[C < 0.0] = 0.0

        if not fillDiagonal:
            return

        cdef int k
        cdef int i
        cdef int j
        cdef f64 CDiag
        for k in range(C.shape[2]):
            for i in range(C.shape[0]):
                CDiag = 0.0
                C[i, i, k] = 0.0
                for j in range(C.shape[0]):
                    CDiag += C[j, i, k]
                C[i, i, k] = -CDiag


    cpdef set_pops_escape_probability(self, LwAtmosphere a, LwBackground bg, conserveCharge=False, int Niter=100):
        cdef np.ndarray[np.double_t, ndim=3] Gamma
        cdef np.ndarray[np.double_t, ndim=3] C
        cdef f64 delta
        cdef int k
        cdef np.ndarray[np.double_t, ndim=1] deltaNe

        self.compute_collisions()
        Gamma = np.asarray(self.Gamma)
        C = np.asarray(self.C)

        if conserveCharge:
            prevN = np.copy(self.n)

        self.atom.ng = Ng(0,0,0, self.atom.n.flatten())
        start = time.time()
        for it in range(Niter):
            Gamma.fill(0.0)
            Gamma += C
            gamma_matrices_escape_prob(&self.atom, bg.background, a.atmos)
            try:
                stat_eq(&self.atom)
            except:
                raise ExplodingMatrixError('Singular Matrix')
            self.atom.ng.accelerate(self.atom.n.flatten())
            delta = self.atom.ng.max_change()
            if delta < 3e-2:
                end = time.time()
                print('Converged: %s, %d\nTime: %f' % (self.atomicModel.element.name, it, end-start))
                break
        else:
            print('Escape probability didn\'t converge for %s, setting LTE populations' % self.atomicModel.element.name)
            n = np.asarray(self.n)
            n[:] = np.asarray(self.nStar)

        if conserveCharge:
            deltaNe = np.sum((np.asarray(self.n) - prevN) * np.asarray(self.stages)[:, None], axis=0)

            for k in range(self.atmos.Nspace):
                self.atmos.ne[k] += deltaNe[k]

            for k in range(self.atmos.Nspace):
                if self.atmos.ne[k] < 1e6:
                    self.atmos.ne[k] = 1e6

    cpdef setup_wavelength(self, int la):
        self.atom.setup_wavelength(la)

    def update_profiles(self, polarised=False):
        np.asarray(self.vBroad)[:] = self.atomicModel.vBroad(self.atmos)
        for t in self.trans:
            if polarised and t.polarisable:
                t.compute_polarised_profiles()
            else:
                t.compute_phi()

    @property
    def Nlevel(self):
        return self.atom.Nlevel

    @property
    def Ntrans(self):
        return self.atom.Ntrans

    @property
    def vBroad(self):
        return np.asarray(self.vBroad)

    @property
    def Gamma(self):
        return np.asarray(self.Gamma)

    @property
    def C(self):
        return np.asarray(self.C)

    @property
    def nTotal(self):
        return np.asarray(self.nTotal)

    @property
    def n(self):
        return np.asarray(self.n)

    @property
    def nStar(self):
        return np.asarray(self.nStar)

    @property
    def trans(self):
        return self.trans

    @property
    def element(self):
        return self.atomicModel.element

cdef JRest_to_numpy(F64Arr2D JRest):
    if JRest.data() is NULL:
        raise AttributeError
    cdef np.npy_intp shape[2]
    shape[0] = <np.npy_intp> JRest.shape(0)
    shape[1] = <np.npy_intp> JRest.shape(1)
    ndarray = np.PyArray_SimpleNewFromData(2, &shape[0],
                                            np.NPY_FLOAT64, <void*>JRest.data())
    return ndarray

cdef JRest_from_numpy(Spectrum spect, f64[:,::1] JRest):
    spect.JRest = F64Arr2D(f64_view_2(JRest))

cdef class LwSpectrum:
    cdef Spectrum spect
    cdef f64[::1] wavelength
    cdef f64[:,:,::1] I
    cdef f64[:,::1] J
    cdef f64[:,:,:,::1] Quv

    def __init__(self, wavelength, Nrays, Nspace, Noutgoing):
        self.wavelength = wavelength
        cdef int Nspect = self.wavelength.shape[0]
        self.I = np.zeros((Nspect, Nrays, Noutgoing))
        self.J = np.zeros((Nspect, Nspace))

        self.spect.wavelength = f64_view(self.wavelength)
        self.spect.I = f64_view_3(self.I)
        self.spect.J = f64_view_2(self.J)

    def setup_stokes(self):
        self.Quv = np.zeros((3, self.I.shape[0], self.I.shape[1], self.I.shape[2]))
        self.spect.Quv = f64_view_4(self.Quv)

    def __getstate__(self):
        state = {}
        state['wavelength'] = np.asarray(self.wavelength)
        state['I'] = np.asarray(self.I)
        state['J'] = np.asarray(self.J)
        try:
            state['Quv'] = np.asarray(self.Quv)
        except AttributeError:
            state['Quv'] = None

        try:
            state['JRest'] = np.copy(JRest_to_numpy(self.spect.JRest))
        except AttributeError:
            state['JRest'] = None

        return state

    def __setstate__(self, state):
        self.wavelength = state['wavelength']
        self.spect.wavelength = f64_view(self.wavelength)
        self.I = state['I']
        self.spect.I = f64_view_3(self.I)
        self.J = state['J']
        self.spect.J = f64_view_2(self.J)

        if state['Quv'] is not None:
            self.Quv = state['Quv']
            self.spect.Quv = f64_view_4(self.Quv)

        if state['JRest'] is not None:
            JRest_from_numpy(self.spect, state['JRest'])

    def interp_J_from_state(self, prevSpect):
        cdef np.ndarray[np.double_t, ndim=2] J = np.asarray(self.J)
        cdef int k
        for k in range(self.J.shape[1]):
            J[:, k] = np.interp(self.wavelength, prevSpect.wavelength, prevSpect.J[:, k])

    @property
    def wavelength(self):
        return np.asarray(self.wavelength)

    @property
    def I(self):
        return np.squeeze(np.asarray(self.I))

    @property
    def J(self):
        return np.squeeze(np.asarray(self.J))

    @property
    def Quv(self):
        return np.squeeze(np.asarray(self.Quv))


cdef class LwContext:
    cdef Context ctx
    cdef LwAtmosphere atmos
    cdef LwSpectrum spect
    cdef LwBackground background
    cdef LwDepthData depthData
    cdef public dict kwargs
    cdef public object eqPops
    cdef list activeAtoms
    cdef list detailedAtoms
    cdef bool_t conserveCharge
    cdef bool_t hprd
    cdef public object crswCallback
    cdef public object crswDone
    cdef dict __dict__

    def __init__(self, atmos, spect, eqPops,
                 ngOptions=None, initSol=None,
                 conserveCharge=False, hprd=False,
                 crswCallback=None, Nthreads=1,
                 backgroundProvider=None,
                 formalSolver=None,
                 interpFn=None):
        self.__dict__ = {}
        self.kwargs = {'atmos': atmos, 'spect': spect, 'eqPops': eqPops, 'ngOptions': ngOptions, 'initSol': initSol, 'conserveCharge': conserveCharge, 'hprd': hprd, 'Nthreads': Nthreads, 'backgroundProvider': backgroundProvider, 'formalSolver': formalSolver, 'interpFn': interpFn}

        self.atmos = LwAtmosphere(atmos, spect.wavelength.shape[0])
        self.spect = LwSpectrum(spect.wavelength, atmos.Nrays,
                                atmos.Nspace, atmos.Noutgoing)
        self.conserveCharge = conserveCharge
        self.hprd = hprd

        self.background = LwBackground(self.atmos, eqPops, spect.radSet,
                                       spect.wavelength, provider=backgroundProvider)
        self.eqPops = eqPops

        activeAtoms = spect.radSet.activeAtoms
        detailedAtoms = spect.radSet.detailedAtoms
        self.activeAtoms = [LwAtom(a, self.atmos, eqPops, spect, self.background, ngOptions=ngOptions, initSol=initSol, conserveCharge=conserveCharge) for a in activeAtoms]
        self.detailedAtoms = [LwAtom(a, self.atmos, eqPops, spect, self.background, ngOptions=None, initSol=InitialSolution.Lte, detailed=True) for a in detailedAtoms]

        self.ctx.atmos = &self.atmos.atmos
        self.ctx.spect = &self.spect.spect
        self.ctx.background = &self.background.background

        cdef LwAtom la
        for la in self.activeAtoms:
            self.ctx.activeAtoms.push_back(&la.atom)
        for la in self.detailedAtoms:
            self.ctx.detailedAtoms.push_back(&la.atom)

        if self.hprd:
            self.configure_hprd_coeffs()

        if crswCallback is None:
            self.crswCallback = UnityCrswIterator()
            self.crswDone = True
        else:
            self.crswCallback = crswCallback
            self.crswDone = False

        shape = (self.spect.I.shape[0], self.atmos.Nrays, self.atmos.Nspace)
        self.depthData = LwDepthData(*shape)
        self.ctx.depthData = &self.depthData.depthData

        self.set_formal_solver(formalSolver)
        self.set_interp_fn(interpFn)
        self.setup_threads(Nthreads)

    def __getstate__(self):
        state = {}
        state['kwargs'] = self.kwargs
        state['eqPops'] = self.eqPops
        state['activeAtoms'] = self.activeAtoms
        state['detailedAtoms'] = self.detailedAtoms
        state['conserveCharge'] = self.conserveCharge
        state['hprd'] = self.hprd
        state['atmos'] = self.atmos
        state['spect'] = self.spect
        state['background'] = self.background
        state['crswDone'] = self.crswDone
        if not self.crswDone:
            state['crswCallback'] = self.crswCallback
        else:
            state['crswCallback'] = None
        state['depthData'] = self.depthData
        return state

    def __setstate__(self, state):
        self.kwargs = state['kwargs']
        self.eqPops = state['eqPops']
        self.atmos = state['atmos']
        self.activeAtoms = state['activeAtoms']
        self.detailedAtoms = state['detailedAtoms']
        self.conserveCharge = state['conserveCharge']
        self.hprd = state['hprd']
        self.spect = state['spect']
        self.background = state['background']

        self.crswDone = state['crswDone']
        if state['crswCallback'] is None:
            self.crswCallback = UnityCrswIterator()
        else:
            self.crswCallback = state['crswCallback']

        self.ctx.atmos = &self.atmos.atmos
        self.ctx.spect = &self.spect.spect
        self.ctx.background = &self.background.background

        cdef LwAtom la
        for la in self.activeAtoms:
            self.ctx.activeAtoms.push_back(&la.atom)
        for la in self.detailedAtoms:
            self.ctx.detailedAtoms.push_back(&la.atom)

        if self.hprd:
            self.configure_hprd_coeffs()

        shape = (self.spect.I.shape[0], self.atmos.Nrays, self.atmos.Nspace)
        self.depthData = state['depthData']
        self.ctx.depthData = &self.depthData.depthData
        self.set_formal_solver(self.kwargs['formalSolver'])
        self.set_interp_fn(self.kwargs['interpFn'])

        self.setup_threads(state['kwargs']['Nthreads'])

    def set_formal_solver(self, formalSolver):
        cdef LwFormalSolverManager fsMan = FormalSolvers
        cdef int fsIdx
        if formalSolver is not None:
            fsIdx = fsMan.names.index(formalSolver)
        else:
            fsIdx = fsMan.default_formal_solver(self.ctx.atmos.Ndim)
        cdef FormalSolver fs = fsMan.manager.formalSolvers[fsIdx]
        self.ctx.formalSolver = fs

    def set_interp_fn(self, interpFn):
        cdef LwInterpFnManager interpMan = InterpFns
        cdef int interpIdx
        cdef InterpFn
        try:
            if interpFn is not None:
                interpIdx = interpMan.names.index(interpFn)
            else:
                interpIdx = interpMan.default_interp(self.ctx.atmos.Ndim)
            interp = interpMan.manager.fns[interpIdx]
            self.ctx.interpFn = interp
            return
        except:
            pass
        # self.ctx.interpFn = InterpFn()

    @property
    def Nthreads(self):
        return self.ctx.Nthreads

    @Nthreads.setter
    def Nthreads(self, value):
        cdef int prevValue = self.ctx.Nthreads
        self.ctx.Nthreads = int(value)
        if prevValue != value:
            self.update_threads()

    cdef setup_threads(self, int Nthreads):
        self.ctx.Nthreads = Nthreads
        self.ctx.initialise_threads()

    cpdef update_threads(self):
        self.ctx.update_threads()

    cpdef compute_profiles(self, polarised=False):
        atoms = self.activeAtoms + self.detailedAtoms
        for atom in atoms:
            atom.update_profiles(polarised=polarised)

    cpdef formal_sol_gamma_matrices(self, fixCollisionalRates=False, lambdaIterate=False):
        cdef LwAtom atom
        cdef np.ndarray[np.double_t, ndim=3] Gamma
        cdef f64 crswVal = self.crswCallback()
        if crswVal == 1.0:
            self.crswDone = True
        else:
            print('CRSW: %.2e'%crswVal)

        for atom in self.activeAtoms:
            Gamma = np.asarray(atom.Gamma)
            Gamma.fill(0.0)
            if not fixCollisionalRates:
                atom.compute_collisions()
            Gamma += crswVal * np.asarray(atom.C)

        self.atmos.compute_bcs(self.spect)

        cdef f64 dJ = formal_sol_gamma_matrices(self.ctx, lambdaIterate)
        print('dJ = %.2e' % dJ)
        return dJ

    cpdef formal_sol(self):
        # cdef LwAtom atom
        # cdef np.ndarray[np.double_t, ndim=3] Gamma
        # for atom in self.activeAtoms:
        #     Gamma = np.asarray(atom.Gamma)
        #     Gamma.fill(0.0)
        #     atom.compute_collisions()
        #     Gamma += atom.C

        self.atmos.compute_bcs(self.spect)
        cdef f64 dJ = formal_sol(self.ctx)
        # cdef f64 dJ = formal_sol_gamma_matrices(self.ctx)
        # print('dJ = %.2e' % dJ)
        return dJ

    cpdef update_deps(self, temperature=True, ne=True, vturb=True, vlos=True, B=True, background=True, hprd=True):
        if vlos or B:
            self.atmos.update_projections()

        if temperature or vturb:
            self.compute_profiles()

        if temperature or ne:
            self.eqPops.update_lte_atoms_Hmin_pops(self.kwargs['atmos'], conserveCharge=self.conserveCharge, updateTotals=True)

        if background and any([temperature, ne, vturb, vlos]):
            self.background.update_background(self.atmos)

        if self.hprd and hprd:
            self.update_hprd_coeffs()

    cpdef time_dep_update(self, f64 dt, prevTimePops=None):
        atoms = self.activeAtoms

        cdef LwAtom atom
        cdef Atom* a
        cdef f64 delta
        cdef f64 maxDelta = 0.0
        cdef bool_t accelerated

        if prevTimePops is None:
            # TODO(cmo): Do we need to preserve the previous J too, or can we just reset I and J to 0 if needed? (to prevent NaN poisoning)
            prevTimePops = [np.copy(atom.n) for atom in atoms]

        for i, atom in enumerate(atoms):
            a = &atom.atom

            try:
                time_dependent_update(a, f64_view_2(prevTimePops[i]), dt)
            except:
                raise ExplodingMatrixError('Singular Matrix')
            accelerated = a.ng.accelerate(a.n.flatten())
            delta = a.ng.max_change()
            maxDelta = max(maxDelta, delta)
            s = '    %s delta = %6.4e' % (atom.atomicModel.element.name, delta)
            if accelerated:
                s += ' (accelerated)'
            print(s)


        return maxDelta, prevTimePops

    cpdef time_dep_restore_prev_pops(self, prevTimePops):
        cdef LwAtom atom
        cdef int i
        for i, atom in enumerate(self.activeAtoms):
            np.asarray(atom.n)[:] = prevTimePops[i]

        np.asarray(self.spect.I).fill(0.0)
        np.asarray(self.spect.J).fill(0.0)

    cpdef time_dep_conserve_charge(self, prevTimePops):
        cdef np.ndarray[np.double_t, ndim=1] deltaNe
        cdef LwAtom atom
        cdef int i, k

        atoms = self.activeAtoms
        for i, atom in enumerate(atoms):
            deltaNe = np.sum((np.asarray(atom.n) - prevTimePops[i]) * np.asarray(atom.stages)[:, None], axis=0)
            for k in range(self.atmos.Nspace):
                self.atmos.ne[k] += deltaNe[k]

            for k in range(self.atmos.Nspace):
                if self.atmos.ne[k] < 1e6:
                    self.atmos.ne[k] = 1e6

    cpdef clear_ng(self):
        cdef LwAtom atom
        for atom in self.activeAtoms:
            atom.atom.ng.clear()

    cpdef stat_equil(self):
        atoms = self.activeAtoms

        cdef LwAtom atom
        cdef Atom* a
        cdef f64 delta
        cdef f64 maxDelta = 0.0
        cdef bool_t accelerated
        cdef LwTransition t
        cdef int k
        cdef np.ndarray[np.double_t, ndim=1] deltaNe

        for atom in atoms:
            a = &atom.atom
            if not a.ng.init:
                a.ng.accelerate(a.n.flatten())
            try:
                stat_eq(a)
            except:
                raise ExplodingMatrixError('Singular Matrix')
            accelerated = a.ng.accelerate(a.n.flatten())
            delta = a.ng.max_change()
            s = '    %s delta = %6.4e' % (atom.atomicModel.element.name, delta)
            if accelerated:
                s += ' (accelerated)'
            print(s)
            maxDelta = max(maxDelta, delta)

        if self.conserveCharge:
            self.nr_post_update()

        return maxDelta

    cpdef update_projections(self):
        self.atmos.update_projections()

    cpdef setup_stokes(self, recompute=False):
        try:
            if self.atmos.B.shape[0] == 0:
                raise ValueError('Please specify B-field')
        except:
            raise ValueError('Please specify B-field')

        atoms = self.activeAtoms + self.detailedAtoms
        atomsHavePolarisedProfile = True
        try:
            atoms[0].phiQ
        except AttributeError:
            atomsHavePolarisedProfile = False

        if recompute or not atomsHavePolarisedProfile:
            for atom in self.activeAtoms:
                for t in atom.trans:
                    t.compute_polarised_profiles()
            for atom in self.detailedAtoms:
                for t in atom.trans:
                    t.compute_polarised_profiles()

        self.spect.setup_stokes()

    cpdef single_stokes_fs(self, recompute=False):
        self.setup_stokes(recompute=recompute)

        self.atmos.compute_bcs(self.spect)
        cdef f64 dJ = formal_sol_full_stokes(self.ctx)
        return dJ

    cpdef prd_redistribute(self, int maxIter=3, f64 tol=1e-2):
        cdef PrdIterData prdIter = redistribute_prd_lines(self.ctx, maxIter, tol)
        print('      PRD dRho = %.2e, (sub-iterations: %d)' % (prdIter.dRho, prdIter.iter))
        return prdIter.dRho, prdIter.iter

    cdef configure_hprd_coeffs(self):
        configure_hprd_coeffs(self.ctx)

    cpdef update_hprd_coeffs(self):
        self.configure_hprd_coeffs()
        # NOTE(cmo): configure_hprd_coeffs throws away all of the interpolation
        # stuff stored on each line, and allocates a new block for it, this
        # means that the Transitions sitting in the threading factories are now
        # pointing to stale data, so we regenerate the entire threading
        # context. This is a bit wasteful, but at 8 threads it takes < 3 ms, vs
        # 500 ms+ for the coeffs on an average CaII + MgII case.
        self.update_threads()

    @property
    def activeAtoms(self):
        return self.activeAtoms

    @property
    def spect(self):
        return self.spect

    @property
    def atmos(self):
        return self.atmos

    @property
    def background(self):
        return self.background

    @property
    def depthData(self):
        return self.depthData

    @property
    def pops(self):
        # return self.arguments['eqPops']
        return self.eqPops

    def state_dict(self):
        return self.__getstate__()

    @staticmethod
    def construct_from_state_dict_with(sd, atmos=None, spect=None, eqPops=None,
                                       ngOptions=None, initSol=None, conserveCharge=None,
                                       hprd=None, preserveProfiles=False, fromScratch=False,
                                       backgroundProvider=None):
        """
        stateDict will not be deepcopied by this function -- do that yourself if needed.
        """
        sd = copy(sd)
        sd['kwargs'] = copy(sd['kwargs'])
        args = sd['kwargs']
        wavelengthSubset = False

        if ngOptions is not None:
            args['ngOptions'] = ngOptions
        if initSol is not None:
            args['initSol'] = initSol
        if conserveCharge is not None:
            args['conserveCharge'] = conserveCharge
        if hprd is not None:
            args['hprd'] = hprd
        if backgroundProvider is not None:
            args['backgroundProvider'] = backgroundProvider

        if atmos is not None:
            args['atmos'] = atmos
            if not eqPops:
                # TODO(cmo); This should also probably recompute ICE
                args['eqPops'] = copy(args['eqPops'])
                args['eqPops'].atmos = atmos
                args['eqPops'].update_lte_atoms_Hmin_pops(args['atmos'], conserveCharge=args['conserveCharge'])
        if eqPops is not None:
            args['eqPops'] = eqPops
        if spect is not None:
            prevSpect = args['spect']
            args['spect'] = spect
            wavelengthSubset = spect.wavelength[0] >= prevSpect.wavelength[0] and spect.wavelength[-1] <= prevSpect.wavelength[-1]
        if not fromScratch:
            prevInitSol = args['initSol']
            args['initSol'] = InitialSolution.Lte

        ctx = LwContext(**args)

        if fromScratch:
            return ctx

        if wavelengthSubset:
            ctx.spect.interp_J_from_state(sd['spect'])

        # TODO(cmo): I don't really like the way we use __getstate__ here, for
        # pickling the new approach is better. Performance implact is probably
        # negligble...
        cdef LwAtom a
        for a in ctx.activeAtoms:
            for s in sd['activeAtoms']:
                if a.atomicModel.element == s.atomicModel.element:
                    levels = a.atomicModel.levels == s.atomicModel.levels
                    if not levels:
                        break
                    trans = a.atomicModel.lines == s.atomicModel.lines
                    trans = trans and a.atomicModel.continua == s.atomicModel.continua
                    popsOnly = False
                    if not trans:
                        popsOnly = True
                    a.load_pops_rates_prd_from_state(s.__getstate__(), popsOnly=popsOnly, preserveProfiles=preserveProfiles)
                    break
            else:
                if prevInitSol == InitialSolution.EscapeProbability:
                    a.set_pops_escape_probability(ctx.atmos, ctx.background, conserveCharge=ctx.conserveCharge)


        for a in ctx.detailedAtoms:
            for s in sd['detailedAtoms']:
                if a.atomicModel == s.atomicModel:
                    a.load_pops_rates_prd_from_state(s.__getstate__())
                    break

        return ctx

    def compute_rays(self, wavelengths=None, mus=None, stokes=False,
                     refinePrd=False, squeeze=True):
        state = deepcopy(self.state_dict())
        if wavelengths is not None:
            spect = state['kwargs']['spect'].subset_configuration(wavelengths)
        else:
            spect = None

        cdef LwContext rhoCtx, rayCtx
        if refinePrd:
            rhoCtx = self.construct_from_state_dict_with(state, spect=spect)
            rhoCtx.prd_redistribute(maxIter=100)
            sd = rhoCtx.state_dict()
            atmos = sd['kwargs']['atmos']
            if mus is not None:
                if isinstance(mus, dict):
                    atmos.rays(**mus)
                else:
                    atmos.rays(mus)
            rayCtx = self.construct_from_state_dict_with(sd)
        else:
            atmos = state['kwargs']['atmos']
            if mus is not None:
                if isinstance(mus, dict):
                    atmos.rays(**mus)
                else:
                    atmos.rays(mus)
            rayCtx = self.construct_from_state_dict_with(state, spect=spect)

        if stokes:
            rayCtx.single_stokes_fs()
            Iwav = np.asarray(rayCtx.spect.I)
            if squeeze:
                Iwav = np.squeeze(Iwav)
            Iquv = np.zeros((4, *Iwav.shape))
            Iquv[0, :] = Iwav
            Iquv[1:, :] = np.asarray(rayCtx.spect.Quv)
            return Iquv
        else:
            rayCtx.formal_sol()
            Iwav = np.asarray(rayCtx.spect.I)
            if squeeze:
                Iwav = np.squeeze(Iwav)
            return Iwav

    def contrib_fn(self, line, wavelengths=None, mu=None, refinePrd=False):
        state = deepcopy(self.state_dict())
        if wavelengths is None:
            wavelengths = line.wavelength
        spect = state['kwargs']['spect'].subset_configuration(wavelengths)
        if refinePrd:
            rhoCtx = self.construct_from_state_dict_with(state, spect=spect)
            rhoCtx.prd_redistribute(maxIter=100)
            sd = rhoCtx.state_dict()
            atmos = sd['kwargs']['atmos']
            if mu is not None:
                atmos.rays(mu)
            rayCtx = self.construct_from_state_dict_with(sd)
        else:
            atmos = state['kwargs']['atmos']
            if mu is not None:
                atmos.rays(mu)
            rayCtx = self.construct_from_state_dict_with(state, spect=spect)
        if mu is None:
            mu = atmos.muz[-1]

        cdef f64[:,::1] pops
        cdef LwAtom atom
        cdef LwTransition trans = None
        for a in rayCtx.activeAtoms:
            if a.atomicModel.element == line.atom.element:
                for t in a.trans:
                    if t.i == line.i and t.j == line.j:
                        trans = t
                        pops = a.n
                        atom = a
                        break
            if trans is not None:
                break
        else:
            raise ValueError('Unable to find transition on active atoms')

        cdef int la, k
        cdef f64[:,::1] chiLine = np.zeros((wavelengths.shape[0], rayCtx.atmos.Nspace))
        cdef f64[:,::1] etaLine = np.zeros((wavelengths.shape[0], rayCtx.atmos.Nspace))


        cdef f64[::1] Uji = np.zeros(atmos.Nspace)
        cdef f64[::1] Vij = np.zeros(atmos.Nspace)
        cdef f64[::1] Vji = np.zeros(atmos.Nspace)

        for la in range(wavelengths.shape[0]):
            if not trans.active[la]:
                continue

            atom.setup_wavelength(la)
            trans.uv(la, 0, True, Uji, Vij, Vji)
            for k in range(rayCtx.atmos.Nspace):
                chiLine[la, k] = pops[line.i, k] * Vij[k] - pops[line.j, k] * Vji[k]
                etaLine[la, k] = pops[line.j, k] * Uji[k]

        cdef f64[:,::1] chiBg = rayCtx.background.chi
        cdef f64[:,::1] chiTot = np.zeros((wavelengths.shape[0], rayCtx.atmos.Nspace))
        cdef f64[:,::1] tau = np.zeros((wavelengths.shape[0], rayCtx.atmos.Nspace))
        cdef f64[::1] height = rayCtx.atmos.height
        cdef f64[::1] tau_ref = rayCtx.atmos.tau_ref

        for la in range(wavelengths.shape[0]):
            for k in range(rayCtx.atmos.Nspace):
                chiTot[la, k] = chiLine[la, k] + chiBg[la, k]

            tau[la, 0] = 0.5 * chiTot[la, 0] * (height[0] - height[1])
            for k in range(1, rayCtx.atmos.Nspace):
                tau[la, k] = tau[la, k-1] + 0.5 * (chiTot[la, k-1] + chiTot[la, k]) * (height[k-1] - height[k])

        cdef f64[:,::1] SLine = np.asarray(etaLine) / (np.asarray(chiLine) + 1e-40)
        cdef f64[:,::1] contFn = (np.asarray(chiTot) / mu * np.exp(-np.asarray(tau) / mu) * np.asarray(SLine))

        result = {'contFn': np.asarray(contFn), 'SLine': np.asarray(SLine), 'tau': np.asarray(tau), 'chiTot': np.asarray(chiTot), 'chiLine': np.asarray(chiLine), 'chiBg': np.asarray(chiBg), 'etaLine': np.asarray(etaLine)}

        return result

cdef class LwFormalSolverManager:
    cdef FormalSolverManager manager
    cdef public list paths
    cdef public list names

    def __init__(self):
        self.paths = []
        self.names = []
        cdef int i
        cdef int size
        cdef const char* name

        for i in range(self.manager.formalSolvers.size()):
            name = self.manager.formalSolvers[i].name
            self.names.append(name.decode('UTF-8'))

    def load_fs_from_path(self, str path):
        if path in self.paths:
            raise ValueError('Tried to load a pre-existing path')

        self.paths.append(path)
        byteStore = path.encode('UTF-8')
        cdef const char* cPath = byteStore
        cdef bool_t success = self.manager.load_fs_from_path(cPath)
        if not success:
            raise ValueError('Failed to load Formal Solver from library at %s' % path)

        cdef const char* name = self.manager.formalSolvers[self.manager.formalSolvers.size()-1].name
        self.names.append(name.decode('UTF-8'))

    def default_formal_solver(self, Ndim):
        if Ndim == 1:
            return self.names.index('piecewise_bezier3_1d')
        elif Ndim == 2:
            return self.names.index('piecewise_besser_2d')
        else:
            raise ValueError()

cdef class LwInterpFnManager:
    cdef InterpFnManager manager
    cdef public list paths
    cdef public list names

    def __init__(self):
        self.paths = []
        self.names = []
        cdef int i
        cdef int size
        cdef const char* name

        for i in range(self.manager.fns.size()):
            name = self.manager.fns[i].name
            self.names.append(name.decode('UTF-8'))

    def load_fs_from_path(self, str path):
        if path in self.paths:
            raise ValueError('Tried to load a pre-existing path')

        self.paths.append(path)
        byteStore = path.encode('UTF-8')
        cdef const char* cPath = byteStore
        cdef bool_t success = self.manager.load_fn_from_path(cPath)
        if not success:
            raise ValueError('Failed to load interpolation function from library at %s' % path)

        cdef const char* name = self.manager.fns[self.manager.fns.size()-1].name
        self.names.append(name.decode('UTF-8'))

    def default_interp(self, Ndim):
        if Ndim == 2:
            return self.names.index('interp_besser_2d')
        else:
            raise ValueError("Unexpected Ndim")

FormalSolvers = LwFormalSolverManager()
InterpFns = LwInterpFnManager()
