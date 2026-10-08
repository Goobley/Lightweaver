import numpy as np
cimport numpy as np
from CmoArray cimport *
from libcpp cimport bool as bool_t
from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.math cimport sqrt, exp, copysign
from libc.stdint cimport int64_t
from .atmosphere import BoundaryCondition, ZeroRadiation, ThermalisedRadiation, PeriodicRadiation, NoBc
from .atomic_model import AtomicLine, LineType, LineProfileState
from .utils import InitialSolution, ExplodingMatrixError, UnityCrswIterator, check_shape_exception, get_fs_iter_libs
from .atomic_table import PeriodicTable
from .atomic_set import lte_pops
from .iteration_update import IterationUpdate
from .deprecation import accepts_old_kwargs, deprecated_alias, remap_old_keys
from weno4 import weno4
import lightweaver.constants as Const
import lightweaver.config as lwConfig
import time
import os
from enum import Enum, auto
from copy import copy, deepcopy

include 'CmoArrayHelper.pyx'

# NOTE(cmo): Some late binding stuff to be able to use numpy C API
np.import_array()

ctypedef np.int8_t i8
ctypedef int64_t i64
ctypedef Array1NonOwn[np.int32_t] I32View
ctypedef Array1NonOwn[bool_t] BoolView
ctypedef Array2NonOwn[np.int32_t] BcIdxs

# NOTE(cmo): Define everything we need from the C++ code.
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

    cdef cppclass FsIterationFns:
        int Ndim
        bool_t dimensionSpecific
        bool_t respectsFormalSolver
        bool_t defaultPerAtomStorage
        bool_t defaultWlaGijStorage
        const char* name

    cdef cppclass FsIterationFnsManager:
        vector[FsIterationFns] fns
        bool_t load_fns_from_path(const char* path)

cdef extern from "LwIterationResult.hpp":
    cdef cppclass IterationResult:
        bool_t updatedJ
        f64 dJMax
        int dJMaxIdx

        bool_t updatedPops
        vector[f64] dPops
        vector[int] dPopsMaxIdx
        bool_t ngAccelerated

        bool_t updatedNe
        f64 dNe
        int dNeMaxIdx

        bool_t updatedRho
        vector[f64] dRho
        vector[int] dRhoMaxIdx
        int NprdSubIter
        bool_t updatedJPrd
        vector[f64] dJPrdMax
        vector[int] dJPrdMaxIdx

cdef extern from "LwExtraParams.hpp":
    cdef cppclass ExtraParams:
        # NOTE(cmo): The const char* overloads are just to make Cython happy.
        ExtraParams()
        bool_t contains(const string& key)
        bool_t contains(const char* key)
        void insert[T](const string& key, T value) except +
        void insert[T](const char* key, T value) except +
        T& get_as[T](const string& key) except +
        T& get_as[T](const char* key) except +


cdef extern from "Lightweaver.hpp":
    cdef enum RadiationBc:
        UNINITIALISED
        ZERO
        THERMALISED
        PERIODIC
        CALLABLE

    cdef cppclass AtmosphericBoundaryCondition:
        RadiationBc type
        F64Arr3D bcData
        BcIdxs idxs

        AtmosphericBoundaryCondition()
        AtmosphericBoundaryCondition(RadiationBc typ, int Nwave, int Nmu,
                                     int Nspace, BcIdxs indexVector)
        void set_bc_data(F64View3D data)

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

cdef extern from "FastBackground.hpp":
    cdef cppclass BackgroundContinuum:
        int i
        int j
        int laStart
        int laEnd
        F64View alpha
        BackgroundContinuum(int i, int j, f64 minLambda, f64 lambdaEdge,
                            F64View crossSection, F64View globalWavelength)

    cdef cppclass ResonantRayleighLine:
        f64 Aji
        f64 gRatio
        f64 lambda0
        f64 lambdaMax
        ResonantRayleighLine(f64 A, f64 gjgi, f64 lambda0, f64 lambdaMax)

    cdef cppclass BackgroundAtom:
        F64View2D n;
        F64View2D nStar;
        vector[BackgroundContinuum] continua;
        vector[ResonantRayleighLine] resonanceScatterers;
        BackgroundAtom()

    cdef cppclass FastBackgroundContext:
        int Nthreads
        void initialise(int numThreads)
        void basic_background(BackgroundData* bd, Atmosphere* atmos)
        void bf_opacities(BackgroundData* bd, vector[BackgroundAtom]* atoms,
                          Atmosphere* atmos)
        void rayleigh_scatter(BackgroundData* bd, vector[BackgroundAtom]* atoms,
                              Atmosphere* atmos)


cdef extern from "Ng.hpp":
    cdef cppclass NgChange:
        f64 dMax
        i64 dMaxIdx

    cdef cppclass NgArgs:
        int nOrder
        int nPeriod
        int nDelay
        f64 threshold
        f64 lowerThreshold

    cdef cppclass Ng:
        int Norder
        int Nperiod
        int Ndelay
        f64 threshold
        f64 lowerThreshold
        bool_t init
        Ng()
        Ng(const NgArgs& args, F64View sol)
        bool_t accelerate(F64View sol)
        NgChange max_change()
        NgChange relative_change_from_prev(F64View newSol)
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
        int Nred
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

        void recompute_gII()
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

        vector[Transition*] trans
        Ng ng

        int Nlevel
        int Ntrans
        void setup_wavelength(int la)
        void init_scratch(i64 Nspace, bool_t detailed, bool_t wlaGijStorage, bool_t defaulPerAtomStorage)

    cdef cppclass DepthData:
        bool_t fill
        F64View4D chi
        F64View4D eta
        F64View4D I

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
        FsIterationFns iterFns
        void initialise_threads()
        void update_threads()

    cdef cppclass PrdIterData:
        int iter
        f64 dRho

    cdef cppclass NrTimeDependentData:
        f64 dt
        vector[F64View2D] nPrev

    cdef IterationResult formal_sol_gamma_matrices(Context& ctx, bool_t lambdaIterate, ExtraParams params) except +
    cdef IterationResult formal_sol(Context& ctx, bool_t upOnly, ExtraParams params) except +
    cdef IterationResult formal_sol_full_stokes(Context& ctx, bool_t updateJ,
                                                bool_t upOnly, ExtraParams params) except +
    cdef IterationResult redistribute_prd_lines(Context& ctx, int maxIter, f64 tol, ExtraParams params) except +
    cdef void stat_eq(Context& ctx, Atom* atom, ExtraParams params) except +
    cdef void stat_eq_impl(Atom* atom) except +
    cdef void time_dependent_update(Context& ctx,  Atom* atomIn,
                                    F64View2D nOld, f64 dt, ExtraParams params) except +
    cdef void nr_post_update(Context& ctx, vector[Atom*]* atoms,
                             const vector[F64View3D]& dC,
                             F64View backgroundNe,
                             const NrTimeDependentData& timeDepData,
                             f64 crswVal,
                             ExtraParams params) except +
    cdef void configure_hprd_coeffs(Context& ctx)
    cdef void configure_hprd_coeffs(Context& ctx, bool_t includeDetailedAtoms)

cdef extern from "Lightweaver.hpp" namespace "EscapeProbability":
    cdef void gamma_matrices_escape_prob(Atom* a, Background& background,
                                         const Atmosphere& atmos)

cdef ExtraParams dict2ExtraParams(dict d):
    """
    Convert a dict to an ExtraParams object accepted by Lightweaver's cpp API.
    Will raise Exceptions (mostly Type and ValueError) on incompatible input.
    All keys are expected to be strings.

    Acceptable value types:
      - str
      - bool
      - int (up to the max supported by int64)
      - float
      - np.ndarray (up to 4 dimensions, dtype either <f8 or <i8 and C-contiguous).

    N.B. Arrays may be modified by the underlying function.
    """
    # NOTE(cmo): I do not like this function at all
    supportedTypes = (str, bool, int, float, np.ndarray)
    cdef ExtraParams result = ExtraParams()

    cdef char* kPtr
    cdef char* vPtr
    cdef bool_t  bVal
    cdef i64 iVal
    cdef f64 fVal
    cdef f64[::1] f64View1
    cdef f64[:,::1] f64View2
    cdef f64[:,:,::1] f64View3
    cdef f64[:,:,:,::1] f64View4
    cdef i64[::1] i64View1
    cdef i64[:,::1] i64View2
    cdef i64[:,:,::1] i64View3
    cdef i64[:,:,:,::1] i64View4

    for k, v in d.items():
        if type(k) is not str:
            raise TypeError(("Dictionary keys for ExtraParams must be str, "
                            f"got '{type(k)}'' for key {k}"))

        if type(v) not in supportedTypes:
            raise TypeError((f"Value for key {k} is not of a supported type, "
                             f"got {type(v)}, expected one of {supportedTypes}"))

        # NOTE(cmo): Whilst this will be passed through const std::string&, it's
        # hash is what is stored in the underlying data structure, which will be
        # done by the end of the insert call. Also strings can no longer be COW
        # in C++11+.
        key = k.encode('UTF-8')
        kPtr = key

        if type(v) is str:
            val = v.encode('UTF-8')
            vPtr = val
            result.insert(kPtr, vPtr)
        elif type(v) is bool:
            bVal = v
            result.insert(kPtr, bVal)
        elif type(v) is int:
            iVal = v
            result.insert(kPtr, iVal)
        elif type(v) is float:
            fVal = v
            result.insert(kPtr, fVal)
        elif type(v) is np.ndarray:
            if v.ndim > 4:
                raise ValueError(("Unsupported number of dimensions on value "
                                f"associated with {k}, max supported is 4, "
                                f"got {v.ndim}"))
            if v.dtype == np.float64:
                if v.ndim == 1:
                    f64View1 = v
                    result.insert(kPtr, f64_view(f64View1))
                elif v.ndim == 2:
                    f64View2 = v
                    result.insert(kPtr, f64_view_2(f64View2))
                elif v.ndim == 3:
                    f64View3 = v
                    result.insert(kPtr, f64_view_3(f64View3))
                elif v.ndim == 4:
                    f64View4 = v
                    result.insert(kPtr, f64_view_4(f64View4))
            elif v.dtype == np.int64:
                if v.ndim == 1:
                    i64View1 = v
                    result.insert(kPtr,
                                  Array1NonOwn[i64](&i64View1[0], i64View1.shape[0]))
                elif v.ndim == 2:
                    i64View2 = v
                    result.insert(kPtr,
                                  Array2NonOwn[i64](&i64View2[0,0],
                                                    i64View2.shape[0],
                                                    i64View2.shape[1]))
                elif v.ndim == 3:
                    i64View3 = v
                    result.insert(kPtr,
                                  Array3NonOwn[i64](&i64View3[0,0,0],
                                                    i64View3.shape[0],
                                                    i64View3.shape[1],
                                                    i64View3.shape[2]))
                elif v.ndim == 4:
                    i64View4 = v
                    result.insert(kPtr,
                                  Array4NonOwn[i64](&i64View4[0,0,0,0],
                                                    i64View4.shape[0],
                                                    i64View4.shape[1],
                                                    i64View4.shape[2],
                                                    i64View4.shape[3]))
            else:
                raise TypeError((f"Got array with type {v.dtype} for key {k}, ",
                                 "only contiguous float64 and int64 are supported."))
    return result

cdef class LwDepthData:
    '''
    Simple object to lazily hold data that isn't usually stored during the
    Formal Solution (full angularly dependent emissivity, opacity, and
    intensity at every point), due to the high memory cost. This is a part of
    the Context and doesn't need to be instantiated directly.
    '''
    cdef object shape
    cdef DepthData depth_data
    cdef f64[:,:,:,::1] chi
    cdef f64[:,:,:,::1] eta
    cdef f64[:,:,:,::1] I

    def __init__(self, Nlambda, Nmu, Nspace):
        self.shape = (Nlambda, Nmu, 2, Nspace)
        self.depth_data.fill = 0

    def __getstate__(self):
        s = {}
        s['shape'] = self.shape
        s['fill'] = bool(self.fill)
        try:
            s['chi'] = np.copy(np.asarray(self.chi))
            s['eta'] = np.copy(np.asarray(self.eta))
            s['I'] = np.copy(np.asarray(self.I))
        except AttributeError:
            s['chi'] = None
            s['eta'] = None
            s['I'] = None

        return s

    def __setstate__(self, s):
        self.shape = s['shape']
        self.depth_data.fill = int(s['fill'])
        if s['chi'] is not None:
            self.chi = s['chi']
            self.depth_data.chi = f64_view_4(self.chi)
            self.eta = s['eta']
            self.depth_data.eta = f64_view_4(self.eta)
            self.I = s['I']
            self.depth_data.I = f64_view_4(self.I)

    @property
    def fill(self):
        '''
        Set this to True to fill the arrays, this will take care of
        allocating the space if not previously done.
        '''
        return bool(self.depth_data.fill)

    @fill.setter
    def fill(self, value):
        try:
            self.depth_data.fill = int(value)
            if value:
                self.chi
        except AttributeError:
            self.chi = np.zeros(self.shape)
            self.depth_data.chi = f64_view_4(self.chi)
            self.eta = np.zeros(self.shape)
            self.depth_data.eta = f64_view_4(self.eta)
            self.I = np.zeros(self.shape)
            self.depth_data.I = f64_view_4(self.I)

    @property
    def chi(self):
        '''
        Full depth dependent opacity [Nlambda, Nmu, Up/Down, Nspace].
        '''
        return np.asarray(self.chi)

    @property
    def eta(self):
        '''
        Full depth dependent emissivity [Nlambda, Nmu, Up/Down, Nspace].
        '''
        return np.asarray(self.eta)

    @property
    def I(self):
        '''
        Full depth dependent intensity [Nlambda, Nmu, Up/Down, Nspace].
        '''
        return np.asarray(self.I)

def BC_to_enum(bc):
    '''
    Returns the C++ enum associated with the type of python BoundaryCondition
    object.
    '''
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

cdef verify_bc_array_sizes(AtmosphericBoundaryCondition* abc, f64[:,:,::1] pyArr, str location):
    cdef int dim0 = abc.bcData.shape(0)
    cdef int dim1 = abc.bcData.shape(1)
    cdef int dim2 = abc.bcData.shape(2)
    if dim0 != pyArr.shape[0] or dim1 != pyArr.shape[1] or dim2 != pyArr.shape[2]:
        raise ValueError('BC returned from python does not match expected shape for %s (%d, %d, %d), got %s' % (location, dim0, dim1, dim2, repr(pyArr.shape)))

cdef class LwAtmosphere:
    '''
    Storage for the C++ class, ensuring all of the arrays remained pinned
    from python. Usually constructed by the Context.

    Parameters
    ----------
    atmos : Atmosphere
        The python atmosphere object.
    Nwavelengths : int
        The number of wavelengths used in the wavelength grid.
    '''
    cdef Atmosphere atmos
    cdef f64[::1] x
    cdef f64[::1] y
    cdef f64[::1] z
    cdef f64[::1] temperature
    cdef f64[::1] ne
    cdef f64[::1] vx
    cdef f64[::1] vy
    cdef f64[::1] vz
    cdef f64[:,::1] vlos_mu
    cdef f64[::1] B
    cdef f64[::1] gamma_B
    cdef f64[::1] chi_B
    cdef f64[:,::1] cos_gamma
    cdef f64[:,::1] cos2chi
    cdef f64[:,::1] sin2chi
    cdef f64[::1] vturb
    cdef f64[::1] nh_tot
    cdef f64[::1] muz
    cdef f64[::1] muy
    cdef f64[::1] mux
    cdef f64[::1] wmu
    # TODO(cmo): I don't really like storing Nwave here, but I don't know how
    # much of a choice we have.
    cdef int Nwave

    cdef public object py_atmos

    def __init__(self, atmos, Nwavelengths):
        cdef int Nwave = Nwavelengths
        self.Nwave = Nwave
        self.py_atmos = atmos

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

        self.x = atmos.x
        check_shape_exception(self.x, Nx, name='x')
        self.y = atmos.y
        check_shape_exception(self.y, Ny, name='y')
        self.z = atmos.z
        check_shape_exception(self.z, Nz, name='y')

        self.temperature = atmos.temperature
        check_shape_exception(self.temperature, Nspace, name='temperature')
        self.ne = atmos.ne
        check_shape_exception(self.ne, Nspace, name='ne')

        self.vz = atmos.vz
        check_shape_exception(self.vz, Nspace, name='vz')
        self.vx = atmos.vx
        if atmos.vx.size > 0:
            check_shape_exception(self.vx, Nspace, name='vx')
        self.vy = atmos.vy
        if atmos.vy.size > 0:
            check_shape_exception(self.vy, Nspace, name='vy')

        self.vturb = atmos.vturb
        check_shape_exception(self.vturb, Nspace, name='vturb')
        self.nh_tot = atmos.nh_tot
        check_shape_exception(self.nh_tot, Nspace, name='vturb')
        try:
            self.muz = atmos.muz
            check_shape_exception(self.muz, Nrays, name='muz')
            self.muy = atmos.muy
            check_shape_exception(self.muy, Nrays, name='muy')
            self.mux = atmos.mux
            check_shape_exception(self.mux, Nrays, name='mux')
            self.wmu = atmos.wmu
            check_shape_exception(self.wmu, Nrays, name='wmu')
        except AttributeError as e:
            raise ValueError(f'One of the quadrature values not found, was .quadrature called on the Atmosphere object? (Caught: {e})')
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
        self.atmos.nHTot = f64_view(self.nh_tot)
        self.atmos.muz = f64_view(self.muz)
        self.atmos.muy = f64_view(self.muy)
        self.atmos.mux = f64_view(self.mux)
        self.atmos.wmu = f64_view(self.wmu)

        if atmos.B is not None:
            self.B = atmos.B
            check_shape_exception(self.B, Nspace, name='B')
            self.gamma_B = atmos.gamma_B
            check_shape_exception(self.gamma_B, Nspace, name='gamma_B')
            self.chi_B = atmos.chi_B
            check_shape_exception(self.chi_B, Nspace, name='chi_B')
            if self.B.shape[0] != self.gamma_B.shape[0] or self.B.shape[0] != self.chi_B.shape[0]:
                raise ValueError(f'Shapes of B, gamma_B, and chi_B don\'t match, verify that these are correctly set in the Atmosphere provided to Context. (B: {self.B.shape}, chi_B: {self.chi_B.shape}, gamma_B: {self.gamma_B.shape}.')
            self.atmos.B = f64_view(self.B)
            self.atmos.gammaB = f64_view(self.gamma_B)
            self.atmos.chiB = f64_view(self.chi_B)
            self.cos_gamma = np.zeros((Nrays, Nspace))
            self.atmos.cosGamma = f64_view_2(self.cos_gamma)
            self.cos2chi = np.zeros((Nrays, Nspace))
            self.atmos.cos2chi = f64_view_2(self.cos2chi)
            self.sin2chi = np.zeros((Nrays, Nspace))
            self.atmos.sin2chi = f64_view_2(self.sin2chi)


        self.vlos_mu = np.zeros((Nrays, Nspace))
        self.atmos.vlosMu = f64_view_2(self.vlos_mu)

        self.configure_bcs(atmos)
        self.update_projections()

    def configure_bcs(self, atmos):
        cdef int Nx = max(atmos.Nx, 1)
        cdef int Ny = max(atmos.Ny, 1)
        cdef int Nz = atmos.Nz

        cdef int Nbcx = Nz * Ny
        cdef int Nbcy = Nz * Nx
        cdef int Nbcz = Nx * Ny

        cdef int Nrays = self.Nrays
        s = atmos.structure
        cdef np.int32_t[:,::1] xLowerIdxs = self.py_atmos.x_lower_bc.index_vector
        self.atmos.xLowerBc = AtmosphericBoundaryCondition(BC_to_enum(s.x_lower_bc),
                                                           self.Nwave, Nrays, Nbcx,
                                                           BcIdxs(&xLowerIdxs[0,0],
                                                                  xLowerIdxs.shape[0],
                                                                  xLowerIdxs.shape[1]))
        cdef np.int32_t[:,::1] xUpperIdxs = self.py_atmos.x_upper_bc.index_vector
        self.atmos.xUpperBc = AtmosphericBoundaryCondition(BC_to_enum(s.x_upper_bc),
                                                           self.Nwave, Nrays, Nbcx,
                                                           BcIdxs(&xUpperIdxs[0,0],
                                                                  xUpperIdxs.shape[0],
                                                                  xUpperIdxs.shape[1]))
        cdef np.int32_t[:,::1] yLowerIdxs = self.py_atmos.y_lower_bc.index_vector
        self.atmos.yLowerBc = AtmosphericBoundaryCondition(BC_to_enum(s.y_lower_bc),
                                                           self.Nwave, Nrays, Nbcy,
                                                           BcIdxs(&yLowerIdxs[0,0],
                                                                  yLowerIdxs.shape[0],
                                                                  yLowerIdxs.shape[1]))
        cdef np.int32_t[:,::1] yUpperIdxs = self.py_atmos.y_upper_bc.index_vector
        self.atmos.yUpperBc = AtmosphericBoundaryCondition(BC_to_enum(s.y_upper_bc),
                                                           self.Nwave, Nrays, Nbcy,
                                                           BcIdxs(&yUpperIdxs[0,0],
                                                                  yUpperIdxs.shape[0],
                                                                  yUpperIdxs.shape[1]))
        cdef np.int32_t[:,::1] zLowerIdxs = self.py_atmos.z_lower_bc.index_vector
        self.atmos.zLowerBc = AtmosphericBoundaryCondition(BC_to_enum(s.z_lower_bc),
                                                           self.Nwave, Nrays, Nbcz,
                                                           BcIdxs(&zLowerIdxs[0,0],
                                                                  zLowerIdxs.shape[0],
                                                                  zLowerIdxs.shape[1]))
        cdef np.int32_t[:,::1] zUpperIdxs = self.py_atmos.z_upper_bc.index_vector
        self.atmos.zUpperBc = AtmosphericBoundaryCondition(BC_to_enum(s.z_upper_bc),
                                                           self.Nwave, Nrays, Nbcz,
                                                           BcIdxs(&zUpperIdxs[0,0],
                                                                  zUpperIdxs.shape[0],
                                                                  zUpperIdxs.shape[1]))

    def compute_bcs(self, LwSpectrum spect):
        cdef f64[:,:,::1] bc
        cdef int mu, la
        cdef F64View3D data
        cdef AtmosphericBoundaryCondition* abc

        if self.atmos.zLowerBc.type == CALLABLE:
            if np.all(self.py_atmos.z_lower_bc.index_vector == -1):
                abc = &self.atmos.zLowerBc
                bc = np.zeros((self.Nwave, abc.bcData.shape(1), abc.bcData.shape(2)))
            else:
                # NOTE: Call user-overridable hooks positionally, so subclasses written with
                # the pre-1.0 parameter names keep working.
                bc = self.py_atmos.z_lower_bc.compute_bc(self.py_atmos, spect)
            verify_bc_array_sizes(&self.atmos.zLowerBc, bc, 'z_lower_bc')
            data = f64_view_3(bc)
            self.atmos.zLowerBc.set_bc_data(data)

        if self.atmos.zUpperBc.type == CALLABLE:
            if np.all(self.py_atmos.z_upper_bc.index_vector == -1):
                abc = &self.atmos.zUpperBc
                bc = np.zeros((self.Nwave, abc.bcData.shape(1), abc.bcData.shape(2)))
            else:
                bc = self.py_atmos.z_upper_bc.compute_bc(self.py_atmos, spect)
            verify_bc_array_sizes(&self.atmos.zUpperBc, bc, 'z_upper_bc')
            data = f64_view_3(bc)
            self.atmos.zUpperBc.set_bc_data(data)

        if self.atmos.xLowerBc.type == CALLABLE:
            if np.all(self.py_atmos.x_lower_bc.index_vector == -1):
                abc = &self.atmos.xLowerBc
                bc = np.zeros((self.Nwave, abc.bcData.shape(1), abc.bcData.shape(2)))
            else:
                bc = self.py_atmos.x_lower_bc.compute_bc(self.py_atmos, spect)
            verify_bc_array_sizes(&self.atmos.xLowerBc, bc, 'x_lower_bc')
            data = f64_view_3(bc)
            self.atmos.xLowerBc.set_bc_data(data)

        if self.atmos.xUpperBc.type == CALLABLE:
            if np.all(self.py_atmos.x_upper_bc.index_vector == -1):
                abc = &self.atmos.xUpperBc
                bc = np.zeros((self.Nwave, abc.bcData.shape(1), abc.bcData.shape(2)))
            else:
                bc = self.py_atmos.x_upper_bc.compute_bc(self.py_atmos, spect)
            verify_bc_array_sizes(&self.atmos.xUpperBc, bc, 'x_upper_bc')
            data = f64_view_3(bc)
            self.atmos.xUpperBc.set_bc_data(data)

        if self.atmos.yLowerBc.type == CALLABLE:
            if np.all(self.py_atmos.y_lower_bc.index_vector == -1):
                abc = &self.atmos.yLowerBc
                bc = np.zeros((self.Nwave, abc.bcData.shape(1), abc.bcData.shape(2)))
            else:
                bc = self.py_atmos.y_lower_bc.compute_bc(self.py_atmos, spect)
            verify_bc_array_sizes(&self.atmos.yLowerBc, bc, 'y_lower_bc')
            data = f64_view_3(bc)
            self.atmos.yLowerBc.set_bc_data(data)

        if self.atmos.yUpperBc.type == CALLABLE:
            if np.all(self.py_atmos.y_upper_bc.index_vector == -1):
                abc = &self.atmos.yUpperBc
                bc = np.zeros((self.Nwave, abc.bcData.shape(1), abc.bcData.shape(2)))
            else:
                bc = self.py_atmos.y_upper_bc.compute_bc(self.py_atmos, spect)
            verify_bc_array_sizes(&self.atmos.yUpperBc, bc, 'y_upper_bc')
            data = f64_view_3(bc)
            self.atmos.yUpperBc.set_bc_data(data)

    def update_projections(self):
        '''
        Update all arrays of projected terms in the atmospheric model.
        '''
        self.atmos.update_projections()
        build_intersection_list(&self.atmos)

    def __getstate__(self):
        state = {}
        state['py_atmos'] = self.py_atmos
        state['x'] = self.py_atmos.x
        state['y'] = self.py_atmos.y
        state['z'] = self.py_atmos.z
        state['temperature'] = self.py_atmos.temperature
        state['ne'] = self.py_atmos.ne
        state['vx'] = self.py_atmos.vx
        state['vy'] = self.py_atmos.vy
        state['vz'] = self.py_atmos.vz
        state['vlos_mu'] = np.asarray(self.vlos_mu)
        try:
            state['B'] = self.py_atmos.B
            state['gamma_B'] = self.py_atmos.gamma_B
            state['chi_B'] = self.py_atmos.chi_B
            state['cos_gamma'] = np.asarray(self.cos_gamma)
            state['cos2chi'] = np.asarray(self.cos2chi)
            state['sin2chi'] = np.asarray(self.sin2chi)
        except AttributeError:
            state['B'] = None
            state['gamma_B'] = None
            state['chi_B'] = None
            state['cos_gamma'] = None
            state['cos2chi'] = None
            state['sin2chi'] = None
        state['vturb'] = self.py_atmos.vturb
        state['nh_tot'] = self.py_atmos.nh_tot
        state['muz'] = self.py_atmos.muz
        state['muy'] = self.py_atmos.muy
        state['mux'] = self.py_atmos.mux
        state['wmu'] = self.py_atmos.wmu
        state['Nwave'] = self.Nwave
        state['Ndim'] = self.Ndim
        state['Nx'] = self.Nx
        state['Ny'] = self.Ny
        state['Nz'] = self.Nz

        return state

    def __setstate__(self, state):
        self.py_atmos = state['py_atmos']
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
        self.vlos_mu = state['vlos_mu']
        self.atmos.vlosMu = f64_view_2(self.vlos_mu)
        if state['B'] is not None:
            self.B = state['B']
            self.atmos.B = f64_view(self.B)
            self.gamma_B = state['gamma_B']
            self.atmos.gammaB = f64_view(self.gamma_B)
            self.chi_B = state['chi_B']
            self.atmos.chiB = f64_view(self.chi_B)
            self.cos_gamma = state['cos_gamma']
            self.atmos.cosGamma = f64_view_2(self.cos_gamma)
            self.cos2chi = state['cos2chi']
            self.atmos.cos2chi = f64_view_2(self.cos2chi)
            self.sin2chi = state['sin2chi']
            self.atmos.sin2chi = f64_view_2(self.sin2chi)
        self.vturb = state['vturb']
        self.atmos.vturb = f64_view(self.vturb)
        self.nh_tot = state['nh_tot']
        self.atmos.nHTot = f64_view(self.nh_tot)
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
        cdef int Nrays = self.vlos_mu.shape[0]
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

        self.configure_bcs(self.py_atmos)
        build_intersection_list(&self.atmos)

    @property
    def Nspace(self):
        '''
        The number of points in the atmosphere.
        '''
        return self.atmos.Nspace

    @property
    def Nrays(self):
        '''
        The number of rays in the angular quadrature.
        '''
        return self.atmos.Nrays

    @property
    def Ndim(self):
        '''
        The dimensionality of the atmosphere.
        '''
        return self.atmos.Ndim

    @property
    def Nx(self):
        '''
        The number of points along the x dimension.
        '''
        return self.atmos.Nx

    @property
    def Ny(self):
        '''
        The number of points along the y dimension.
        '''
        return self.atmos.Ny

    @property
    def Nz(self):
        '''
        The number of points along the z dimension.
        '''
        return self.atmos.Nz

    @property
    def x(self):
        '''
        The x grid.
        '''
        return np.asarray(self.x)

    @property
    def y(self):
        '''
        The y grid.
        '''
        return np.asarray(self.y)

    @property
    def z(self):
        '''
        The z grid.
        '''
        return np.asarray(self.z)

    @property
    def height(self):
        '''
        The z (altitude) grid.
        '''
        return np.asarray(self.z)

    @property
    def temperature(self):
        '''
        The temperature structure of the atmospheric model (flat array).
        '''
        return np.asarray(self.temperature)

    @property
    def ne(self):
        '''
        The electron density structure of the atmospheric model (flat array).
        '''
        return np.asarray(self.ne)

    @property
    def vx(self):
        '''
        The x-velocity structure of the atmospheric model (flat array).
        '''
        return np.asarray(self.vx)

    @property
    def vy(self):
        '''
        The y-velocity structure of the atmospheric model (flat array).
        '''
        return np.asarray(self.vy)

    @property
    def vz(self):
        '''
        The z-velocity structure of the atmospheric model (flat array).
        '''
        return np.asarray(self.vz)

    @property
    def vlos(self):
        '''
        The z-velocity structure of the atmospheric model for 1D atmospheres
        (flat array).
        '''
        if self.py_atmos.Ndim > 1:
            raise ValueError('vlos is ambiguous when Ndim > 1, use vx, vy, or vz instead.')
        return np.asarray(self.vz)

    @property
    def vlos_mu(self):
        '''
        The projected line of sight veloctity for each ray in the atmosphere.
        '''
        return np.asarray(self.vlos_mu)

    @property
    def B(self):
        '''
        The magnetic field structure for the atmosphereic model (flat array).
        '''
        return np.asarray(self.B)

    @property
    def gamma_B(self):
        '''
        Magnetic field co-altitude.
        '''
        return np.asarray(self.gamma_B)

    @property
    def chi_B(self):
        '''
        Magnetic field azimuth
        '''
        return np.asarray(self.chi_B)

    @property
    def cos_gamma(self):
        '''
        cosine of gamma_B
        '''
        return np.asarray(self.cos_gamma)

    @property
    def cos2chi(self):
        '''
        cosine of 2*chi
        '''
        return np.asarray(self.cos2chi)

    @property
    def sin2chi(self):
        '''
        sine of 2*chi
        '''
        return np.asarray(self.sin2chi)

    @property
    def vturb(self):
        '''
        Microturbelent velocity structure of the atmospheric model.
        '''
        return np.asarray(self.vturb)

    @property
    def nh_tot(self):
        '''
        Total hydrogen number density strucutre.
        '''
        return np.asarray(self.nh_tot)

    @property
    def muz(self):
        '''
        Cosine of angle with z-axis for each ray.
        '''
        return np.asarray(self.muz)

    @property
    def muy(self):
        '''
        Cosine of angle with y-axis for each ray.
        '''
        return np.asarray(self.muy)

    @property
    def mux(self):
        '''
        Cosine of angle with x-axis for each ray.
        '''
        return np.asarray(self.mux)

    @property
    def wmu(self):
        '''
        Integration weights for angular quadrature.
        '''
        return np.asarray(self.wmu)

    # Deprecated names (to be removed in a future release).
    vlosMu = deprecated_alias('vlos_mu')
    gammaB = deprecated_alias('gamma_B')
    chiB = deprecated_alias('chi_B')
    cosGamma = deprecated_alias('cos_gamma')
    nHTot = deprecated_alias('nh_tot')
    pyAtmos = deprecated_alias('py_atmos')


cdef class BackgroundProvider:
    '''
    Base class for implementing background packages. Inherit from this to
    implement a new background scheme.

    Parameters
    ---------
    eq_pops : SpeciesStateTable
        The populations of all species present in the simulation.
    rad_set : RadiativeSet
        The atomic models and configuration data.
    wavelength : np.ndarray
        The array of wavelengths at which to compute the background.

    '''
    def __init__(self, eq_pops, rad_set, wavelength):
        pass

    # cpdef compute_background(self, LwAtmosphere atmos, f64[:,::1] chi, f64[:,::1] eta, f64[:,::1] sca):
    cpdef compute_background(self, LwAtmosphere atmos, chi, eta, sca):
        '''
        The function called by the backend to compute the background.

        Parameters
        ----------
        atmos : LwAtmosphere
            The atmospheric model.
        chi : np.ndarray
            Array in which to store the background opacity [Nlambda, Nspace].
        eta : np.ndarray
            Array in which to store the background emissivity [Nlambda,
            Nspace].
        sca : np.ndarray
            Array in which to store the background scattering [Nlambda,
            Nspace].
        '''
        raise NotImplementedError

cdef class BasicBackground(BackgroundProvider):
    '''
    Basic background implementation used by default in Lightweaver;
    equivalent to RH's treatment i.e. H- opacity, CH, OH, H2 continuum
    opacities if present, continua from all passive atoms in the
    RadiativeSet, Thomson and Rayleigh scattering (from H and He).
    '''
    cdef BackgroundData bd
    cdef object eq_pops
    cdef object rad_set

    cdef f64[::1] ch_pops
    cdef f64[::1] oh_pops
    cdef f64[::1] h2_pops
    cdef f64[::1] hmin_pops
    cdef f64[:,::1] h_pops

    cdef f64[::1] wavelength

    def __init__(self, eq_pops, rad_set, wavelength):
        super().__init__(eq_pops, rad_set, wavelength)
        self.eq_pops = eq_pops
        self.rad_set = rad_set

        if 'CH' in eq_pops:
            self.ch_pops = eq_pops['CH']
            self.bd.chPops = f64_view(self.ch_pops)
        if 'OH' in eq_pops:
            self.oh_pops = eq_pops['OH']
            self.bd.ohPops = f64_view(self.oh_pops)
        if 'H2' in eq_pops:
            self.h2_pops = eq_pops['H2']
            self.bd.h2Pops = f64_view(self.h2_pops)

        self.hmin_pops = eq_pops['H-']
        self.bd.hMinusPops = f64_view(self.hmin_pops)
        self.h_pops = eq_pops['H']
        self.bd.hPops = f64_view_2(self.h_pops)

        self.wavelength = wavelength
        self.bd.wavelength = f64_view(self.wavelength)

    # cpdef compute_background(self, LwAtmosphere atmos, f64[:,::1] chi, f64[:,::1] eta, f64[:,::1] sca):
    cpdef compute_background(self, LwAtmosphere atmos, chi_in, eta_in, sca_in):
        cdef int Nlambda = self.wavelength.shape[0]
        cdef int Nspace = atmos.Nspace
        cdef f64[:,::1] chi = chi_in
        cdef f64[:,::1] eta = eta_in
        cdef f64[:,::1] sca = sca_in

        # NOTE(cmo): Update h_pops in case it changed LTE<->NLTE
        self.h_pops = self.eq_pops['H']

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

        if 'H' in self.rad_set:
            h_pops = self.eq_pops['H']
            rayH = RayleighScatterer(atmos, self.rad_set['H'], h_pops)
            for la in range(self.wavelength.shape[0]):
                if rayH.scatter(self.wavelength[la], scaLine):
                    for k in range(atmos.Nspace):
                        sca[la, k] += scaLine[k]

        if 'He' in self.rad_set:
            hePops = self.eq_pops['He']
            rayHe = RayleighScatterer(atmos, self.rad_set['He'], hePops)
            for la in range(self.wavelength.shape[0]):
                if rayHe.scatter(self.wavelength[la], scaLine):
                    for k in range(atmos.Nspace):
                        sca[la, k] += scaLine[k]

    cpdef bf_opacities(self, LwAtmosphere atmos, f64[:,::1] chi, f64[:,::1] eta):
        atoms = self.rad_set.passive_atoms
        if len(atoms) == 0:
            return

        continua = []
        cdef f64 sigma0 = 32.0 / (3.0 * sqrt(3.0)) * Const.QElectron**2 / (4.0 * np.pi * Const.Epsilon0) / (Const.MElectron * Const.CLight) * Const.HPlanck / (2.0 * Const.ERydberg)
        for a in atoms:
            for c in a.continua:
                continua.append(c)

        cdef f64[:, ::1] alpha = np.zeros((self.wavelength.shape[0], len(continua)))
        cdef int i, la, k, Z
        cdef f64 n_eff, gbf_0, wav, edge, lambdaMin
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
        cdef f64[:,::1] n_star
        cdef f64[:,::1] n
        for i, c in enumerate(continua):
            n_star = self.eq_pops.atomic_pops[c.atom.element].n_star
            n = self.eq_pops.atomic_pops[c.atom.element].n

            ci = c.i
            cj = c.j
            for la in range(self.wavelength.shape[0]):
                twohnu3_c2 = twohc / self.wavelength[la]**3
                for k in range(atmos.Nspace):
                    gijk = n_star[ci, k] / n_star[cj, k] * expla[la, k]
                    chi[la, k] += alpha[la, i] * (1.0 - expla[la, k]) * n[ci, k]
                    eta[la, k] += twohnu3_c2 * gijk * alpha[la, i] * n[cj, k]

    def __getstate__(self):
        state = {}
        state['eq_pops'] = self.eq_pops
        state['rad_set'] = self.rad_set
        if 'CH' in self.eq_pops:
            state['ch_pops'] = self.eq_pops['CH']
        else:
            state['ch_pops'] = None

        if 'OH' in self.eq_pops:
            state['oh_pops'] = self.eq_pops['OH']
        else:
            state['oh_pops'] = None

        if 'H2' in self.eq_pops:
            state['h2_pops'] = self.eq_pops['H2']
        else:
            state['h2_pops'] = None

        state['hmin_pops'] = self.eq_pops['H-']
        state['h_pops'] = self.eq_pops['H']
        state['wavelength'] = np.asarray(self.wavelength)

        return state

    def __setstate__(self, state):
        self.eq_pops = state['eq_pops']
        self.rad_set = state['rad_set']

        if state['ch_pops'] is not None:
            self.ch_pops = state['ch_pops']
            self.bd.chPops = f64_view(self.ch_pops)
        if state['oh_pops'] is not None:
            self.oh_pops = state['oh_pops']
            self.bd.ohPops = f64_view(self.oh_pops)
        if state['h2_pops'] is not None:
            self.h2_pops = state['h2_pops']
            self.bd.h2Pops = f64_view(self.h2_pops)

        self.hmin_pops = state['hmin_pops']
        self.bd.hMinusPops = f64_view(self.hmin_pops)
        self.h_pops = state['h_pops']
        self.bd.hPops = f64_view_2(self.h_pops)

        self.wavelength = state['wavelength']
        self.bd.wavelength = f64_view(self.wavelength)

    @classmethod
    def _reconstruct(cls, state):
        o = cls.__new__(cls)
        o.__setstate__(state)
        return o

    def __reduce__(self):
        return self._reconstruct, (self.__getstate__(),)

cdef class FastBackground(BackgroundProvider):
    '''
    A faster implementation (due to C++ implementations) of BasicBackground
    supporting multiple threads.
    '''
    cdef BackgroundData bd
    cdef object eq_pops
    cdef object rad_set

    cdef f64[::1] ch_pops
    cdef f64[::1] oh_pops
    cdef f64[::1] h2_pops
    cdef f64[::1] hmin_pops
    cdef f64[:,::1] h_pops
    cdef f64[::1] wavelength

    cdef FastBackgroundContext ctx
    cdef int Nthreads

    def __init__(self, eq_pops, rad_set, wavelength, Nthreads=1):
        super().__init__(eq_pops, rad_set, wavelength)
        self.eq_pops = eq_pops
        self.rad_set = rad_set

        if 'CH' in eq_pops:
            self.ch_pops = eq_pops['CH']
            self.bd.chPops = f64_view(self.ch_pops)
        if 'OH' in eq_pops:
            self.oh_pops = eq_pops['OH']
            self.bd.ohPops = f64_view(self.oh_pops)
        if 'H2' in eq_pops:
            self.h2_pops = eq_pops['H2']
            self.bd.h2Pops = f64_view(self.h2_pops)

        self.hmin_pops = eq_pops['H-']
        self.bd.hMinusPops = f64_view(self.hmin_pops)
        self.h_pops = eq_pops['H']
        self.bd.hPops = f64_view_2(self.h_pops)

        self.wavelength = wavelength
        self.bd.wavelength = f64_view(self.wavelength)

        self.Nthreads = Nthreads
        self.ctx.initialise(self.Nthreads)

    cpdef compute_background(self, LwAtmosphere atmos, chi_in, eta_in, sca_in):
        cdef int Nlambda = self.wavelength.shape[0]
        cdef int Nspace = atmos.Nspace
        cdef f64[:,::1] chi = chi_in
        cdef f64[:,::1] eta = eta_in
        cdef f64[:,::1] sca = sca_in

        # NOTE(cmo): Update h_pops in case it changed LTE<->NLTE
        self.h_pops = self.eq_pops['H']

        # TODO(cmo): How UV fudge works here is a problem for future me.

        self.bd.chi = f64_view_2(chi)
        self.bd.eta = f64_view_2(eta)
        self.bd.scatt = f64_view_2(sca)

        cdef vector[BackgroundAtom] atoms
        cdef BackgroundAtom* atom
        passive_atoms = self.rad_set.passive_atoms
        # NOTE(cmo): This length should always be enough, but it's a tiny
        # amount of memory
        atoms.reserve(len(passive_atoms) + 2)
        # NOTE(cmo): Make sure all arrays remain backed by memory
        storage = []
        for a in passive_atoms:
            atoms.push_back(BackgroundAtom())
            atom = &atoms.back();
            atom.n = f64_view_2(self.eq_pops.atomic_pops[a.element].n)
            atom.nStar = f64_view_2(self.eq_pops.atomic_pops[a.element].n_star)
            atom.continua.reserve(len(a.continua))
            for c in a.continua:
                alpha = c.alpha(np.asarray(self.wavelength))
                storage.append(alpha)
                atom.continua.push_back(BackgroundContinuum(c.i, c.j, c.min_lambda, c.lambda_edge,
                                                            f64_view(alpha),
                                                            self.bd.wavelength))
            if a.element == PeriodicTable[1] or a.element == PeriodicTable[2]:
                atom.resonanceScatterers.reserve(len(a.lines))
                for l in a.lines:
                    if l.i == 0:
                        atom.resonanceScatterers.push_back(
                            ResonantRayleighLine(l.Aji,
                                                 l.j_level.g / l.i_level.g,
                                                 l.lambda0,
                                                 l.wavelength()[-1])
                                                 )
        for a in self.rad_set.active_atoms + self.rad_set.detailed_atoms:
            if a.element == PeriodicTable[1] or a.element == PeriodicTable[2]:
                atoms.push_back(BackgroundAtom())
                atom = &atoms.back();
                atom.n = f64_view_2(self.eq_pops.atomic_pops[a.element].n)
                atom.nStar = f64_view_2(self.eq_pops.atomic_pops[a.element].n_star)
                atom.resonanceScatterers.reserve(len(a.lines))
                for l in a.lines:
                    if l.i == 0:
                        atom.resonanceScatterers.push_back(
                            ResonantRayleighLine(l.Aji,
                                                 l.j_level.g / l.i_level.g,
                                                 l.lambda0,
                                                 l.wavelength()[-1])
                                                 )

        self.ctx.basic_background(&self.bd, &atmos.atmos)
        self.ctx.rayleigh_scatter(&self.bd, &atoms, &atmos.atmos)
        self.ctx.bf_opacities(&self.bd, &atoms, &atmos.atmos)

        cdef int la, k
        for la in range(Nlambda):
            for k in range(Nspace):
                chi[la, k] += sca[la, k]

    def __getstate__(self):
        state = {}
        state['eq_pops'] = self.eq_pops
        state['rad_set'] = self.rad_set
        if 'CH' in self.eq_pops:
            state['ch_pops'] = self.eq_pops['CH']
        else:
            state['ch_pops'] = None

        if 'OH' in self.eq_pops:
            state['oh_pops'] = self.eq_pops['OH']
        else:
            state['oh_pops'] = None

        if 'H2' in self.eq_pops:
            state['h2_pops'] = self.eq_pops['H2']
        else:
            state['h2_pops'] = None

        state['hmin_pops'] = self.eq_pops['H-']
        state['h_pops'] = self.eq_pops['H']
        state['wavelength'] = np.asarray(self.wavelength)
        state['Nthreads'] = self.Nthreads

        return state

    def __setstate__(self, state):
        self.eq_pops = state['eq_pops']
        self.rad_set = state['rad_set']

        if state['ch_pops'] is not None:
            self.ch_pops = state['ch_pops']
            self.bd.chPops = f64_view(self.ch_pops)
        if state['oh_pops'] is not None:
            self.oh_pops = state['oh_pops']
            self.bd.ohPops = f64_view(self.oh_pops)
        if state['h2_pops'] is not None:
            self.h2_pops = state['h2_pops']
            self.bd.h2Pops = f64_view(self.h2_pops)

        self.hmin_pops = state['hmin_pops']
        self.bd.hMinusPops = f64_view(self.hmin_pops)
        self.h_pops = state['h_pops']
        self.bd.hPops = f64_view_2(self.h_pops)

        self.wavelength = state['wavelength']
        self.bd.wavelength = f64_view(self.wavelength)
        self.Nthreads = state['Nthreads']
        self.ctx.initialise(self.Nthreads)

    @classmethod
    def _reconstruct(cls, state):
        o = cls.__new__(cls)
        o.__setstate__(state)
        return o

    def __reduce__(self):
        return self._reconstruct, (self.__getstate__(),)


cdef class LwBackground:
    '''
    Storage and driver for the background computations in Lightweaver. The
    storage is allocated and managed by this class, before being passed to
    C++ when necessary. This class is also responsible for calling the
    BackgroundProvider instance used (by default FastBackground with one thread).
    '''
    cdef Background background
    cdef object eq_pops
    cdef object rad_set

    cdef BackgroundProvider provider

    cdef f64[::1] wavelength
    cdef f64[:,::1] chi
    cdef f64[:,::1] eta
    cdef f64[:,::1] sca

    def __init__(self, atmosphere, eq_pops, rad_set, wavelength, provider=None):
        cdef LwAtmosphere atmos = atmosphere
        self.eq_pops = eq_pops
        self.rad_set = rad_set

        self.wavelength = wavelength

        cdef int Nlambda = self.wavelength.shape[0]
        cdef int Nspace = atmos.Nspace

        self.chi = np.zeros((Nlambda, Nspace))
        self.eta = np.zeros((Nlambda, Nspace))
        self.sca = np.zeros((Nlambda, Nspace))

        if provider is None:
            self.provider = FastBackground(eq_pops, rad_set, wavelength, Nthreads=1)
        else:
            self.provider = provider(eq_pops, rad_set, wavelength)

        chiPy = np.asarray(self.chi)
        etaPy = np.asarray(self.eta)
        scaPy = np.asarray(self.sca)
        # NOTE: Call user-overridable hooks positionally, so subclasses written with the
        # pre-1.0 parameter names keep working.
        self.provider.compute_background(atmos, chiPy, etaPy, scaPy)

        self.background.chi = f64_view_2(self.chi)
        self.background.eta = f64_view_2(self.eta)
        self.background.sca = f64_view_2(self.sca)

    cpdef update_background(self, LwAtmosphere atmos):
        '''
        Recompute the background opacities, perhaps in the case where, for
        example, the atmospheric parameters have been updated.

        Parameters
        ----------
        atmos : LwAtmosphere
            The atmosphere in which to compute the background opacities and
            emissivities.
        '''
        chiPy = np.asarray(self.chi)
        etaPy = np.asarray(self.eta)
        scaPy = np.asarray(self.sca)
        self.provider.compute_background(atmos, chiPy, etaPy, scaPy)

    def __getstate__(self):
        state = {}
        state['eq_pops'] = self.eq_pops
        state['rad_set'] = self.rad_set
        state['provider'] = self.provider
        state['wavelength'] = np.asarray(self.wavelength)
        state['chi'] = np.asarray(self.chi)
        state['eta'] = np.asarray(self.eta)
        state['sca'] = np.asarray(self.sca)

        return state

    def __setstate__(self, state):
        self.eq_pops = state['eq_pops']
        self.rad_set = state['rad_set']
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
        '''
        The background opacity [Nlambda, Nspace].
        '''
        return np.asarray(self.chi)

    @property
    def eta(self):
        '''
        The background eta [Nlambda, Nspace].
        '''
        return np.asarray(self.eta)

    @property
    def sca(self):
        '''
        The background scattering [Nlambda, Nspace].
        '''
        return np.asarray(self.sca)


cdef class RayleighScatterer:
    '''
    For computing Rayleigh scattering, used by BasicBackground.
    '''
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
                f = l.Aji * (l.j_level.g / g0) * (l.lambda0 * Const.NM_TO_M)**2 / self.C
                fomega += f * lambda2**2

        cdef f64 sigmaRayleigh = self.sigmaE * fomega

        cdef int k
        for k in range(sca.shape[0]):
            sca[k] = sigmaRayleigh * self.pops[0, k]

        return True

cdef class LwTransition:
    '''
    Storage and access to transition data used by backend. Instantiated by
    Context.

    Parameters
    ----------
    trans : AtomicTransition
        The transition model object.
    compAtom : LwAtom
        The computational atom to which this computational transition
        belongs.
    atmos : LwAtmosphere
        The computational atmosphere in which this transition is to be used.
    spect : SpectrumConfiguration
        The spectral configuration of the simulation.

    Attributes
    ----------
    trans_model : AtomicTransition
        The transition model object.
    '''
    cdef Transition trans
    cdef f64[:, :, :, ::1] phi
    cdef f64[:, :, :, ::1] phi_Q
    cdef f64[:, :, :, ::1] phi_U
    cdef f64[:, :, :, ::1] phi_V
    cdef f64[:, :, :, ::1] psi_Q
    cdef f64[:, :, :, ::1] psi_U
    cdef f64[:, :, :, ::1] psi_V
    cdef f64[::1] wphi
    cdef f64[::1] alpha
    cdef f64[::1] wavelength
    cdef i8[::1] active
    cdef f64[::1] Qelast
    cdef f64[::1] a_damp
    cdef f64[:, ::1] rho_prd
    cdef f64[::1] Rij
    cdef f64[::1] Rji
    cdef public object trans_model
    cdef LwAtmosphere atmos
    cdef object spect
    cdef public LwAtom atom

    def __init__(self, trans, compAtom, atmos, spect):
        self.trans_model = trans
        cdef LwAtom atom = compAtom
        self.atom = atom
        cdef LwAtmosphere a = atmos
        self.atmos = a
        self.spect = spect
        trans_id = trans.trans_id
        self.wavelength = spect.trans_wavelengths[trans_id]
        self.trans.wavelength = f64_view(self.wavelength)
        self.trans.i = trans.i
        self.trans.j = trans.j
        self.trans.polarised = False
        Nblue = spect.blue_idx[trans_id]
        self.trans.Nblue = Nblue
        Nred = spect.red_idx[trans_id]
        self.trans.Nred = Nred
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
            self.a_damp = np.zeros(Nspace)
            self.trans.Qelast = f64_view(self.Qelast)
            self.trans.aDamp = f64_view(self.a_damp)
            self.phi = np.zeros((Nlambda, Nrays, 2, Nspace))
            self.wphi = np.zeros(Nspace)
            self.trans.phi = f64_view_4(self.phi)
            self.trans.wphi = f64_view(self.wphi)
            if trans.type == LineType.PRD:
                self.rho_prd = np.ones((Nlambda, Nspace))
                self.trans.rhoPrd = f64_view_2(self.rho_prd)
        else:
            self.trans.type = CONTINUUM
            self.alpha = trans.alpha(np.asarray(self.wavelength))
            self.trans.alpha = f64_view(self.alpha)
            self.trans.dopplerWidth = 1.0
            self.trans.lambda0 = trans.lambda0

        self.active = spect.active_wavelengths[trans_id].astype(np.int8)
        self.trans.active = BoolView(<bool_t*>&self.active[0], self.active.shape[0])

        atomicState = self.atom.model_pops
        self.Rij = atomicState.radiative_rates[(self.trans.i, self.trans.j)]
        self.Rji = atomicState.radiative_rates[(self.trans.j, self.trans.i)]
        self.trans.Rij = f64_view(self.Rij)
        self.trans.Rji = f64_view(self.Rji)

    def __getstate__(self):
        state = {}
        state['atmos'] = self.atmos
        state['atom'] = self.atom
        state['spect'] = self.spect
        state['trans_model'] = self.trans_model
        state['type'] = self.type
        state['Nblue'] = self.trans.Nblue
        state['Nred'] = self.trans.Nred
        trans_id = self.trans_model.trans_id
        state['wavelength'] = self.spect.trans_wavelengths[trans_id]
        state['active'] = np.asarray(self.active)
        model_pops = self.atom.model_pops
        state['Rij'] = model_pops.radiative_rates[(self.trans.i, self.trans.j)]
        state['Rji'] = model_pops.radiative_rates[(self.trans.j, self.trans.i)]
        state['polarised'] = False
        if self.type == 'Line':
            state['phi'] = np.asarray(self.phi)
            try:
                state['phi_Q'] = np.asarray(self.phi_Q)
                state['phi_U'] = np.asarray(self.phi_U)
                state['phi_V'] = np.asarray(self.phi_V)
                state['psi_Q'] = np.asarray(self.psi_Q)
                state['psi_U'] = np.asarray(self.psi_U)
                state['psi_V'] = np.asarray(self.psi_V)
                state['polarised'] = True
            except AttributeError:
                state['phi_Q'] = None
                state['phi_U'] = None
                state['phi_V'] = None
                state['psi_Q'] = None
                state['psi_U'] = None
                state['psi_V'] = None

            state['wphi'] = np.asarray(self.wphi)
            state['Qelast'] = np.asarray(self.Qelast)
            state['a_damp'] = np.asarray(self.a_damp)
            try:
                state['rho_prd'] = np.asarray(self.rho_prd)
            except AttributeError:
                state['rho_prd'] = None

        else:
            state['alpha'] = np.asarray(self.alpha)
        return state

    def __setstate__(self, state):
        self.trans_model = state['trans_model']
        trans = self.trans_model
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
        self.trans.Nred = state['Nred']
        self.trans.polarised = state['polarised']

        if state['type'] == 'Line':
            self.trans.type = LINE
            self.trans.Aji = trans.Aji
            self.trans.Bji = trans.Bji
            self.trans.Bij = trans.Bij
            self.trans.lambda0 = trans.lambda0
            self.trans.dopplerWidth = Const.CLight / self.trans.lambda0
            self.Qelast = state['Qelast']
            self.a_damp = state['a_damp']
            self.trans.Qelast = f64_view(self.Qelast)
            self.trans.aDamp = f64_view(self.a_damp)
            self.phi = state['phi']
            self.wphi = state['wphi']
            self.trans.phi = f64_view_4(self.phi)
            self.trans.wphi = f64_view(self.wphi)
            if state['rho_prd'] is not None:
                self.rho_prd = state['rho_prd']
                self.trans.rhoPrd = f64_view_2(self.rho_prd)

            if state['polarised']:
                self.phi_Q = state['phi_Q']
                self.phi_U = state['phi_U']
                self.phi_V = state['phi_V']
                self.psi_Q = state['psi_Q']
                self.psi_U = state['psi_U']
                self.psi_V = state['psi_V']
                self.trans.phiQ = f64_view_4(self.phi_Q)
                self.trans.phiU = f64_view_4(self.phi_U)
                self.trans.phiV = f64_view_4(self.phi_V)
                self.trans.psiQ = f64_view_4(self.psi_Q)
                self.trans.psiU = f64_view_4(self.psi_U)
                self.trans.psiV = f64_view_4(self.psi_V)
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

    @accepts_old_kwargs
    def load_rates_prd_from_state(self, prev_state, preserve_profiles=True):

        np.asarray(self.Rij)[:] = prev_state['Rij']
        np.asarray(self.Rji)[:] = prev_state['Rji']

        if self.type == 'Continuum':
            return

        cdef int k
        if self.wavelength.shape == prev_state['wavelength'].shape \
           and np.all(self.wavelength == prev_state['wavelength']):
            if prev_state['rho_prd'] is not None:
                np.asarray(self.rho_prd)[:] = prev_state['rho_prd']

            if preserve_profiles:
                np.asarray(self.phi)[:] = prev_state['phi']
                if prev_state['phi_Q'] is not None:
                    np.asarray(self.phi_Q)[:] = prev_state['phi_Q']
                    np.asarray(self.phi_U)[:] = prev_state['phi_U']
                    np.asarray(self.phi_V)[:] = prev_state['phi_V']
                    np.asarray(self.psi_Q)[:] = prev_state['psi_Q']
                    np.asarray(self.psi_U)[:] = prev_state['psi_U']
                    np.asarray(self.psi_V)[:] = prev_state['psi_V']

        else:
            if prev_state['rho_prd'] is not None:
                for k in range(prev_state['rho_prd'].shape[1]):
                    np.asarray(self.rho_prd)[:, k] = np.interp(self.wavelength, prev_state['wavelength'], prev_state['rho_prd'][:, k])


    def compute_phi(self):
        '''
        Computes the line profile phi (phi_num in the technical report), by
        calling compute_phi on the line object. Provides a callback to the
        default Voigt implementation used in the backend.
        Does nothing if called on a continuum.
        '''
        if self.type == 'Continuum':
            return

        cdef Atmosphere* atmos = &self.atmos.atmos
        callbackUsed = False
        def default_voigt_callback(f64[::1] a_damp, f64[::1] v_broad):
            cdef F64View aDampView = f64_view(a_damp)
            cdef F64View vBroadView = f64_view(v_broad)
            self.trans.compute_phi(atmos[0], aDampView, vBroadView)
            nonlocal callbackUsed
            callbackUsed = True
            return np.asarray(self.phi)

        state = LineProfileState(wavelength=np.asarray(self.wavelength),
                                 vlos_mu=np.asarray(self.atmos.vlos_mu),
                                 atmos=self.atmos.py_atmos,
                                 eq_pops=self.atom.eq_pops,
                                 default_voigt_callback=default_voigt_callback,
                                 v_broad=self.atom.v_broad)
        profile = self.trans_model.compute_phi(state)

        cdef f64[:,:,:,::1] phi = profile.phi
        cdef f64[::1] Qelast = profile.Qelast
        cdef f64[::1] a_damp = profile.a_damp
        if not callbackUsed:
            self.phi[...] = phi
        self.Qelast[...] = Qelast
        self.a_damp[...] = a_damp

        self.trans.compute_wphi(self.atmos.atmos)

    cpdef compute_polarised_profiles(self):
        '''
        Compute the polarised line profiles (all of phi, phi_{Q, U, V}, and
        psi_{Q, U, V}) for a Voigt line, this currently doesn't support
        non-standard line profile types, but could do so quite simply by
        following compute_phi.
        Does nothing if the transitions is a continuum or the line is not
        polarisable.
        By calling this and iterating the Context as usual, a field
        '''
        if self.type == 'Continuum':
            return

        if not self.trans_model.polarisable:
            return

        cdef int Nlambda = self.wavelength.shape[0]
        cdef int Nrays = self.atmos.Nrays
        cdef int Nspace = self.atmos.Nspace
        try:
            self.phi_Q
        except AttributeError:
            self.phi_Q = np.zeros((Nlambda, Nrays, 2, Nspace))
            self.phi_U = np.zeros((Nlambda, Nrays, 2, Nspace))
            self.phi_V = np.zeros((Nlambda, Nrays, 2, Nspace))
            self.psi_Q = np.zeros((Nlambda, Nrays, 2, Nspace))
            self.psi_U = np.zeros((Nlambda, Nrays, 2, Nspace))
            self.psi_V = np.zeros((Nlambda, Nrays, 2, Nspace))
            self.trans.phiQ = f64_view_4(self.phi_Q)
            self.trans.phiU = f64_view_4(self.phi_U)
            self.trans.phiV = f64_view_4(self.phi_V)
            self.trans.psiQ = f64_view_4(self.psi_Q)
            self.trans.psiU = f64_view_4(self.psi_U)
            self.trans.psiV = f64_view_4(self.psi_V)

        self.trans.polarised = True

        cdef LwAtom atom = self.atom
        a_damp, Qelast = self.trans_model.damping(self.atmos.py_atmos, atom.eq_pops)

        cdef Atmosphere* atmos = &self.atmos.atmos
        cdef int i
        for i in range(self.Qelast.shape[0]):
            self.Qelast[i] = Qelast[i]
            self.a_damp[i] = a_damp[i]

        z = self.trans_model.zeeman_components()
        cdef LwZeemanComponents zc = LwZeemanComponents(z)

        self.trans.compute_polarised_profiles(atmos[0], self.trans.aDamp, atom.atom.vBroad, zc.zc)

    cpdef recompute_gII(self):
        '''
        Triggers lazy recalculation of gII for this line (if PRD).
        '''
        self.trans.recompute_gII()

    @accepts_old_kwargs
    def uv(self, int la, int mu, bool_t to_obs, f64[::1] Uji not None,
           f64[::1] Vij not None, f64[::1] Vji not None):
        '''
        Thin wrapper for computing U and V using the core. Must be called
        after `atom.setup_wavelength(la)`, and Uji, Vij, Vji must tbe the
        proper size, as no verification is performed.

        Parameters
        ----------
        la : int
            The wavelength index at which to compute U and V.
        mu : int
            The angle index at which to compute U and V.
        Uji, Vij, Vji : np.ndarray
            Storage arrays for the result.
        '''
        # TODO(cmo): Allow these to take None, and allocate if they are. Then
        # return in some UV datastruct
        cdef bint obs = to_obs
        cdef F64View cUji = f64_view(Uji)
        cdef F64View cVij = f64_view(Vij)
        cdef F64View cVji = f64_view(Vji)

        self.trans.uv(la, mu, obs, cUji, cVij, cVji)

    @property
    def j_level(self):
        '''
        Access the upper level on the model object.
        '''
        return self.trans_model.j_level

    @property
    def i_level(self):
        '''
        Access the lower level on the model object.
        '''
        return self.trans_model.i_level

    @property
    def j(self):
        '''
        Index of upper level.
        '''
        return self.trans_model.j

    @property
    def i(self):
        '''
        Index of lower level.
        '''
        return self.trans_model.i

    @property
    def Aji(self):
        '''
        Einstein A for transition.
        '''
        return self.trans.Aji

    @property
    def Bji(self):
        '''
        Einstein Bji for transition.
        '''
        return self.trans.Bji

    @property
    def Bij(self):
        '''
        Einstein Bij for transition.
        '''
        return self.trans.Bij

    @property
    def Nblue(self):
        '''
        Index into global wavelength grid where this transition's local grid
        starts.
        '''
        return self.trans.Nblue

    @property
    def lambda0(self):
        '''
        Line rest wavelength or continuum edge wavelength.
        '''
        return self.trans.lambda0

    @property
    def wphi(self):
        '''
        Multiplicative inverse of integrated line profile at each location in
        the atmosphere, used to ensure terms based on integration across the
        entire line profile (e.g. in the Gamma matrix) are correctly
        normalised.
        '''
        return np.asarray(self.wphi)

    @property
    def phi(self):
        '''
        Numerical line profile. AttributeError for continua.
        '''
        return np.asarray(self.phi)

    @property
    def phi_Q(self):
        return np.asarray(self.phi_Q)

    @property
    def phi_U(self):
        return np.asarray(self.phi_U)

    @property
    def phi_V(self):
        return np.asarray(self.phi_V)

    @property
    def psi_Q(self):
        return np.asarray(self.psi_Q)

    @property
    def psi_U(self):
        return np.asarray(self.psi_U)

    @property
    def psi_V(self):
        return np.asarray(self.psi_V)

    @property
    def Rij(self):
        '''
        Upwards radiative rates for the transition throughout the atmosphere.
        '''
        return np.asarray(self.Rij)

    @property
    def Rji(self):
        '''
        Downward radiative rates for the transition throughout the atmosphere.
        '''
        return np.asarray(self.Rji)

    @property
    def rho_prd(self):
        '''
        Ratio of emission to absorption profiles throughout the atmosphere,
        in the case of PRD lines.
        '''
        return np.asarray(self.rho_prd)

    @property
    def alpha(self):
        '''
        The wavelength-dependent cross-section for a continuum.
        AttributeError for lines.
        '''
        return np.asarray(self.alpha)

    @property
    def wavelength(self):
        '''
        The transition's local wavelength grid.
        '''
        return np.asarray(self.wavelength)

    @property
    def active(self):
        '''
        The active wavelength mask for this transition.
        '''
        return np.asarray(self.active).astype(np.bool)

    @property
    def Qelast(self):
        '''
        The elastic collision rate for this transition in the atmosphere,
        needed for PRD.
        '''
        return np.asarray(self.Qelast)

    @property
    def a_damp(self):
        '''
        The Voigt damping parameter for this transition in the atmosphere.
        '''
        return np.asarray(self.a_damp)

    @property
    def polarisable(self):
        '''
        The polarisability of the transition, based on model data.
        '''
        return self.trans_model.polarisable

    @property
    def type(self):
        '''
        The type of transition (Line or Continuum) as a str.
        '''
        if self.trans.type == LINE:
            return 'Line'
        else:
            return 'Continuum'

    # Deprecated names (to be removed in a future release).
    jLevel = deprecated_alias('j_level')
    iLevel = deprecated_alias('i_level')
    phiQ = deprecated_alias('phi_Q')
    phiU = deprecated_alias('phi_U')
    phiV = deprecated_alias('phi_V')
    psiQ = deprecated_alias('psi_Q')
    psiU = deprecated_alias('psi_U')
    psiV = deprecated_alias('psi_V')
    rhoPrd = deprecated_alias('rho_prd')
    aDamp = deprecated_alias('a_damp')
    transModel = deprecated_alias('trans_model')

cdef class LwZeemanComponents:
    '''
    Stores the Zeeman components to be passed to the backend, only exists
    transiently.
    '''
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
    '''
    Storage and access to computational atomic data used by backend. Sets up
    the computations transitions (LwTransition) present on the model.
    Instantiated by Context.

    Attributes
    ----------
    atomic_model : AtomicModel
        The atomic model object associated with this computational atom.
    model_pops : AtomicState
        The population data for this species, in a python accessible form.

    Parameters
    ----------
    atom : AtomicModel
        The atomic model object associated with this computational atom.
    atmos : LwAtmosphere
        The computational atmosphere to be used in the simulation.
    eq_pops : SpeciesStateTable
        The population of species present in the simulation.
    spect : SpectrumConfiguration
        The configuration of the spectral grids.
    background : LwBackground
        The background opacity terms, currently only used in the case of
        escape probability initial solution.
    detailed : bool, optional
        Whether the atom is in detailed static or fully active mode (default:
        False).
    init_sol : InitialSolution, optional
        The initial solution to use for the atomic populations (default: LTE).
    ng_options : NgOptions, optional
        The Ng acceleration options (default: None)
    conserve_charge : bool, optional
        Whether to conserve charge whilst setting populations from escape
        probability (ignored otherwise) (default: False).
    fs_iter_scheme_properties : dict, optional
        The properties of the FsIterScheme used as a dict, can be obtained from
        the `FsIterSchemeManager`. Only necessary keys are boolean
        `defaultWlaGijStorage` and `defaultPerAtomStorage` to determine
        allocation of `wla`, `gij`, `eta`, `U`, and `chi` on the underlying
        object. If not supplied, both of these default to True.
    '''
    cdef Atom atom
    cdef f64[::1] v_broad
    cdef f64[:,:,::1] Gamma
    cdef f64[:,:,::1] C
    cdef f64[::1] n_total
    cdef f64[:,::1] n_star
    cdef f64[:,::1] n
    cdef f64[::1] stages

    cdef public object atomic_model
    cdef public object model_pops
    cdef LwAtmosphere atmos
    cdef object eq_pops
    cdef list trans
    cdef bool_t detailed
    cdef dict fs_iter_scheme_properties

    def __init__(self, atom, atmos, eq_pops, spect, background,
                 detailed=False, init_sol=None, ng_options=None,
                 conserve_charge=False, fs_iter_scheme_properties=None):
        self.atomic_model = atom
        self.detailed = detailed
        cdef LwAtmosphere a = atmos
        self.atmos = a
        self.atom.atmos = &a.atmos
        self.eq_pops = eq_pops
        model_pops = eq_pops.atomic_pops[atom.element]
        self.model_pops = model_pops

        self.v_broad = atom.v_broad(atmos)
        self.atom.vBroad = f64_view(self.v_broad)
        self.n_total = model_pops.n_total
        self.atom.nTotal = f64_view(self.n_total)

        self.trans = []
        for t in atom.transitions:
            if spect.active_trans[t.trans_id]:
                self.trans.append(LwTransition(t, self, atmos, spect))

        cdef LwTransition lt
        for lt in self.trans:
            self.atom.trans.push_back(&lt.trans)

        cdef int Nlevel = len(atom.levels)
        cdef int Ntrans = len(self.trans)
        self.atom.Nlevel = Nlevel
        self.atom.Ntrans = Ntrans

        cdef bool_t defaultPerAtomStorage = True
        cdef bool_t defaultWlaGijStorage = True
        if fs_iter_scheme_properties is not None:
            self.fs_iter_scheme_properties = fs_iter_scheme_properties
            defaultPerAtomStorage = fs_iter_scheme_properties['defaultPerAtomStorage']
            defaultWlaGijStorage = fs_iter_scheme_properties['defaultWlaGijStorage']
        else:
            self.fs_iter_scheme_properties = {
                'defaultPerAtomStorage': defaultPerAtomStorage,
                'defaultWlaGijStorage': defaultPerAtomStorage
            }

        if not self.detailed:
            self.Gamma = np.zeros((Nlevel, Nlevel, atmos.Nspace))
            self.atom.Gamma = f64_view_3(self.Gamma)

            self.C = np.zeros((Nlevel, Nlevel, atmos.Nspace))
            self.atom.C = f64_view_3(self.C)

        self.atom.init_scratch(self.atmos.Nspace, detailed,
                               defaultWlaGijStorage, defaultPerAtomStorage)

        self.stages = np.array([l.stage for l in self.atomic_model.levels], dtype=np.float64)
        self.atom.stages = f64_view(self.stages)
        self.n_star = model_pops.n_star
        self.atom.nStar = f64_view_2(self.n_star)

        doInitSol = True
        self.n = model_pops.n
        self.atom.n = f64_view_2(self.n)

        if self.detailed:
            doInitSol = False
            ng_options = None

        if init_sol is None:
            init_sol = InitialSolution.Lte

        if doInitSol and init_sol == InitialSolution.Zero:
            raise ValueError('Zero radiation InitialSolution not currently supported')

        if doInitSol and init_sol == InitialSolution.EscapeProbability and Ntrans > 0:
            self.set_pops_escape_probability(self.atmos, background, conserve_charge=conserve_charge)

        cdef NgArgs args
        if ng_options is not None:
            args.nOrder = ng_options.Norder
            args.nPeriod = ng_options.Nperiod
            args.nDelay = ng_options.Ndelay
            args.threshold = ng_options.threshold
            args.lowerThreshold = ng_options.lower_threshold
        else:
            args.nOrder = 0
            args.nPeriod = 0
            args.nDelay = 0
            args.threshold = 0.0
            args.lowerThreshold = 0.0

        self.atom.ng = Ng(args, self.atom.n.flatten())

    def __getstate__(self):
        state = {}
        state['atomic_model'] = self.atomic_model
        state['model_pops'] = self.model_pops
        state['atmos'] = self.atmos
        state['eq_pops'] = self.eq_pops
        state['trans'] = self.trans
        state['detailed'] = self.detailed
        state['v_broad'] = np.asarray(self.v_broad)
        state['n_total'] = self.model_pops.n_total
        state['n_star'] = self.model_pops.n_star
        state['n'] = self.model_pops.n
        state['stages'] = np.asarray(self.stages)
        state['Ng'] = (self.atom.ng.Norder, self.atom.ng.Nperiod, self.atom.ng.Ndelay, self.atom.ng.threshold, self.atom.ng.lowerThreshold)
        if self.detailed:
            state['Gamma'] = None
            state['C'] = None
        else:
            state['Gamma'] = np.asarray(self.Gamma)
            state['C'] = np.asarray(self.C)
        state['fs_iter_scheme_properties'] = self.fs_iter_scheme_properties

        return state

    def __setstate__(self, state):
        self.atomic_model = state['atomic_model']
        self.model_pops = state['model_pops']
        cdef LwAtmosphere a = state['atmos']
        self.atmos = a
        self.atom.atmos = &a.atmos
        self.eq_pops = state['eq_pops']

        self.detailed = state['detailed']

        self.v_broad = state['v_broad']
        self.atom.vBroad = f64_view(self.v_broad)
        self.n_total = state['n_total']
        self.atom.nTotal = f64_view(self.n_total)

        self.trans = state['trans']
        cdef LwTransition lt
        for lt in self.trans:
            self.atom.trans.push_back(&lt.trans)

        cdef int Nlevel = len(self.atomic_model.levels)
        cdef int Ntrans = len(self.trans)
        self.atom.Nlevel = Nlevel
        self.atom.Ntrans = Ntrans

        if not self.detailed:
            self.Gamma = state['Gamma']
            self.atom.Gamma = f64_view_3(self.Gamma)

            self.C = state['C']
            self.atom.C = f64_view_3(self.C)

        self.stages = state['stages']
        self.atom.stages = f64_view(self.stages)
        self.n_star = state['n_star']
        self.atom.nStar = f64_view_2(self.n_star)
        self.n = state['n']
        self.atom.n = f64_view_2(self.n)

        ng = state['Ng']
        cdef NgArgs args
        args.nOrder = ng[0]
        args.nPeriod = ng[1]
        args.nDelay = ng[2]
        args.threshold = ng[3]
        args.lowerThreshold = ng[4]
        self.atom.ng = Ng(args, self.atom.n.flatten())
        self.fs_iter_scheme_properties = state['fs_iter_scheme_properties']
        cdef bool_t defaultPerAtomStorage = self.fs_iter_scheme_properties['defaultPerAtomStorage']
        cdef bool_t defaultWlaGijStorage = self.fs_iter_scheme_properties['defaultWlaGijStorage']
        self.atom.init_scratch(self.atmos.Nspace, self.detailed,
                               defaultWlaGijStorage, defaultPerAtomStorage)

    @accepts_old_kwargs
    def load_pops_rates_prd_from_state(self, prev_state, pops_only=False, preserve_profiles=False):
        cdef NgArgs args
        if not self.detailed:
            np.asarray(self.n)[:] = prev_state['n']
            ng = prev_state['Ng']
            args.nOrder = ng[0]
            args.nPeriod = ng[1]
            args.nDelay = ng[2]
            args.threshold = ng[3]
            args.lowerThreshold = ng[4]
            self.atom.ng = Ng(args, self.atom.n.flatten())

        if pops_only:
            return

        cdef LwTransition t
        cdef int i
        for i, t in enumerate(self.trans):
            for st in prev_state['trans']:
                if st.trans_model.i == t.i and st.trans_model.j == t.j:
                    t.load_rates_prd_from_state(st.__getstate__(), preserve_profiles=preserve_profiles)
                    break

    @accepts_old_kwargs
    def compute_collisions(self, fill_diagonal=False):
        cdef np.ndarray[np.double_t, ndim=3] C = np.asarray(self.C)
        C.fill(0.0)
        for col in self.atomic_model.collisions:
            # NOTE: Call user-overridable hooks positionally, so subclasses written with the
            # pre-1.0 parameter names keep working.
            col.compute_rates(self.atmos.py_atmos, self.eq_pops, C)
        C[C < 0.0] = 0.0

        if not fill_diagonal:
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


    cpdef set_pops_escape_probability(self, LwAtmosphere a, LwBackground bg, conserve_charge=False, int Niter=100):
        cdef np.ndarray[np.double_t, ndim=3] Gamma
        cdef np.ndarray[np.double_t, ndim=3] C
        cdef f64 delta
        cdef NgChange maxChange
        cdef int k
        cdef np.ndarray[np.double_t, ndim=1] deltaNe

        self.compute_collisions()
        Gamma = np.asarray(self.Gamma)
        C = np.asarray(self.C)

        if conserve_charge:
            prevN = np.copy(self.n)

        cdef NgArgs args
        args.nOrder = 0
        args.nPeriod = 0
        args.nDelay = 0
        args.threshold = 0.0
        args.lowerThreshold = 0.0
        self.atom.ng = Ng(args, self.atom.n.flatten())
        start = time.time()
        for it in range(Niter):
            Gamma.fill(0.0)
            Gamma += C
            gamma_matrices_escape_prob(&self.atom, bg.background, a.atmos)
            try:
                stat_eq_impl(&self.atom)
            except:
                raise ExplodingMatrixError('Singular Matrix')
            self.atom.ng.accelerate(self.atom.n.flatten())
            maxChange = self.atom.ng.max_change()
            delta = maxChange.dMax
            if delta < 3e-2:
                end = time.time()
                break
        else:
            print('Escape probability didn\'t converge for %s, setting LTE populations' % self.atomic_model.element.name)
            n = np.asarray(self.n)
            n[:] = np.asarray(self.n_star)

        if conserve_charge:
            deltaNe = np.sum((np.asarray(self.n) - prevN) * np.asarray(self.stages)[:, None], axis=0)

            for k in range(self.atmos.Nspace):
                self.atmos.ne[k] += deltaNe[k]

            for k in range(self.atmos.Nspace):
                if self.atmos.ne[k] < 1e6:
                    self.atmos.ne[k] = 1e6

    cpdef setup_wavelength(self, int la):
        '''
        Initialise the wavelength dependent arrays for the wavelength at
        index la.
        '''
        self.atom.setup_wavelength(la)

    def compute_profiles(self, polarised=False):
        '''
        Compute the line profiles for the spectral lines on the model.

        Parameters
        ----------
        polarised : bool, optional
            If True, and the lines are polarised, then the full Stokes line
            profiles will be computed, otherwise the scalar case will be
            computed (default: False). Lines that already have polarised
            profiles set up always have these recomputed, to keep them
            consistent with the scalar profile.
        '''
        np.asarray(self.v_broad)[:] = self.atomic_model.v_broad(self.atmos)
        cdef LwTransition t
        for t in self.trans:
            if polarised or t.trans.polarised:
                t.compute_polarised_profiles()
            else:
                t.compute_phi()

    @property
    def Nlevel(self):
        '''
        The number of levels in the atomic model.
        '''
        return self.atom.Nlevel

    @property
    def Ntrans(self):
        '''
        The number of transitions in the atomic model.
        '''
        return self.atom.Ntrans

    @property
    def v_broad(self):
        '''
        The broadening velocity associated with this atomic model in this
        atmosphere.
        '''
        return np.asarray(self.v_broad)

    @property
    def Gamma(self):
        '''
        The Gamma iteration matrix [Nlevel, Nlevel, Nspace].
        '''
        return np.asarray(self.Gamma)

    @property
    def C(self):
        '''
        The collisional rates matrix [Nlevel, Nlevel, Nspace]. This is filled
        s.t. C_{ji} is C[i, j] to facilitate addition to Gamma.
        '''
        return np.asarray(self.C)

    @property
    def n_total(self):
        '''
        The total number density of the model throughout the atmosphere.
        '''
        return np.asarray(self.n_total)

    @property
    def n(self):
        '''
        The atomic populations (NLTE if in use) [Nlevel, Nspace].
        '''
        return np.asarray(self.n)

    @property
    def n_star(self):
        '''
        The LTE populations for this species in this atmosphere [Nlevel, Nspace].
        '''
        return np.asarray(self.n_star)

    @property
    def stages(self):
        '''
        The ionisation stage of each level of this model.
        '''
        return np.asarray(self.stages)

    @property
    def trans(self):
        '''
        List of computational transitions (LwTransition).
        '''
        return self.trans

    @property
    def element(self):
        '''
        The element identifier for this atomic model.
        '''
        return self.atomic_model.element

    # Deprecated names (to be removed in a future release).
    vBroad = deprecated_alias('v_broad')
    nTotal = deprecated_alias('n_total')
    nStar = deprecated_alias('n_star')
    atomicModel = deprecated_alias('atomic_model')
    modelPops = deprecated_alias('model_pops')

cdef JRest_to_numpy(F64Arr2D& JRest):
    if JRest.data() is NULL:
        raise AttributeError
    cdef np.npy_intp shape[2]
    shape[0] = <np.npy_intp> JRest.shape(0)
    shape[1] = <np.npy_intp> JRest.shape(1)
    ndarray = np.PyArray_SimpleNewFromData(2, &shape[0],
                                            np.NPY_FLOAT64, <void*>JRest.data())
    return ndarray

cdef JRest_from_numpy(Spectrum& spect, f64[:,::1] JRest):
    spect.JRest = F64Arr2D(f64_view_2(JRest))

cdef class LwSpectrum:
    '''
    Storage and access to spectrum data used by backend. Instantiated by
    Context.

    Parameters
    ----------
    wavelength : np.ndarray
        The wavelength grid used in the simulation [nm].
    Nrays : int
        The number of rays in the angular quadrature.
    Nspace : int
        The number of points in the atmospheric model.
    Noutgoing : int
        The number of outgoing point in the atmosphere, essentially
        max(Ny*Nx, Nx, 1), (when used in an array these elements will be
        ordered as a flattened array of [Ny, Nx]).
    '''
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

    @accepts_old_kwargs
    def interp_J_from_state(self, prev_spect):
        cdef np.ndarray[np.double_t, ndim=2] J = np.asarray(self.J)
        cdef int k
        for k in range(self.J.shape[1]):
            J[:, k] = np.interp(self.wavelength, prev_spect.wavelength, prev_spect.J[:, k])

    @property
    def wavelength(self):
        '''
        Wavelength grid used [nm].
        '''
        return np.asarray(self.wavelength)

    @property
    def I(self):
        '''
        Intensity [J/s/m2/sr/Hz], shape is squeeze([Nlambda, Nmu, Noutgoing]).
        '''
        return np.squeeze(np.asarray(self.I))

    @property
    def J(self):
        '''
        Angle-averaged intensity [J/s/m2/sr/Hz], shape is squeeze([Nlambda, Nspace]).
        '''
        return np.squeeze(np.asarray(self.J))

    @property
    def Quv(self):
        '''
        Q, U and V Stokes parameters [J/s/m2/sr/Hz], shape is squeeze([3,
        Nlambda, Nmu, Noutgoing]).
        '''
        return np.squeeze(np.asarray(self.Quv))


cdef class LwContext:
    '''
    Context that configures and drives the backend. Whilst the class is named
    LwContext (to avoid cython collisions) it is exposed from the lightweaver
    package as Context.

    Attributes
    ----------
    kwargs : dict
        A dictionary of all inputs provided to the context, stored as the
        under the argument names to `__init__`.
    eq_pops : SpeciesStateTable
        The populations of each species in the simulation.
    conserve_charge : bool
        Whether charge is being conserved in the calculations.
    nr_h_only : bool
        Whether H is the only element included in charge conservation
    detailed_atom_prd : bool
        Whether PRD emission rho is computed for PRD lines with detailed static
        populations.
    crsw_callback : CrswIterator
        The object controlling the value of the Collisional Radiative
        Switching term.
    crsw_done : bool
        Indicates whether CRSW is done (i.e. the parameter has reached 1).

    Parameters
    ----------
    atmos : Atmosphere
        The atmospheric structure object.
    spect : SpectrumConfiguration
        The configuration of wavelength grids and active atoms/transitions.
    eq_pops : SpeciesStateTable
        The initial populations and storage for these populations during the
        simulation.
    ng_options : NgOptions, optional
        The parameters for Ng acceleration in the simulation (default: No
        acceleration).
    init_sol : InitialSolution, optional
        The starting solution for the population of all active species
        (default: LTE).
    conserve_charge : bool, optional
        Whether to conserve charge in the simulation (default: False).
    nr_h_only : bool, optional
        Only include hydrogen in charge conservation calculations (default: False).
    hprd : bool, optional
        Whether to use the Hybrid PRD method to account for velocity shifts in
        the atmosphere (if PRD is used otherwise, then it is angle-averaged).
    detailed_atom_prd: bool, optional
        Whether to compute the PRD emission coefficient rho for PRD lines on
        atoms with detailed static populations (default, True).
    crsw_callback : CrswIterator, optional
        An instance of CrswIterator (or derived thereof) to control
        collisional radiative swtiching (default: None for UnityCrswIterator
        i.e. no CRSW).
    Nthreads : int, optional
        Number of threads to use in the computation of the formal solution,
        default 1.
    background_provider : BackgroundProvider, optional
        Implementation for the background, if non-standard. Must follow the
        BackgroundProvider interface.
    formal_solver : str, optional
        Name of formal_solver registered with the FormalSolvers object.
    interp_fn : str, optional
        Name of interpolation function to use in the multi-dimensional formal
        solver. Must be registered with InterpFns.
    '''
    cdef Context ctx
    cdef LwAtmosphere atmos
    cdef LwSpectrum spect
    cdef LwBackground background
    cdef LwDepthData depth_data
    cdef public dict kwargs
    cdef public object eq_pops
    cdef list active_atoms
    cdef list detailed_atoms
    cdef public bool_t conserve_charge
    cdef public bool_t nr_h_only
    cdef public bool_t detailed_atom_prd
    cdef bool_t hprd
    cdef public object crsw_callback
    cdef public object crsw_done
    cdef dict __dict__

    def __init__(self, atmos, spect, eq_pops,
                 ng_options=None, init_sol=None,
                 conserve_charge=False,
                 nr_h_only=False,
                 detailed_atom_prd=True,
                 hprd=False,
                 crsw_callback=None, Nthreads=1,
                 background_provider=None,
                 formal_solver=None,
                 interp_fn=None,
                 fs_iter_scheme=None):
        self.kwargs = {
            'atmos': atmos,
            'spect': spect,
            'eq_pops': eq_pops,
            'ng_options': ng_options,
            'init_sol': init_sol,
            'conserve_charge': conserve_charge,
            'nr_h_only': nr_h_only,
            'detailed_atom_prd': detailed_atom_prd,
            'hprd': hprd,
            'Nthreads': Nthreads,
            'background_provider': background_provider,
            'formal_solver': formal_solver,
            'interp_fn': interp_fn,
            'fs_iter_scheme': fs_iter_scheme
        }
        cdef dict fs_iter_scheme_properties = self.get_fs_iter_scheme_properties(fs_iter_scheme)

        self.atmos = LwAtmosphere(atmos, spect.wavelength.shape[0])
        self.spect = LwSpectrum(spect.wavelength, atmos.Nrays,
                                atmos.Nspace, atmos.Noutgoing)
        self.conserve_charge = conserve_charge
        self.nr_h_only = nr_h_only
        self.hprd = hprd
        self.detailed_atom_prd = detailed_atom_prd

        self.background = LwBackground(self.atmos, eq_pops, spect.rad_set,
                                       spect.wavelength, provider=background_provider)
        self.eq_pops = eq_pops

        active_atoms = spect.rad_set.active_atoms
        detailed_atoms = spect.rad_set.detailed_atoms
        self.active_atoms = [LwAtom(a, self.atmos, eq_pops, spect,
                                   self.background, ng_options=ng_options,
                                   init_sol=init_sol,
                                   conserve_charge=conserve_charge,
                                   fs_iter_scheme_properties=fs_iter_scheme_properties)
                            for a in active_atoms]
        self.detailed_atoms = [LwAtom(a, self.atmos, eq_pops, spect,
                                     self.background, ng_options=None,
                                     init_sol=InitialSolution.Lte, detailed=True,
                                     fs_iter_scheme_properties=fs_iter_scheme_properties)
                              for a in detailed_atoms]

        self.ctx.atmos = &self.atmos.atmos
        self.ctx.spect = &self.spect.spect
        self.ctx.background = &self.background.background

        cdef LwAtom la
        for la in self.active_atoms:
            self.ctx.activeAtoms.push_back(&la.atom)
        for la in self.detailed_atoms:
            self.ctx.detailedAtoms.push_back(&la.atom)

        if self.hprd:
            self.configure_hprd_coeffs()

        if crsw_callback is None:
            self.crsw_callback = UnityCrswIterator()
            self.crsw_done = True
        else:
            self.crsw_callback = crsw_callback
            self.crsw_done = False

        shape = (self.spect.I.shape[0], self.atmos.Nrays, self.atmos.Nspace)
        self.depth_data = LwDepthData(*shape)
        self.ctx.depthData = &self.depth_data.depth_data

        self.set_formal_solver(formal_solver, in_constructor=True)
        self.set_interp_fn(interp_fn)
        self.set_fs_iter_scheme(fs_iter_scheme)
        self.setup_threads(Nthreads)

        self.compute_profiles()

    def __getstate__(self):
        state = {}
        state['kwargs'] = self.kwargs
        state['eq_pops'] = self.eq_pops
        state['active_atoms'] = self.active_atoms
        state['detailed_atoms'] = self.detailed_atoms
        state['conserve_charge'] = self.conserve_charge
        state['nr_h_only'] = self.nr_h_only
        state['detailed_atom_prd'] = self.detailed_atom_prd
        state['hprd'] = self.hprd
        state['atmos'] = self.atmos
        state['spect'] = self.spect
        state['background'] = self.background
        state['crsw_done'] = self.crsw_done
        if not self.crsw_done:
            state['crsw_callback'] = self.crsw_callback
        else:
            state['crsw_callback'] = None
        state['depth_data'] = self.depth_data
        return state

    def __setstate__(self, state):
        self.kwargs = state['kwargs']
        self.eq_pops = state['eq_pops']
        self.atmos = state['atmos']
        self.active_atoms = state['active_atoms']
        self.detailed_atoms = state['detailed_atoms']
        self.conserve_charge = state['conserve_charge']
        self.nr_h_only = state['nr_h_only']
        self.detailed_atom_prd = state['detailed_atom_prd']
        self.hprd = state['hprd']
        self.spect = state['spect']
        self.background = state['background']

        self.crsw_done = state['crsw_done']
        if state['crsw_callback'] is None:
            self.crsw_callback = UnityCrswIterator()
        else:
            self.crsw_callback = state['crsw_callback']

        self.ctx.atmos = &self.atmos.atmos
        self.ctx.spect = &self.spect.spect
        self.ctx.background = &self.background.background

        cdef LwAtom la
        for la in self.active_atoms:
            self.ctx.activeAtoms.push_back(&la.atom)
        for la in self.detailed_atoms:
            self.ctx.detailedAtoms.push_back(&la.atom)

        if self.hprd:
            self.configure_hprd_coeffs()

        shape = (self.spect.I.shape[0], self.atmos.Nrays, self.atmos.Nspace)
        self.depth_data = state['depth_data']
        self.ctx.depthData = &self.depth_data.depth_data
        self.set_formal_solver(self.kwargs['formal_solver'], in_constructor=True)
        self.set_interp_fn(self.kwargs['interp_fn'])
        self.set_fs_iter_scheme(self.kwargs['fs_iter_scheme'])

        self.setup_threads(self.kwargs['Nthreads'])

    @accepts_old_kwargs
    def set_formal_solver(self, formal_solver, in_constructor=False):
        '''
        For internal use. Set the formal solver through the constructor.
        '''
        cdef LwFormalSolverManager fsMan = FormalSolvers
        cdef int fsIdx
        if formal_solver is not None:
            fsIdx = fsMan.names.index(formal_solver)
        else:
            fsIdx = fsMan.default_formal_solver(self.ctx.atmos.Ndim)
        cdef FormalSolver fs = fsMan.manager.formalSolvers[fsIdx]
        # NOTE(cmo): The iteration cores solve a single wavelength per call;
        # wide formal solvers are not yet supported.
        if fs.width != 1:
            raise ValueError('Formal solver %s has width %d, but only width 1 is supported.'
                             % (fsMan.names[fsIdx], fs.width))
        self.ctx.formalSolver = fs

        # NOTE(cmo): If the FS is wide we may need to reconfigure the wide backing stores.
        # But we haven't initialised that system yet when calling in the constructor.
        if not in_constructor:
            self.update_threads()

    @accepts_old_kwargs
    def set_interp_fn(self, interp_fn):
        '''
        For internal use. Set the interpolation function through the
        constructor.
        '''
        cdef LwInterpFnManager interpMan = InterpFns
        cdef int interpIdx
        cdef InterpFn interp
        try:
            if interp_fn is not None:
                interpIdx = interpMan.names.index(interp_fn)
            else:
                interpIdx = interpMan.default_interp(self.ctx.atmos.Ndim)
            interp = interpMan.manager.fns[interpIdx]
            self.ctx.interpFn = interp
            return
        except ValueError as e:
            if self.ctx.atmos.Ndim > 1:
                raise e

    @accepts_old_kwargs
    def set_fs_iter_scheme(self, fs_iter_scheme):
        cdef LwFsIterationManager manager = FsIterationSchemes
        cdef int iterIdx
        cdef FsIterationFns iterFns

        if fs_iter_scheme is not None:
            iterIdx = manager.names.index(fs_iter_scheme)
        else:
            iterIdx = manager.default_scheme()
        iterFns = manager.manager.fns[iterIdx]
        self.ctx.iterFns = iterFns

    @accepts_old_kwargs
    def get_fs_iter_scheme_properties(self, fs_iter_scheme):
        cdef LwFsIterationManager manager = FsIterationSchemes
        cdef FsIterationFns iterFns
        cdef dict result

        if fs_iter_scheme is not None:
            result = manager.scheme_properties(fs_iter_scheme)
        else:
            result = manager.scheme_properties(manager.default_scheme_name())
        return result

    @property
    def Nthreads(self):
        '''
        The number of threads used by the formal solver. A new value can be
        assigned to this, and the necessary support structures will be
        automatically allocated.
        '''
        return self.ctx.Nthreads

    @Nthreads.setter
    def Nthreads(self, value):
        cdef int prevValue = self.ctx.Nthreads
        self.ctx.Nthreads = int(value)
        if prevValue != value:
            self.update_threads()

    @property
    def hprd(self):
        '''
        Whether PRD calculations are using the Hybrid PRD mode.
        '''
        return self.hprd

    cdef setup_threads(self, int Nthreads):
        '''
        Internal.
        '''
        self.ctx.Nthreads = Nthreads
        self.ctx.initialise_threads()

    cpdef update_threads(self):
        '''
        Internal.
        '''
        self.ctx.update_threads()

    cpdef compute_profiles(self, polarised=False):
        '''
        Compute the line profiles for the spectral lines on all active and
        detailed atoms.

        Parameters
        ----------
        polarised : bool, optional
            If True, and the lines are polarised, then the full Stokes line
            profiles will be computed, otherwise the scalar case will be
            computed (default: False).
        '''
        atoms = self.active_atoms + self.detailed_atoms
        for atom in atoms:
            atom.compute_profiles(polarised=polarised)

    cpdef formal_sol_gamma_matrices(self, fix_collisional_rates=False, lambda_iterate=False,
                                    extra_params=None):
        '''
        Compute the formal solution across all wavelengths and fill in the
        Gamma matrix for each active atom, allowing the populations to then
        be updated using the radiative information.

        Will use Nthreads for the formal solution.

        Parameters
        ----------
        fix_collisional_rates : bool, optional
            Whether to not recompute the collisional rates (default: False
            i.e. recompute them).
        lambda_iterate : bool, optional
            Whether to use Lambda iteration (setting the approximate Lambda
            term to zero), may be useful in certain unstable situations
            (default: False).
        extra_params : dict, optional
            Dict of extra parameters to be converted through the
            `dict2ExtraParams` function and passed onto the C++ core.

        Returns
        -------
        update: IterationUpdate
            An object representing the updates to the model. See
            `IterationUpdate` for details.
        '''
        if extra_params is None:
            extra_params = {}
        cdef ExtraParams params = dict2ExtraParams(extra_params)

        cdef LwAtom atom
        cdef np.ndarray[np.double_t, ndim=3] Gamma
        cdef f64 crswVal = self.crsw_callback()
        if crswVal == 1.0:
            self.crsw_done = True

        for atom in self.active_atoms:
            Gamma = np.asarray(atom.Gamma)
            Gamma.fill(0.0)
            if not fix_collisional_rates:
                atom.compute_collisions()
            Gamma += crswVal * np.asarray(atom.C)

        self.atmos.compute_bcs(self.spect)

        cdef IterationResult maxChange = formal_sol_gamma_matrices(self.ctx, lambda_iterate, params)
        update = IterationUpdate_from_IterationResult(self, maxChange)
        update.crsw = crswVal
        return update

    cpdef formal_sol(self, up_only=True, extra_params=None):
        '''
        Compute the formal solution across all wavelengths (used by
        `compute_rays`). Only computes upgoing rays by default, which has
        implication on boundary conditions in 2D.

        Parameters
        ----------
        up_only : bool, optional
            Only compute upgoing rays, (default: True)
        extra_params : dict, optional
            Dict of extra parameters to be converted through the
            `dict2ExtraParams` function and passed onto the C++ core.

        Returns
        -------
        update: IterationUpdate
            An object representing the updates to the model. See
            `IterationUpdate` for details.
        '''

        if extra_params is None:
            extra_params = {}
        cdef ExtraParams params = dict2ExtraParams(extra_params)

        self.atmos.compute_bcs(self.spect)

        cdef IterationResult maxChange = formal_sol(self.ctx, up_only, params)
        update = IterationUpdate_from_IterationResult(self, maxChange)
        return update


    cpdef update_deps(self, temperature=True, ne=True, vturb=True,
                      vlos=True, B=True, background=True, hprd=True,
                      quiet=True):
        '''
        Update various dependent parameters in the simulation after changes
        to different components. If a component has not been adjust then its
        associated argument can be set to False. By default, all standard
        dependent components are recomputed (e.g. projected velocities, line
        profiles, LTE populations, background terms).

        Parameters
        ----------
        temperature : bool, optional
            Whether the temperature has been modified.
        ne : bool, optional
            Whether the electron density has been modified (this affects the
            line damping, and hence the profiles).
        vturb : bool, optional
            Whether the microturbulent velocity has been modified.
        vlos : bool, optional
            Whether the bulk velocity field has been modified.
        B : bool, optional
            Whether the magnetic field has been modified. Polarised profiles
            are recomputed for lines on which they have been set up.
        background : bool, optional
            Whether the background needs updating.
        hprd : bool, optional
            Whether the hybrid PRD terms need updating. These only depend on
            the projected velocity, so are only recomputed if `vlos` is also
            True.
        quiet : bool, optional
            Whether to print any update information from these functions
            (default: True).
        '''
        if vlos or B:
            self.atmos.update_projections()

        if temperature or ne:
            self.eq_pops.update_lte_atoms_hmin_pops(self.kwargs['atmos'], conserve_charge=self.conserve_charge,
                                                   update_totals=True, quiet=quiet)

        # NOTE(cmo): Profiles must follow the LTE update, as the damping
        # depends on the perturber populations.
        if any([temperature, ne, vturb, vlos, B]):
            self.compute_profiles()
            if temperature or ne or vturb:
                # NOTE(cmo): The PRD gII depends on v_broad and a_damp, flag it
                # for lazy recomputation.
                for atom in self.active_atoms + self.detailed_atoms:
                    for t in atom.trans:
                        t.recompute_gII()

        if background and any([temperature, ne, vturb, vlos]):
            self.background.update_background(self.atmos)

        # NOTE(cmo): The H-PRD coefficients only depend on the projected velocity.
        if self.hprd and hprd and vlos:
            self.update_hprd_coeffs()

    cpdef rel_diff_pops(self):
        '''
        Internal.
        '''
        cdef LwAtom atom
        cdef Atom* a
        cdef NgChange maxChange
        cdef f64 delta
        cdef f64 maxDelta = 0.0
        cdef int i
        atoms = self.active_atoms

        update = IterationUpdate(self, updated_pops=True)

        for i, atom in enumerate(atoms):
            a = &atom.atom
            maxChange = a.ng.relative_change_from_prev(a.n.flatten())
            delta = maxChange.dMax
            maxDelta = max(maxDelta, delta)
            update.dpops.append(maxChange.dMax)
            update.dpops_max_idx.append(maxChange.dMaxIdx)

        return update

    cpdef rel_diff_ng_accelerate(self):
        '''
        Internal.
        '''
        cdef LwAtom atom
        cdef Atom* a
        cdef NgChange maxChange
        cdef f64 delta
        cdef f64 maxDelta = 0.0
        cdef int i
        atoms = self.active_atoms

        update = IterationUpdate(self, updated_pops=True)

        for i, atom in enumerate(atoms):
            a = &atom.atom
            accelerated = a.ng.accelerate(a.n.flatten())
            maxChange = a.ng.max_change()
            delta = maxChange.dMax
            maxDelta = max(maxDelta, delta)
            update.dpops.append(maxChange.dMax)
            update.dpops_max_idx.append(maxChange.dMaxIdx)
            update.ng_accelerated.append(accelerated)

        return update

    cpdef time_dep_update(self, f64 dt, prev_time_pops=None, ng_update=None,
                          int chunk_size=20, extra_params=None):
        '''
        Update the populations of active atoms using the current values of
        their Gamma matrices. This function solves the time-dependent kinetic
        equilibrium equations (ignoring advective terms). Currently uses a
        fully implicit (theta = 0) integrator.

        Parameters
        ----------
        dt : float
            The timestep length [s].
        prev_time_pops : list of np.ndarray or None
            The NLTE populations for each active atom at the start of the
            timestep (order matching that of Context.active_atoms). This does
            not need to be provided the first time time_dep_update is called
            for a timestep, as if this parameter is None then this list will
            be constructed, and returned as the second return value, and can
            then be passed in again for additional iterations on a timestep.
        ng_update : bool, optional
            Whether to apply Ng Acceleration (default: None, to apply automatic
            behaviour), will only accelerate if the counter on the Ng accelerator
            has seen enough steps since the previous acceleration (set in Context
            initialisation).
        chunk_size : int, optional
            Not currently used.
        extra_params : dict, optional
            Dict of extra parameters to be converted through the
            `dict2ExtraParams` function and passed onto the C++ core.

        Returns
        -------
        update: IterationUpdate
            An object representing the updates to the model. See
            `IterationUpdate` for details.
        prev_time_pops : list of np.ndarray
            The input needed as `prev_time_pops` if this function is to be called
            again for this timestep.
        '''
        atoms = self.active_atoms

        if ng_update is None:
            if self.conserve_charge:
                ng_update = False
            else:
                ng_update = True

        if extra_params is None:
            extra_params = {}
        cdef ExtraParams params = dict2ExtraParams(extra_params)

        cdef LwAtom atom
        cdef Atom* a
        cdef f64 delta
        cdef f64 maxDelta = 0.0
        cdef bool_t accelerated
        cdef vector[F64View2D] prevTimePopsVec

        if prev_time_pops is None:
            prev_time_pops = [np.copy(atom.n) for atom in atoms]

        for atom in atoms:
            a = &atom.atom
            if not a.ng.init:
                a.ng.accelerate(a.n.flatten())

        try:
            for i, atom in enumerate(atoms):
                a = &atom.atom
                time_dependent_update(self.ctx, a, f64_view_2(prev_time_pops[i]), dt, params)
        except:
            raise ExplodingMatrixError('Singular Matrix')

        if ng_update:
            update = self.rel_diff_ng_accelerate()
        else:
            update = self.rel_diff_pops()

        return update, prev_time_pops

    cpdef time_dep_restore_prev_pops(self, prev_time_pops):
        '''
        Restore the populations to their state prior to the time-dependent
        updates for this timestep. Also resets I and J to 0. May be useful in
        cases where a problem was encountered.

        Parameters
        ----------
        prev_time_pops : list of np.ndarray
            `prev_time_pops` returned by time_dep_update.
        '''
        cdef LwAtom atom
        cdef int i
        for i, atom in enumerate(self.active_atoms):
            np.asarray(atom.n)[:] = prev_time_pops[i]

        np.asarray(self.spect.I).fill(0.0)
        np.asarray(self.spect.J).fill(0.0)

    cpdef clear_ng(self):
        '''
        Resets Ng acceleration objects on all active atoms.
        '''
        cdef LwAtom atom
        for atom in self.active_atoms:
            atom.atom.ng.clear()

    cpdef stat_equil(self, int chunk_size=20, extra_params=None):
        '''
        Update the populations of active atoms using the current values of
        their Gamma matrices. This function solves the time-independent statistical
        equilibrium equations.

        Parameters
        ----------
        chunk_size : int, optional
            Not currently used.
        extra_params : dict, optional
            Dict of extra parameters to be converted through the
            `dict2ExtraParams` function and passed onto the C++ core.

        Returns
        -------
        update: IterationUpdate
            An object representing the updates to the model. See
            `IterationUpdate` for details.
        '''
        atoms = self.active_atoms

        cdef LwAtom atom
        cdef Atom* a
        cdef f64 delta
        cdef f64 maxDelta = 0.0
        cdef bool_t accelerated
        cdef LwTransition t
        cdef int k
        cdef np.ndarray[np.double_t, ndim=1] deltaNe

        if extra_params is None:
            extra_params = {}
        cdef ExtraParams params = dict2ExtraParams(extra_params)

        for atom in atoms:
            a = &atom.atom
            if not a.ng.init:
                a.ng.accelerate(a.n.flatten())

        try:
            for atom in atoms:
                a = &atom.atom
                stat_eq(self.ctx, a, params)
        except:
            raise ExplodingMatrixError('Singular Matrix')

        if self.conserve_charge:
            neStart = np.copy(self.atmos.ne)
            self.nr_post_update(ng_update=False, h_only=self.nr_h_only)

        update = self.rel_diff_ng_accelerate()
        if self.conserve_charge:
            neDiff = np.abs((np.asarray(self.atmos.ne) - neStart)
                            / np.asarray(self.atmos.ne))
            neDiffMaxIdx = neDiff.argmax()
            neDiffMax = neDiff[neDiffMaxIdx]
            maxDelta = max(maxDelta, neDiffMax)
            update.updated_ne = True
            update.dne_max = neDiffMax
            update.dne_max_idx = neDiffMaxIdx

        return update

    def _nr_post_update_impl(self, atoms, dC, f64[::1] backgroundNe,
                             time_dependent_data=None, int chunk_size=5, extra_params=None):
        crswVal = self.crsw_callback.val
        cdef f64 crsw = crswVal
        cdef vector[Atom*] atomVec
        cdef vector[F64View3D] dCVec
        cdef Atom* a
        cdef LwAtom atom
        cdef NrTimeDependentData td
        cdef int i

        if extra_params is None:
            extra_params = {}
        cdef ExtraParams params = dict2ExtraParams(extra_params)

        if time_dependent_data is not None:
            td.dt = time_dependent_data['dt']
            td.nPrev.reserve(len(time_dependent_data['n_prev']))
            for i in range(len(time_dependent_data['n_prev'])):
                td.nPrev.push_back(f64_view_2(time_dependent_data['n_prev'][i]))

        atomVec.reserve(len(atoms))
        for atom in atoms:
            atomVec.push_back(&atom.atom)
        dCVec.reserve(len(dC))
        for c in dC:
            dCVec.push_back(f64_view_3(c))

        try:
            nr_post_update(self.ctx, &atomVec, dCVec, f64_view(backgroundNe), td, crsw, params);
        except:
            raise ExplodingMatrixError('Singular Matrix')

    cpdef update_projections(self):
        '''
        Update all arrays of projected terms in the atmospheric model.
        '''
        self.atmos.update_projections()

    cpdef setup_stokes(self, recompute=False):
        '''
        Configure the Context for Full Stokes radiative transfer.

        Parameters
        ----------
        recompute : bool, optional
            If previously called, and called again with `recompute = True`
            the line profiles will be recomputed.
        '''
        try:
            if self.atmos.B.shape[0] == 0:
                raise ValueError('Please specify B-field')
        except:
            raise ValueError('Please specify B-field')

        cdef LwAtom atom
        cdef LwTransition t
        for atom in self.active_atoms + self.detailed_atoms:
            for t in atom.trans:
                if recompute or not t.trans.polarised:
                    t.compute_polarised_profiles()

        self.spect.setup_stokes()

    cpdef single_stokes_fs(self, recompute=False, update_J=False, up_only=True,
                           extra_params=None):
        '''
        Compute a full Stokes formal solution across all wakelengths in the
        grid, setting up the Context first (it is rarely necessary to call
        setup_stokes directly).

        The full Stokes formal solution is not currently multi-threaded, as
        it is usually only called once at the end of a simulation, however
        this could easily be changed.

        Parameters
        ----------
        recompute : bool, optional
            If previously called, and called again with `recompute = True`
            the line profiles will be recomputed. (Default: False)
        update_J : bool, optional
            Whether to update J on the Context during the calculation (Default: False)
        up_only : bool, optional
            Whether to compute the formal solver only for upgoing rays (used in
            final synthesis).
        extra_params : dict, optional
            Dict of extra parameters to be converted through the
            `dict2ExtraParams` function and passed onto the C++ core.

        Returns
        -------
        update: IterationUpdate
            An object representing the updates to the model. See
            `IterationUpdate` for details.
        '''
        self.setup_stokes(recompute=recompute)
        if extra_params is None:
            extra_params = {}
        cdef ExtraParams params = dict2ExtraParams(extra_params)

        self.atmos.compute_bcs(self.spect)
        cdef IterationResult maxChange = formal_sol_full_stokes(self.ctx, update_J,
                                                                up_only, params)
        update = IterationUpdate_from_IterationResult(self, maxChange)
        return update

    cpdef prd_redistribute(self, int max_iter=3, f64 tol=1e-2, extra_params=None):
        '''
        Update emission profile ratio rho by computing the scattering integral
        for each prd line. Does not affect the populations, interleave before
        each formal solution for a standard problem.

        Parameters
        ----------
        max_iter : int, optional
            The maximum number of iterations of updating rho to be taken (Default: 3).
        tol : float, optional
            The default stopping tolerance for relative changes in rho. If the
            relative change in rho falls below this threshold then this function
            returns i.e. `max_iter` iterations do not need to be taken (Default: 1e-2).
        extra_params : dict, optional
            Dict of extra parameters to be converted through the
            `dict2ExtraParams` function and passed onto the C++ core.

        Returns
        -------
        update: IterationUpdate
            An object representing the updates to the model. See
            `IterationUpdate` for details.
        '''
        if extra_params is None:
            extra_params = {"include_detailed_atoms": self.detailed_atom_prd}
        cdef ExtraParams params = dict2ExtraParams(extra_params)

        cdef IterationResult prdIter = redistribute_prd_lines(self.ctx, max_iter, tol, params)
        update = IterationUpdate_from_IterationResult(self, prdIter)
        return update

    cdef configure_hprd_coeffs(self):
        '''
        Internal.
        '''
        configure_hprd_coeffs(self.ctx, self.detailed_atom_prd)

    cpdef update_hprd_coeffs(self):
        '''
        Update the values of the H-PRD coefficients, this needs to be called
        if changes are made to the atmospheric structure to ensure that the
        interpolation parameters are correct.
        '''
        self.configure_hprd_coeffs()
        # NOTE(cmo): configure_hprd_coeffs throws away all of the interpolation
        # stuff stored on each line, and allocates a new block for it, this
        # means that the Transitions sitting in the threading factories are now
        # pointing to stale data, so we regenerate the entire threading
        # context. This is a bit wasteful, but at 8 threads it takes < 3 ms, vs
        # 500 ms+ for the coeffs on an average CaII + MgII case.
        self.update_threads()

    @property
    def active_atoms(self):
        '''
        All active computational atomic models (LwAtom).
        '''
        return self.active_atoms

    @property
    def detailed_atoms(self):
        '''
        All detailed static computational atomic models (LwAtom).
        '''
        return self.detailed_atoms

    @property
    def spect(self):
        '''
        The spectrum storage object (LwSpectrum).
        '''
        return self.spect

    @property
    def atmos(self):
        '''
        The atmospheric model storage object (LwAtmosphere).
        '''
        return self.atmos

    @property
    def background(self):
        '''
        The background storage object (LwBackground).
        '''
        return self.background

    @property
    def depth_data(self):
        '''
        Configuration and storage for full depth-dependent data of large
        parameters (LwDepthData).
        '''
        return self.depth_data

    def state_dict(self):
        '''
        Return the state dictionary for the Context, which can be used to
        serialise the entire Context and/or reconstruct it.
        '''
        return self.__getstate__()

    @staticmethod
    def construct_from_state_dict_with(
        sd,
        atmos=None,
        spect=None,
        eq_pops=None,
        ng_options=None,
        init_sol=None,
        conserve_charge=None,
        nr_h_only=None,
        detailed_atom_prd=None,
        hprd=None,
        preserve_profiles=False,
        fromScratch=False,
        background_provider=None
    ):
        """
        Construct a new Context informed by a state dictionary with changes
        provided to this function. This function is primarily aimed at making
        similar versions of a Context, as this can be duplicated much more
        easily by `deepcopy` or `pickle.loads(pickle.dumps(ctx))`. For
        example, wanting to replace the SpectrumConfiguration to run a
        different set of active atoms in the same atmospheric model.
        N.B. stateDict will not be deepcopied by this function -- do that
        yourself if needed.

        Parameters
        ----------
        sd : dict
            The state dictionary, from Context.state_dict.
        atmos : Atmosphere, optional
            Atmospheric model to use instead of the one present in stateDict.
        spect : SpectrumConfiguration, optional
            Spectral configuration to use instead of the one present in
            stateDict.
        eq_pops : SpeciesStateTable, optional
            Species population object to use instead of the one present in
            stateDict.
        ng_options : NgOptions, optional
            Ng acceleration options to use.
        init_sol : InitialSolution, optional
            Initial solution to use, only matters if `fromScratch` is True.
        conserve_charge : bool, optional
            Whether to conserve charge.
        nr_h_only : bool, optional
            Whether to only consider Hydrogen in charge conservation calculations.
        detailed_atom_prd : bool, optional
            Whether to compute the PRD emission coefficient rho for PRD lines on
            detailed atoms.
        hprd : bool, optional
            Whether to use Hybrid-PRD.
        preserve_profiles : bool, optional
            Whether to copy the current line profiles, or compute new ones
            (default: recompute).
        fromScratch : bool, optional
            Whether to construct the new Context, but not make any
            modifications, such as copying profiles and rates.
        background_provider : BackgroundProvider, optional
            The background package to use instead of the one present in
            stateDict.

        Returns
        -------
        ctx : Context
            The new context object for the simulation.
        """
        sd = copy(sd)
        sd['kwargs'] = remap_old_keys(sd['kwargs'], stacklevel=3)
        args = sd['kwargs']
        wavelengthSubset = False

        if ng_options is not None:
            args['ng_options'] = ng_options
        if init_sol is not None:
            args['init_sol'] = init_sol
        if conserve_charge is not None:
            args['conserve_charge'] = conserve_charge
        if nr_h_only is not None:
            args['nr_h_only'] = nr_h_only
        if detailed_atom_prd is not None:
            args['detailed_atom_prd'] = detailed_atom_prd
        if hprd is not None:
            args['hprd'] = hprd
        if background_provider is not None:
            args['background_provider'] = background_provider

        if atmos is not None:
            args['atmos'] = atmos
            if not eq_pops:
                # TODO(cmo); This should also probably recompute ICE
                args['eq_pops'] = copy(args['eq_pops'])
                args['eq_pops'].atmos = atmos
                args['eq_pops'].update_lte_atoms_hmin_pops(args['atmos'], conserve_charge=args['conserve_charge'])
        if eq_pops is not None:
            args['eq_pops'] = eq_pops
        if spect is not None:
            prev_spect = args['spect']
            args['spect'] = spect
            wavelengthSubset = spect.wavelength[0] >= prev_spect.wavelength[0] and spect.wavelength[-1] <= prev_spect.wavelength[-1]
        if not fromScratch:
            prevInitSol = args['init_sol']
            args['init_sol'] = InitialSolution.Lte

        # NOTE: Construct the public Context subclass (with nr_post_update and the
        # deprecated keyword handling), imported here as lightweaver/__init__.py
        # imports this module before defining it.
        from lightweaver import Context
        ctx = Context(**args)

        if fromScratch:
            return ctx

        if wavelengthSubset:
            ctx.spect.interp_J_from_state(sd['spect'])

        # TODO(cmo): I don't really like the way we use __getstate__ here, for
        # pickling the new approach is better. Performance implact is probably
        # negligble...
        cdef LwAtom a
        for a in ctx.active_atoms:
            for s in sd['active_atoms']:
                if a.atomic_model.element == s.atomic_model.element:
                    levels = a.atomic_model.levels == s.atomic_model.levels
                    if not levels:
                        break
                    trans = a.atomic_model.lines == s.atomic_model.lines
                    trans = trans and a.atomic_model.continua == s.atomic_model.continua
                    pops_only = False
                    if not trans:
                        pops_only = True
                    a.load_pops_rates_prd_from_state(s.__getstate__(), pops_only=pops_only, preserve_profiles=preserve_profiles)
                    break
            else:
                if prevInitSol == InitialSolution.EscapeProbability:
                    a.set_pops_escape_probability(ctx.atmos, ctx.background, conserve_charge=ctx.conserve_charge)


        for a in ctx.detailed_atoms:
            for s in sd['detailed_atoms']:
                if a.atomic_model.element == s.atomic_model.element:
                    a.load_pops_rates_prd_from_state(s.__getstate__())
                    break

        return ctx

    @accepts_old_kwargs
    def compute_rays(self, wavelengths=None, mus=None, stokes=False,
                     update_bcs=None, up_only=True, return_ctx=False,
                     refine_prd=False, squeeze=True):
        '''
        Compute the formal solution through a converged simulation for a
        particular ray (or set of rays). The wavelength range can be adjusted
        to focus on particular lines.

        Parameters
        ----------
        wavelengths : np.ndarray, optional
            The wavelengths at which to compute the solution (default: None,
            i.e. the original grid).
        mus : float or sequence of float or dict
            The cosines of the angles between the rays and the z-axis to use,
            if a float or sequence of float then these are taken as muz. If a
            dict, then it is expected to be dictionary unpackable
            (double-splat) into atmos.rays, and can then be used for
            multi-dimensional atmospheres.
        stokes : bool, optional
            Whether to compute a full Stokes solution (default: False).
        update_bcs : Callable[[Atmosphere], None]
            Function to be applied to the Atmosphere (intended to update the
            boundary conditions if needed) before constructing the new
            Context for these rays. If a ray doesn't intersect the boundary
            (i.e. x and y boundaries for muz == 1), then the boundary
            condition can be ignored.
        up_only : bool, optional
            Whether to only compute upgoing rays. Mostly affects the handling of
            boundary conditions for 2D atmospheres. (Default: True).
        return_ctx : bool, optional
            Whether to return the Context used to compute the formal solution
            for these rays. If true, it will be returned as the second value.
            Default: False.
        refine_prd : bool, optional
            Whether to update the rho_prd term by reevaluating the scattering
            integral on the new wavelength grid. This can sometimes visually
            improve the final solution, but is quite computationally costly.
            (default: False i.e. not reevaluated, instead if the wavelength
            grid is different, rho_prd is interpolated onto the new grid).
        squeeze : bool, optional
            Whether to squeeze singular dimensions from the output array
            (default: True).

        Returns
        -------
        intensity : np.ndarray
            The outgoing intensity for the chosen rays. If `stokes=True` then
            the first dimension indicates, in order, the I, Q, U, V
            components.
        '''
        state = deepcopy(self.state_dict())
        if wavelengths is not None:
            spect = state['kwargs']['spect'].subset_configuration(wavelengths)
        else:
            # NOTE(cmo): Subsets handle overlaps differently (which prevents
            # jumps at the edge of grids), so require that here even if the
            # wavelength grid is the same
            spect = state['kwargs']['spect'].subset_configuration(
                state['kwargs']['spect'].wavelength
            )

        cdef LwContext rhoCtx, rayCtx
        if refine_prd:
            rhoCtx = self.construct_from_state_dict_with(state, spect=spect)
            rhoCtx.prd_redistribute(max_iter=100)
            sd = rhoCtx.state_dict()
            atmos = sd['kwargs']['atmos']
            if mus is not None:
                if isinstance(mus, dict):
                    atmos.rays(**mus, up_only=up_only)
                else:
                    atmos.rays(mus, up_only=up_only)
            if update_bcs is not None:
                update_bcs(atmos)
            rayCtx = self.construct_from_state_dict_with(sd)
        else:
            atmos = state['kwargs']['atmos']
            if mus is not None:
                if isinstance(mus, dict):
                    atmos.rays(**mus, up_only=up_only)
                else:
                    atmos.rays(mus, up_only=up_only)
            if update_bcs is not None:
                update_bcs(atmos)
            rayCtx = self.construct_from_state_dict_with(state, spect=spect)

        if stokes:
            rayCtx.single_stokes_fs(up_only=up_only)
            Iwav = np.asarray(rayCtx.spect.I)
            quv = np.asarray(rayCtx.spect.Quv)
            if squeeze:
                Iwav = np.squeeze(Iwav)
                quv = np.squeeze(quv)
            Iquv = np.zeros((4, *Iwav.shape))
            Iquv[0, :] = Iwav
            Iquv[1:, :] = quv
            if return_ctx:
                return Iquv, rayCtx
            else:
                return Iquv
        else:
            rayCtx.formal_sol(up_only=up_only)
            Iwav = np.asarray(rayCtx.spect.I)
            if squeeze:
                Iwav = np.squeeze(Iwav)
            if return_ctx:
                return Iwav, rayCtx
            else:
                return Iwav

    # Deprecated names (to be removed in a future release).
    activeAtoms = deprecated_alias('active_atoms')
    detailedAtoms = deprecated_alias('detailed_atoms')
    depthData = deprecated_alias('depth_data')
    eqPops = deprecated_alias('eq_pops')
    conserveCharge = deprecated_alias('conserve_charge')
    nrHOnly = deprecated_alias('nr_h_only')
    detailedAtomPrd = deprecated_alias('detailed_atom_prd')
    crswCallback = deprecated_alias('crsw_callback')
    crswDone = deprecated_alias('crsw_done')


cdef class LwFormalSolverManager:
    '''
    Storage and enumeration of the different formal solvers loaded for use in
    Lightweaver. There is no need to instantiate this class directly, instead
    there is a single instance of it instantiated as `FormalSolvers`, which
    should be used.

    Attributes
    ----------
    paths : list of str
        The currently loaded paths.
    names : list of str
        The names of all available formal solvers, each of which can be
        passed to the Context constructor.
    '''
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
        '''
        Attempt to load a formal solver, following the Lightweaver API, from
        a shared library at `path`.

        Parameters
        ----------
        path : str
            The path from which to load the formal solver.
        '''
        if path in self.paths:
            raise ValueError('Tried to load a pre-existing path')

        self.paths.append(path)
        byteStore = path.encode('UTF-8')
        cdef const char* cPath = byteStore
        cdef bool_t success = self.manager.load_fs_from_path(cPath)
        if not success:
            raise ValueError('Failed to load Formal Solver from library at %s' % path)

        cdef const char* name = self.manager.formalSolvers.at(self.manager.formalSolvers.size()-1).name
        self.names.append(name.decode('UTF-8'))

    def default_formal_solver(self, Ndim):
        '''
        Returns the name of the default formal solver for a given dimensionality.

        Parameters
        ----------
        Ndim : int
            The dimensionality of the simulation.

        Returns
        -------
        name : str
            The name of the default formal solver.
        '''
        if Ndim == 1:
            return self.names.index(lwConfig.params['FormalSolver1d'])
        elif Ndim == 2:
            return self.names.index(lwConfig.params['FormalSolver2d'])
        else:
            raise ValueError()

cdef class LwInterpFnManager:
    '''
    Storage and enumeration of the different interpolation functions for
    multi-dimensional formal solvers loaded for use in Lightweaver. There is
    no need to instantiate this class directly, instead there is a single
    instance of it instantiated as `InterpFns`, which should be used.

    Attributes
    ----------
    paths : list of str
        The currently loaded paths.
    names : list of str
        The names of all available interpolation functions, each of which can
        be passed to the Context constructor.
    '''
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

    def load_interp_fn_from_path(self, str path):
        '''
        Attempt to load an interpolation function, following the Lightweaver
        API, from a shared library at `path`.

        Parameters
        ----------
        path : str
            The path from which to load the interpolation function.
        '''
        if path in self.paths:
            raise ValueError('Tried to load a pre-existing path')

        self.paths.append(path)
        byteStore = path.encode('UTF-8')
        cdef const char* cPath = byteStore
        cdef bool_t success = self.manager.load_fn_from_path(cPath)
        if not success:
            raise ValueError('Failed to load interpolation function from library at %s' % path)

        cdef const char* name = self.manager.fns.at(self.manager.fns.size()-1).name
        self.names.append(name.decode('UTF-8'))

    def default_interp(self, Ndim):
        '''
        Returns the name of the default interpolation function for a given
        dimensionality.

        Parameters
        ----------
        Ndim : int
            The dimensionality of the simulation.

        Returns
        -------
        name : str
            The name of the default interpolation function.
        '''
        if Ndim == 2:
            return self.names.index('interp_linear_2d')
        else:
            raise ValueError("Unexpected Ndim")

cdef class LwFsIterationManager:
    cdef FsIterationFnsManager manager
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

        schemes = get_fs_iter_libs()
        for s in schemes:
            self.load_fns_from_path(s)

    def load_fns_from_path(self, str path):
        if path in self.paths:
            raise ValueError('Tried to load a pre-existing path')

        self.paths.append(path)
        byteStore = path.encode('UTF-8')
        cdef const char* cPath = byteStore
        cdef bool_t success = self.manager.load_fns_from_path(cPath)
        if not success:
            raise ValueError('Failed to load iteration scheme from library at %s' % path)

        cdef const char* name = self.manager.fns.at(self.manager.fns.size()-1).name
        self.names.append(name.decode('UTF-8'))

    def scheme_properties(self, str name):
        cdef int idx = self.names.index(name)
        cdef FsIterationFns scheme = self.manager.fns.at(idx)
        return {'name': name,
                'Ndim': scheme.Ndim,
                'dimensionSpecific': scheme.dimensionSpecific,
                'respectsFormalSolver': scheme.respectsFormalSolver,
                'defaultPerAtomStorage': scheme.defaultPerAtomStorage,
                'defaultWlaGijStorage': scheme.defaultWlaGijStorage}

    def default_scheme(self):
        try:
            return self.names.index('{IterationScheme}_{SimdImpl}'.format(**lwConfig.params))
        except AttributeError:
            return self.names.index(lwConfig.params['{IterationScheme}'.format(**lwConfig.params)])

    def default_scheme_name(self):
        return self.names[self.default_scheme()]

cdef fvec2list(const vector[f64]& v):
    cdef int i
    result = []
    for i in range(v.size()):
        result.append(v[i])
    return result

cdef ivec2list(const vector[int]& v):
    cdef int i
    result = []
    for i in range(v.size()):
        result.append(v[i])
    return result

cdef IterationUpdate_from_IterationResult(LwContext ctx, IterationResult result):
    update = IterationUpdate(ctx, updated_J=result.updatedJ,
                                  dJ_max=result.dJMax,
                                  dJ_max_idx=result.dJMaxIdx,
                                  updated_pops=result.updatedPops,
                                  dpops=fvec2list(result.dPops),
                                  dpops_max_idx=ivec2list(result.dPopsMaxIdx),
                                  ng_accelerated=result.ngAccelerated,
                                  updated_ne=result.updatedNe,
                                  dne_max=result.dNe,
                                  dne_max_idx=result.dNeMaxIdx,
                                  updated_rho=result.updatedRho,
                                  Nprd_sub_iter=result.NprdSubIter,
                                  drho=fvec2list(result.dRho),
                                  drho_max_idx=ivec2list(result.dRhoMaxIdx),
                                  updated_J_prd=result.updatedJPrd,
                                  dJ_prd_max=fvec2list(result.dJPrdMax),
                                  dJ_prd_max_idx=ivec2list(result.dJPrdMaxIdx))
    return update

FormalSolvers = LwFormalSolverManager()
InterpFns = LwInterpFnManager()
FsIterationSchemes = LwFsIterationManager()