import numpy as np
cimport numpy as np
from CmoArray cimport *
from CmoArrayHelper cimport *
from libcpp cimport bool as bool_t
from libcpp.vector cimport vector
from libc.math cimport sqrt, exp
from Atmosphere import BoundaryCondition
from AtomicModel import AtomicLine, LineType
from scipy.interpolate import interp1d
import Constants as Const
import time
from enum import Enum, auto

ctypedef np.int8_t i8
ctypedef Array1NonOwn[np.int32_t] I32View
ctypedef Array1NonOwn[bool_t] BoolView

class InitialSolution(Enum):
    Lte = auto()
    Zero = auto()
    EscapeProbability = auto()


cdef extern from "Atmosphere.hpp":
    cdef enum RadiationBC:
        ZERO
        THERMALISED

    cdef cppclass Atmosphere:
        F64View cmass
        F64View height
        F64View tau_ref
        F64View temperature
        F64View ne
        F64View vlos
        F64View2D vlosMu
        F64View B
        F64View gammaB
        F64View chiB
        F64View2D cosGamma
        F64View2D cos2chi
        F64View2D sin2chi
        F64View vturb
        F64View nHtot
        F64View muz
        F64View muy
        F64View mux
        F64View wmu
        int Nspace
        int Nrays

        RadiationBC lowerBc
        RadiationBC upperBc

        void update_projections()

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

        Atmosphere* atmos
    cdef void basic_background(BackgroundData* bg)
    cdef f64 Gaunt_bf(f64, f64, int)

cdef extern from "Ng.hpp":
    cdef cppclass Ng:
        bool_t init
        Ng()
        Ng(int nOrder, int nPeriod, int nDelay, F64View sol)
        bool_t accelerate(F64View sol)
        f64 max_change()

cdef extern from "Formal.hpp":
    cdef cppclass Background:
        F64View2D chi
        F64View2D eta
        F64View2D sca

    cdef cppclass Spectrum:
        F64View wavelength
        F64View2D I
        F64View3D Quv
        F64View2D J

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

        void uv(int la, int mu, bool_t toObs, F64View Uji, F64View Vij, F64View Vji)
        void compute_phi(const Atmosphere& atmos, F64View aDamp, F64View vBroad)
        void compute_polarised_profiles(const Atmosphere& atmos, F64View aDamp, F64View vBroad, const ZeemanComponents& z)

    cdef cppclass Atom:
        Atmosphere* atmos
        F64View2D n
        F64View2D nStar
        F64View vBroad
        F64View nTotal
        F64View3D Gamma
        F64View3D C

        F64View eta
        F64View2D gij
        F64View2D wla
        F64View2D V
        F64View2D U
        F64View2D chi

        vector[Transition*] trans
        Ng ng

        int Nlevel
        int Ntrans

    cdef cppclass Context:
        Atmosphere* atmos
        Spectrum* spect
        vector[Atom*] activeAtoms
        Background* background

    cdef f64 gamma_matrices_formal_sol(Context& ctx)
    cdef f64 formal_sol_full_stokes(Context& ctx)
    cdef f64 redistribute_prd_lines(Context& ctx, int maxIter, f64 tol)
    cdef void stat_eq(Atom* atom)
    cdef void configure_hprd_coeffs(Context& ctx)

cdef extern from "Formal.hpp" namespace "EscapeProbability":
    cdef void gamma_matrices_escape_prob(Atom* a, Background& background, const Atmosphere& atmos)


cdef class LwAtmosphere:
    cdef Atmosphere atmos
    cdef f64[::1] cmass
    cdef f64[::1] height
    cdef f64[::1] tau_ref
    cdef f64[::1] temperature
    cdef f64[::1] ne
    cdef f64[::1] vlos
    cdef f64[:,::1] vlosMu
    cdef f64[::1] B
    cdef f64[::1] gammaB
    cdef f64[::1] chiB
    cdef f64[:,::1] cosGamma
    cdef f64[:,::1] cos2chi
    cdef f64[:,::1] sin2chi
    cdef f64[::1] vturb
    cdef f64[::1] nHtot
    cdef f64[::1] muz
    cdef f64[::1] muy
    cdef f64[::1] mux
    cdef f64[::1] wmu

    def __init__(self, atmos):
        self.cmass = atmos.cmass
        self.height = atmos.height
        self.tau_ref = atmos.tau_ref
        self.temperature = atmos.temperature
        self.ne = atmos.ne
        self.vlos = atmos.vlos
        self.vturb = atmos.vturb
        self.nHtot = atmos.nHTot
        self.muz = atmos.muz
        self.muy = atmos.muy
        self.mux = atmos.mux
        self.wmu = atmos.wmu
        self.atmos.cmass = f64_view(self.cmass)
        self.atmos.height = f64_view(self.height)
        self.atmos.tau_ref = f64_view(self.tau_ref)
        self.atmos.temperature = f64_view(self.temperature)
        self.atmos.ne = f64_view(self.ne)
        self.atmos.vlos = f64_view(self.vlos)
        self.atmos.vturb = f64_view(self.vturb)
        self.atmos.nHtot = f64_view(self.nHtot)
        self.atmos.muz = f64_view(self.muz)
        self.atmos.muy = f64_view(self.muy)
        self.atmos.mux = f64_view(self.mux)
        self.atmos.wmu = f64_view(self.wmu)

        cdef int Nspace = atmos.Nspace
        self.atmos.Nspace = Nspace
        cdef int Nrays = atmos.Nrays
        self.atmos.Nrays = Nrays

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

        if atmos.lowerBc == BoundaryCondition.Zero:
            self.atmos.lowerBc = ZERO
        elif atmos.lowerBc == BoundaryCondition.Thermalised:
            self.atmos.lowerBc = THERMALISED
        else:
            raise ValueError('Unknown lowerBc')

        if atmos.upperBc == BoundaryCondition.Zero:
            self.atmos.upperBc = ZERO
        elif atmos.upperBc == BoundaryCondition.Thermalised:
            self.atmos.upperBc = THERMALISED
        else:
            raise ValueError('Unknown lowerBc')

        self.atmos.update_projections()

    @property
    def Nspace(self):
        return self.atmos.Nspace

    @property
    def Nrays(self):
        return self.atmos.Nrays

    
cdef class LwBackground:
    cdef Background background
    cdef BackgroundData bd
    cdef LwAtmosphere atmos
    cdef object eqPops
    cdef object radSet

    cdef f64[::1] chPops
    cdef f64[::1] ohPops
    cdef f64[::1] h2Pops
    cdef f64[::1] hMinusPops
    cdef f64[:,::1] hPops

    cdef f64[::1] wavelength
    cdef f64[:,::1] chi
    cdef f64[:,::1] eta
    cdef f64[:,::1] sca

    def __init__(self, atmosphere, eqPops, radSet, spect):
        cdef LwAtmosphere atmos = atmosphere
        self.atmos = atmos
        self.eqPops = eqPops
        self.radSet = radSet
        self.bd.atmos = &atmos.atmos

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

        self.wavelength = spect.wavelength
        self.bd.wavelength = f64_view(self.wavelength)

        cdef int Nlambda = self.wavelength.shape[0]
        cdef int Nspace = self.atmos.Nspace

        self.chi = np.zeros((Nlambda, Nspace))
        self.bd.chi = f64_view_2(self.chi)
        self.eta = np.zeros((Nlambda, Nspace))
        self.bd.eta = f64_view_2(self.eta)
        self.sca = np.zeros((Nlambda, Nspace))
        self.bd.scatt = f64_view_2(self.sca)

        basic_background(&self.bd)
        self.rayleigh_scattering()
        self.bf_opacities()

        cdef int la, k
        for la in range(Nlambda):
            for k in range(Nspace):
                self.chi[la, k] += self.sca[la, k]

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

    cpdef rayleigh_scattering(self):
        cdef f64[::1] sca = np.zeros(self.atmos.Nspace)
        cdef int k, la
        cdef RayleighScatterer rayH, rayHe

        if 'H' in self.radSet:
            rayH = RayleighScatterer(self.atmos, self.radSet['H'], self.eqPops['H'])
            for la in range(self.wavelength.shape[0]):
                if rayH.scatter(self.wavelength[la], sca):
                    for k in range(self.atmos.Nspace):
                        self.sca[la, k] += sca[k]

        if 'He' in self.radSet:
            rayHe = RayleighScatterer(self.atmos, self.radSet['He'], self.eqPops['He'])
            for la in range(self.wavelength.shape[0]):
                if rayHe.scatter(self.wavelength[la], sca):
                    for k in range(self.atmos.Nspace):
                        self.sca[la, k] += sca[k]

    cpdef bf_opacities(self):
        atoms = self.radSet.passiveAtoms
        print([a.name for a in atoms])
        if len(atoms) == 0:
            return

        continua = []
        cdef f64 sigma0 = 32.0 / (3.0 * sqrt(3.0)) * Const.Q_ELECTRON**2 / (4.0 * np.pi * Const.EPSILON_0) / (Const.M_ELECTRON * Const.CLIGHT) * Const.HPLANCK / (2.0 * Const.E_RYDBERG)
        for a in atoms:
            for c in a.continua:
                continua.append(c)

        cdef f64[:, ::1] alpha = np.zeros((self.wavelength.shape[0], len(continua)))
        cdef int i, la, k, Z
        cdef f64 nEff, gbf_0, wav, edge, lambdaMin
        # TODO(cmo): Check if differences between continuum types actuually matter for us
        for i, c in enumerate(continua):
            # if c.atom.name.strip() == 'H':
            #     Z = c.jLevel.stage
            #     nEff = Z * np.sqrt(Const.E_RYDBERG / (c.jLevel.E_SI - c.iLevel.E_SI))
            #     # gbf_0 = Gaunt_bf(c.lambda0, nEff, Z)
            #     for la in range(self.wavelength.shape[0]):
            #         wav = self.wavelength[la]
            #         if wav >= c.wavelength[0] and wav <= c.wavelength[-1]:
            #             alpha[la, i] = sigma0 * (self.wavelength[la] / c.lambda0)**3 * nEff * Gaunt_bf(self.wavelength[la], nEff, Z)
            # else:
            edge = c.lambdaEdge
            lambdaMin = c.minLambda
            # interpolator = interp1d(c.wavelength, c.alpha, bounds_error=False, fill_value=0.0, kind=1)
            # alphaLa = interpolator(self.wavelength)
            alphaLa = c.compute_alpha(np.asarray(self.wavelength))
            for la in range(self.wavelength.shape[0]):
                if self.wavelength[la] <= edge and self.wavelength[la] >= lambdaMin:
                    alpha[la, i] = alphaLa[la]

        cdef f64[:, ::1] expla = np.zeros((self.wavelength.shape[0], self.atmos.Nspace))
        cdef f64 hc_k = Const.HC / (Const.KBOLTZMANN * Const.NM_TO_M)
        cdef f64 twohc = (2.0 * Const.HC) / Const.NM_TO_M**3
        cdef f64 hc_kla
        for la in range(self.wavelength.shape[0]):
            hc_kla = hc_k / self.wavelength[la]
            for k in range(self.atmos.Nspace):
                expla[la, k] = exp(-hc_kla / self.atmos.temperature[k])
        
        cdef f64 twohnu3_c2
        cdef f64 gijk
        cdef int ci
        cdef int cj
        cdef f64[:,::1] nStar
        cdef f64[:,::1] n
        for i, c in enumerate(continua):
            nStar = self.eqPops.atomicPops[c.atom.name].nStar
            n = self.eqPops.atomicPops[c.atom.name].n

            ci = c.i
            cj = c.j
            for la in range(self.wavelength.shape[0]):
                twohnu3_c2 = twohc / self.wavelength[la]**3
                for k in range(self.atmos.Nspace):
                    gijk = nStar[ci, k] / nStar[cj, k] * expla[la, k]
                    self.chi[la, k] += alpha[la, i] * (1.0 - expla[la, k]) * n[ci, k]
                    self.eta[la, k] += twohnu3_c2 * gijk * alpha[la, i] * n[cj, k]

cdef class RayleighScatterer:
    cdef f64 lambdaLimit
    cdef LwAtmosphere atmos
    cdef f64 C
    cdef f64 sigmaE
    cdef f64[:,::1] pops
    cdef object atom
    cdef bool_t lines

    def __init__(self, atmos, atom, pops):
        if len(atom.lines) == 0:
            self.lines = False
            return

        self.lines = True
        cdef f64 lambdaLimit = 1e6
        cdef f64 lambdaRed
        for l in atom.lines:
            if l.i == 0:
                lambdaRed = l.wavelength[-1]
                lambdaLimit = min(lambdaLimit, lambdaRed)

        self.lambdaLimit = lambdaLimit
        self.atom = atom
        self.atmos = atmos
        self.pops = pops

        C = Const
        self.C = 2.0 * np.pi * (C.Q_ELECTRON / C.EPSILON_0) * C.Q_ELECTRON / C.M_ELECTRON / C.CLIGHT
        self.sigmaE = 8.0 * np.pi / 3.0 * (C.Q_ELECTRON / (np.sqrt(4.0 * np.pi * C.EPSILON_0) * (np.sqrt(C.M_ELECTRON) * C.CLIGHT)))**4

    cpdef scatter(self, f64 wavelength, f64[::1] sca):
        if wavelength <= self.lambdaLimit:
            return False
        if not self.lines:
            return False

        cdef f64 fomega = 0.0
        cdef f64 g0 = self.atom.levels[0].g
        cdef f64 lambdaRed
        cdef f64 f
        for l in self.atom.lines:
            if l.i != 0:
                continue
            
            lambdaRed = l.wavelength[-1]
            if wavelength > lambdaRed:
                lambda2 = 1.0 / ((wavelength / l.lambda0)**2 - 1.0)
                f = l.Aji * (l.jLevel.g / g0) * (l.lambda0 * Const.NM_TO_M)**2 / self.C
                fomega += f * lambda2**2

        cdef f64 sigmaRayleigh = self.sigmaE * fomega

        cdef int k
        for k in range(self.atmos.Nspace):
            sca[k] = sigmaRayleigh * self.pops[0, k]

        return True

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
    cdef object transModel
    cdef object atmos
    cdef object atom

    def __init__(self, trans, compAtom, atmos, spect):
        self.transModel = trans
        self.atom = compAtom
        self.atmos = atmos
        self.wavelength = trans.wavelength
        self.trans.wavelength = f64_view(self.wavelength)
        self.trans.i = trans.i
        self.trans.j = trans.j
        self.trans.polarised = False

        cdef int tIdx = spect.transitions.index(trans)
        self.trans.Nblue = spect.blueIdx[tIdx]

        if isinstance(trans, AtomicLine):
            self.trans.type = LINE
            self.trans.Aji = trans.Aji
            self.trans.Bji = trans.Bji
            self.trans.Bij = trans.Bij
            self.trans.lambda0 = trans.lambda0
            self.trans.dopplerWidth = Const.CLIGHT / self.trans.lambda0
            self.Qelast = np.zeros(self.atmos.Nspace)
            self.aDamp = np.zeros(self.atmos.Nspace)
            self.trans.Qelast = f64_view(self.Qelast)
            self.trans.aDamp = f64_view(self.aDamp)
            self.phi = np.zeros((self.transModel.Nlambda, self.atmos.Nrays, 2, self.atmos.Nspace))
            self.wphi = np.zeros(self.atmos.Nspace)
            self.trans.phi = f64_view_4(self.phi)
            self.trans.wphi = f64_view(self.wphi)
            self.compute_phi()
            if trans.type == LineType.PRD:
                self.rhoPrd = np.ones((self.transModel.Nlambda, self.atmos.Nspace))
                self.trans.rhoPrd = f64_view_2(self.rhoPrd)
        else:
            self.trans.type = CONTINUUM
            self.alpha = trans.alpha
            self.trans.alpha = f64_view(self.alpha)
            self.trans.dopplerWidth = 1.0
            self.trans.lambda0 = trans.lambda0
        
        self.active = np.zeros(len(spect.activeSet), np.int8)
        cdef int i
        for i, s in enumerate(spect.activeSet):
            if trans in s:
                self.active[i] = 1 # sosumi
        self.trans.active = BoolView(<bool_t*>&self.active[0], self.active.shape[0])

        self.Rij = np.zeros(self.atmos.Nspace)
        self.Rji = np.zeros(self.atmos.Nspace)
        self.trans.Rij = f64_view(self.Rij)
        self.trans.Rji = f64_view(self.Rji)
    
    cpdef compute_phi(self):
        if self.type == 'Continuum':
            return

        cdef:
            np.ndarray[np.double_t, ndim=1] Qelast
            np.ndarray[np.double_t, ndim=1] aDamp

        cdef LwAtom atom = self.atom
        aDamp, Qelast = self.transModel.damping(self.atmos, atom.vBroad)

        cdef Atmosphere* atmos = atom.atom.atmos
        cdef int i
        for i in range(Qelast.shape[0]):
            self.Qelast[i] = Qelast[i]
            self.aDamp[i] = aDamp[i]

        self.trans.compute_phi(atmos[0], self.trans.aDamp, atom.atom.vBroad)

    cpdef compute_polarised_profiles(self):
        if self.type == 'Continuum':
            return

        if not self.transModel.polarisable:
            return

        cdef int Nlambda = self.transModel.Nlambda
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
        aDamp, Qelast = self.transModel.damping(self.atmos, atom.vBroad)

        cdef Atmosphere* atmos = atom.atom.atmos
        cdef int i
        for i in range(Qelast.shape[0]):
            self.Qelast[i] = Qelast[i]
            self.aDamp[i] = aDamp[i]

        z = self.transModel.zeeman_components()
        cdef LwZeemanComponents zc = LwZeemanComponents(z)

        self.trans.compute_polarised_profiles(atmos[0], self.trans.aDamp, atom.atom.vBroad, zc.zc)

    def uv(self, int la, int mu, bool_t toObs, f64[::1] Uji not None, f64[::1] Vij not None, f64[::1] Vji not None):
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
    cdef f64[:,::1] V
    cdef f64[:,::1] gij
    cdef f64[:,::1] wla
    cdef object atomicTable
    cdef object atomicModel
    cdef object atmos
    cdef list trans

    def __init__(self, atom, cAtmos, atmos, eqPops, spect, background, initSol=None, ngOptions=None):
        self.atomicModel = atom
        self.atmos = atmos
        cdef LwAtmosphere a = cAtmos
        self.atom.atmos = &a.atmos
        modelPops = eqPops.atomicPops[atom.name]
        self.atomicTable = modelPops.model.atomicTable
        vTherm = 2.0 * Const.KBOLTZMANN / (Const.AMU * modelPops.weight)
        self.vBroad = np.sqrt(vTherm * atmos.temperature + atmos.vturb**2)
        self.atom.vBroad = f64_view(self.vBroad)
        self.nTotal = modelPops.nTotal
        self.atom.nTotal = f64_view(self.nTotal)

        self.trans = []
        for l in atom.lines:
            self.trans.append(LwTransition(l, self, atmos, spect))
        
        for c in atom.continua:
            self.trans.append(LwTransition(c, self, atmos, spect))

        cdef LwTransition lt
        for lt in self.trans:
            self.atom.trans.push_back(&lt.trans)

        cdef int Nlevel = len(atom.levels)
        cdef int Ntrans = len(self.trans)
        self.atom.Nlevel = Nlevel
        self.atom.Ntrans = Ntrans

        self.Gamma = np.zeros((Nlevel, Nlevel, a.Nspace))
        self.atom.Gamma = f64_view_3(self.Gamma)

        self.C = np.zeros((Nlevel, Nlevel, atmos.Nspace))
        self.atom.C = f64_view_3(self.C)

        # TODO(cmo): Rewrite this to reuse an n and nStar from eqPops
        # self.nStar = np.zeros((self.Nlevel, atmos.Nspace))
        self.nStar = modelPops.nStar
        self.atom.nStar = f64_view_2(self.nStar)

        # self.lte_pops()
        self.n = np.copy(self.nStar)
        self.atom.n = f64_view_2(self.n)
        modelPops.pops = np.asarray(self.n)

        self.eta = np.zeros(atmos.Nspace)
        self.atom.eta = f64_view(self.eta)
        self.gij = np.zeros((Ntrans, atmos.Nspace))
        self.atom.gij = f64_view_2(self.gij)
        self.wla = np.zeros((Ntrans, atmos.Nspace))
        self.atom.wla = f64_view_2(self.wla)

        self.V = np.zeros((Nlevel, atmos.Nspace))
        self.atom.V = f64_view_2(self.V)
        self.U = np.zeros((Nlevel, atmos.Nspace))
        self.atom.U = f64_view_2(self.U)

        self.chi = np.zeros((Nlevel, atmos.Nspace))
        self.atom.chi = f64_view_2(self.chi)

        if initSol is None:
            initSol = InitialSolution.EscapeProbability

        if initSol == InitialSolution.Zero:
            raise ValueError('Zero radiation InitialSolution not currently supported')

        cdef np.ndarray[np.double_t, ndim=3] Gamma
        cdef np.ndarray[np.double_t, ndim=3] C
        cdef f64 delta
        cdef LwBackground bg = background

        if initSol == InitialSolution.EscapeProbability:
            self.compute_collisions()
            Gamma = np.asarray(self.Gamma)
            C = np.asarray(self.C)

            self.atom.ng = Ng(0,0,0, self.atom.n.flatten())
            start = time.time()
            for it in range(100):
                Gamma.fill(0.0)
                Gamma += C
                gamma_matrices_escape_prob(&self.atom, bg.background, a.atmos)
                stat_eq(&self.atom)
                self.atom.ng.accelerate(self.atom.n.flatten())
                delta = self.atom.ng.max_change()
                if delta < 1e-2:
                    end = time.time()
                    print('Converged: %s, %d\nTime: %f' % (self.atomicModel.name, it, end-start))
                    break
            else:
                print('Escape probability didn\'t converge for %s, setting LTE populations' % self.atomicModel.name)
                n = np.asarray(self.n)
                n[:] = np.asarray(self.nStar)

        if ngOptions is not None:
            self.atom.ng = Ng(ngOptions.Norder, ngOptions.Nperiod, ngOptions.Ndelay, self.atom.n.flatten())
        else:
            self.atom.ng = Ng(0,0,0, self.atom.n.flatten())

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
    def n(self):
        return np.asarray(self.n)

    @property
    def nStar(self):
        return np.asarray(self.nStar)

    @property
    def trans(self):
        return self.trans

    cpdef lte_pops(self, debeye=True):
        cdef f64 c1 = (Const.HPLANCK / (2.0 * np.pi * Const.M_ELECTRON)) * (Const.HPLANCK / Const.KBOLTZMANN)
        cdef c2 = 0.0
        cdef int i, Z
        if debeye:
            c2 = np.sqrt(8.0 * np.pi / Const.KBOLTZMANN) * (Const.Q_ELECTRON**2 / (4.0 * np.pi * Const.EPSILON_0))**1.5
            nDebeye = np.zeros(self.Nlevel)
            for i in range(1, self.Nlevel):
                stage = self.atomicModel.levels[i].stage
                Z = stage
                for m in range(1, stage - self.atomicModel.levels[0].stage + 1):
                    nDebeye[i] += Z
                    Z += 1
        dEion = c2 * np.sqrt(self.atmos.ne / self.atmos.temperature)
        cNe_T = 0.5  * self.atmos.ne * (c1 / self.atmos.temperature)**1.5
        total = np.ones(self.atmos.Nspace)

        nStar = np.asarray(self.nStar)
        ground = self.atomicModel.levels[0]
        cdef f64 dE, gi0
        cdef int dZ
        for i in range(1, self.Nlevel):
            dE = self.atomicModel.levels[i].E_SI - ground.E_SI
            gi0 = self.atomicModel.levels[i].g / ground.g
            dZ = self.atomicModel.levels[i].stage - ground.stage
            if debeye:
                dE_kT = (dE - nDebeye[i] * dEion) / (Const.KBOLTZMANN * self.atmos.temperature)
            else:
                dE_kT = dE / (Const.KBOLTZMANN * self.atmos.temperature)

            nst = gi0 * np.exp(-dE_kT)
            nStar[i, :] = nst
            nStar[i, :] /= cNe_T**dZ
            # for m in range(1, dZ + 1):
            #     nStar[i, :] /= cNe_T
            total +=  nStar[i]

        nStar[0] = self.nTotal / total
        for i in range(1, self.Nlevel):
            nStar[i] *= nStar[0]

    def compute_collisions(self):
        cdef np.ndarray[np.double_t, ndim=3] C = np.asarray(self.C)
        C.fill(0.0)
        cdef np.ndarray[np.double_t, ndim=2] nStar = np.asarray(self.nStar)
        for col in self.atomicModel.collisions:
            col.compute_rates(self.atmos, nStar, C)
        C[C < 0.0] = 0.0

cdef class LwSpectrum:
    cdef Spectrum spect
    cdef f64[::1] wavelength
    cdef f64[:,::1] I
    cdef f64[:,::1] J
    cdef f64[:,:,::1] Quv

    def __init__(self, wavelength, Nrays, Nspace):
        self.wavelength = wavelength
        cdef int Nspect = self.wavelength.shape[0]
        self.I = np.zeros((Nspect, Nrays))
        self.J = np.zeros((Nspect, Nspace))

        self.spect.wavelength = f64_view(self.wavelength)
        self.spect.I = f64_view_2(self.I)
        self.spect.J = f64_view_2(self.J)

    def setup_stokes(self):
        self.Quv = np.zeros((3, self.I.shape[0], self.I.shape[1]))
        self.spect.Quv = f64_view_3(self.Quv)

    @property
    def wavelength(self):
        return np.asarray(self.wavelength)

    @property
    def I(self):
        return np.asarray(self.I)

    @property
    def J(self):
        return np.asarray(self.J)

    @property
    def Quv(self):
        return np.asarray(self.Quv)


cdef class LwContext:
    cdef Context ctx
    cdef LwAtmosphere atmos
    cdef LwSpectrum spect
    cdef LwBackground background
    cdef dict arguments
    cdef object atomicTable
    cdef list activeAtoms

    def __init__(self, atmos, spect, radSet, eqPops, atomicTable, ngOptions=None, initSol=None):
        self.arguments = {'atmos': atmos, 'spect': spect, 'radSet': radSet, 'eqPops': eqPops, 'atomicTable': atomicTable, 'ngOptions': ngOptions, 'initSol': initSol}

        self.atmos = LwAtmosphere(atmos)
        self.spect = LwSpectrum(spect.wavelength, atmos.Nrays, atmos.Nspace)
        self.atomicTable = atomicTable

        self.background = LwBackground(self.atmos, eqPops, radSet, spect)

        activeAtoms = radSet.activeAtoms
        self.activeAtoms = [LwAtom(a, self.atmos, atmos, eqPops, spect, self.background, ngOptions=ngOptions, initSol=initSol) for a in activeAtoms]

        self.ctx.atmos = &self.atmos.atmos
        self.ctx.spect = &self.spect.spect
        self.ctx.background = &self.background.background

        cdef LwAtom la
        for la in self.activeAtoms:
            self.ctx.activeAtoms.push_back(&la.atom)

    def gamma_matrices_formal_sol(self):
        cdef LwAtom atom
        cdef np.ndarray[np.double_t, ndim=3] Gamma
        for atom in self.activeAtoms:
            Gamma = np.asarray(atom.Gamma)
            Gamma.fill(0.0)
            atom.compute_collisions()
            Gamma += atom.C

        cdef f64 dJ = gamma_matrices_formal_sol(self.ctx)
        print('dJ = %.2e' % dJ)
        return dJ

    def stat_equil(self):
        atoms = self.activeAtoms

        cdef LwAtom atom
        cdef Atom* a
        cdef f64 delta
        cdef f64 maxDelta = 0.0
        cdef bool_t accelerated

        for atom in atoms:
            a = &atom.atom
            if not a.ng.init:
                a.ng.accelerate(a.n.flatten())
            stat_eq(a)
            accelerated = a.ng.accelerate(a.n.flatten())
            delta = a.ng.max_change()
            s = '    %s delta = %6.4e' % (atom.atomicModel.name, delta)
            if accelerated:
                s += ' (accelerated)'

            print(s)
            maxDelta = max(maxDelta, delta)

        return maxDelta

    def update_projections(self):
        self.atmos.atmos.update_projections()

    def single_stokes_fs(self):
        assert self.atmos.B.shape[0] != 0

        for atom in self.activeAtoms:
            for t in atom.trans:
                t.compute_polarised_profiles()

        self.spect.setup_stokes()

        cdef f64 dJ = formal_sol_full_stokes(self.ctx)
        return dJ

    def prd_redistribute(self, int maxIter=3, f64 tol=1e-2):
        cdef f64 dRho = redistribute_prd_lines(self.ctx, maxIter, tol)
        print('      PRD dRho = %.2e' % dRho)
        return dRho

    def configure_hprd_coeffs(self):
        configure_hprd_coeffs(self.ctx)

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










def blah():
    cdef f64[::1] a
    a = np.zeros(10)
    print(type(a))

cdef class Blah:
    cdef f64[::1] a
    cdef f64[::1] b
    cdef int c

    def __init__(self):
        self.a = np.zeros(10)
        c = 3

    def printit(self):
        print(dir(self))


