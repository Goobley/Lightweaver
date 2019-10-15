import numpy as np
from scipy import special
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List
from AtomicModel import AtomicModel, AtomicLine, AtomicContinuum
from AtomicSet import RadiativeSet, SpectrumConfiguration
from AtomicTable import AtomicTable
import Constants as Const
from Atmosphere import Atmosphere, ScaleType
import matplotlib.pyplot as plt
from scipy.linalg import solve
from typing import Union
from numba import njit
from witt import witt

# NOTE(cmo): Easiest way to get the Voigt out of scipy. No idea what the precision is
def voigt_H(a, v):
    z = (v + 1j * a)
    return special.wofz(z).real
    
# NOTE(cmo): The Voigt G (used in the psi profile for Full Stokes in RH), simply appears to be the complex part of the same function https://doi.org/10.1016/j.jqsrt.2006.08.011
def voigt_G(a, v):
    z = (v + 1j * a)
    return special.wofz(z).complex

class TransitionType(Enum):
    Line = auto()
    Continuum = auto()

class InitialSolution(Enum):
    Lte = auto()
    Zero = auto()

# NOTE(cmo): Uij, Vij, Vji defined as per RH paper. We relate these by the expressions given in Uitenbroek 2001
@dataclass
class UV:
    # Uij = 0
    Uji: np.ndarray
    Vij: np.ndarray
    Vji: np.ndarray

# NOTE(cmo): Is it a line? Is it a continuum? This represents both!
class ComputationalTransition:
    def __init__(self, trans: Union[AtomicLine, AtomicContinuum], compAtom: 'ComputationalAtom', atmos: Atmosphere, spect: SpectrumConfiguration):
        self.transModel = trans
        self.atom = compAtom
        self.wavelength = trans.wavelength
        if isinstance(trans, AtomicLine):
            self.type = TransitionType.Line
            self.Aji = trans.Aji
            self.Bji = trans.Bji
            self.Bij = trans.Bij
            self.lambda0 = trans.lambda0
        else:
            self.type = TransitionType.Continuum

        self.i = trans.i
        self.j = trans.j

        tIdx = spect.transitions.index(trans)
        # The index for the starting point of the transition-local wavelength array in the global wavelength array
        self.Nblue = spect.blueIdx[tIdx]

        self.atmos = atmos

        # Compute the line profile, if we're a line
        self.compute_phi()

        # Store the transitions where we're active
        self.active = np.zeros(len(spect.activeSet), np.bool)
        for i, s in enumerate(spect.activeSet):
            if trans in s:
                self.active[i] = True

        # "our" gij will be given to us by the Atom that owns us
        self.gij = None

    # @property
    # def j(self):
    #     return self.transModel.j

    @property
    def jLevel(self):
        return self.transModel.jLevel

    # @property
    # def i(self):
    #     return self.transModel.i

    @property
    def iLevel(self):
        return self.transModel.iLevel

    # @property
    # def Aji(self):
    #     return self.transModel.Aji

    # @property
    # def Bji(self):
    #     return self.transModel.Bji

    # @property
    # def Bij(self):
    #     return self.transModel.Bij

    @property
    def alpha(self):
        return self.transModel.alpha

    # @property
    # def lambda0(self):
    #     return self.transModel.lambda0

    # NOTE(cmo): Wavelength weights
    def wlambda(self, la: Optional[int]=None) -> Union[float, np.ndarray]:
        if self.type == TransitionType.Line:
            dopplerWidth = Const.CLIGHT / self.lambda0
        else:
            dopplerWidth = 1.0

        if la is not None:
            if la == 0:
                return 0.5 * (self.wavelength[1] - self.wavelength[0]) * dopplerWidth
            elif la == self.wavelength.shape[0]-1:
                return 0.5 * (self.wavelength[-1] - self.wavelength[-2]) * dopplerWidth
            else:
                return 0.5 * (self.wavelength[la+1] - self.wavelength[la-1]) * dopplerWidth

        wla = np.zeros_like(self.wavelength)
        wla[0] = 0.5 * (self.wavelength[1] - self.wavelength[0])
        wla[-1] = 0.5 * (self.wavelength[-1] - self.wavelength[-2])
        wla[1:-1] = 0.5 * (self.wavelength[2:] - self.wavelength[:-2])

        return dopplerWidth * wla

    # NOTE(cmo): A poorly named "lambda transition" i.e. the mapping between la in the global array and lt into the transition local wavelength array (where the local array is a contiguous subset of the global array)
    def lt(self, la: int) -> int:
        return la - self.Nblue

    # NOTE(cmo): Computes the emission profile, and the associated integration weights. These are stored in self.phi and self.wphi respectively. Phi has dimensions[Nlambda, Nrays, Up/Down, Nspace], wphi is simply [Nspace] as it is integrated across the other dimensions of phi
    def compute_phi(self):
        if self.type == TransitionType.Continuum:
            return

        sqrtPi = np.sqrt(np.pi)
        # get damping
        aDamp = self.transModel.damping(self.atmos, self.atom.vBroad)
        phi = np.zeros((self.transModel.Nlambda, self.atmos.Nrays, 2, self.atmos.Nspace))
        wPhi = np.zeros(self.atmos.Nspace)

        wlambda = self.wlambda()

        vLos = np.zeros((self.atmos.Nrays, self.atmos.Nspace))
        for mu in range(self.atmos.Nrays):
            vLos[mu, :] = self.atmos.muz[mu] * self.atmos.vlos / self.atom.vBroad

        for la in range(self.wavelength.shape[0]):
            v = (self.wavelength[la] - self.lambda0) * Const.CLIGHT / (self.atom.vBroad * self.lambda0)
            for mu in range(self.atmos.Nrays):
                wlamu = wlambda * 0.5 * self.atmos.wmu[mu]
                for toFrom, sign in enumerate([-1.0, 1.0]):
                    vk = v + sign * vLos[mu]
                    # Will probably end up needing some None slicing to make this work
                    phi[la, mu, toFrom, :] = voigt_H(aDamp, vk) / (sqrtPi * self.atom.vBroad)
                    # wPhi[:] += (np.sum(phi, axis=(1,2)) * wlamu[:, None]).sum(axis=0)
                    wPhi[:] += phi[la, mu, toFrom, :] * wlamu[la]

        # Store inverse of weight
        self.wphi = 1.0 / wPhi
        self.phi = phi

    # NOTE(cmo): Computes the U's and V's as specified in the RH paper, using the Uitenbroek 2001 expressions
    def uv(self, la, mu, toFrom) -> UV:
        lt = self.lt(la)
        # hc_4pil = 0.25 * Const.HC / np.pi / self.wavelength[lt]
        hc_4pi = 0.25 * Const.HC / np.pi
        # Vij = np.zeros(self.atmos.Nspace)
        # Vji = np.zeros(self.atmos.Nspace)
        # Uji = np.zeros(self.atmos.Nspace)

        if self.type == TransitionType.Line:
            phi = self.phi[lt, mu, toFrom, :]
            Vij = hc_4pi * self.Bij * phi
            Vji = self.gij * Vij
            # Uji = hc_4pi * self.Aji * phi
            Uji = self.Aji / self.Bji * Vji
        else:
            # Vij = hc_4pi * self.alpha[lt]
            Vij = self.alpha[lt]
            Vji = self.gij * Vij
            Uji = 2 * Const.HC / (Const.NM_TO_M *self.wavelength[lt])**3 * Vji

        return UV(Uji, Vij, Vji)


class ComputationalAtom:
    def __init__(self, atom: AtomicModel, atmos: Atmosphere, atomicTable: AtomicTable, spect: SpectrumConfiguration, initSol: InitialSolution=InitialSolution.Lte):
        self.atomicModel = atom
        self.atmos = atmos
        self.atomicTable = atomicTable
        vTherm = 2.0 * Const.KBOLTZMANN / (Const.AMU * atomicTable[atom.name].weight)
        self.vBroad = np.sqrt(vTherm * atmos.temperature + atmos.vturb**2)
        self.spect = spect
        self.ntotal = atomicTable[atom.name].abundance * atmos.nHTot

        self.trans: List[ComputationalTransition] = []
        for l in atom.lines:
            self.trans.append(ComputationalTransition(l, self, atmos, spect))
        for c in atom.continua:
            self.trans.append(ComputationalTransition(c, self, atmos, spect))

        self.Nlevel = len(atom.levels)
        self.Ntrans = len(self.trans)

        self.setup_Gamma()

        # THis needs to be initialised correctly...
        # CaII normally wants the zero-radiation solution
        # Ivan's goes from LTE
        self.n = np.zeros((self.Nlevel, atmos.Nspace))
        self.nstar = np.zeros((self.Nlevel, atmos.Nspace))

        self.lte_pops()
        self.n = np.copy(self.nstar)

        if initSol != InitialSolution.Lte:
            raise ValueError("Currently only LTE InitialSolution supported")

    def lte_pops(self, debye=False):
        c1 = (Const.HPLANCK / (2.0 * np.pi * Const.M_ELECTRON)) * (Const.HPLANCK / Const.KBOLTZMANN)

        c2 = 0.0
        if debye:
            c2 = np.sqrt(8.0 * np.pi / Const.KBOLTZMANN) * (Const.Q_ELECTRON**2 / (4.0 * np.pi * Const.EPSILON_0))**1.5
            nDebye = np.zeros(self.Nlevel)
            for i in range(1, self.Nlevel):
                stage = self.atomicModel.levels[i].stage
                Z = stage
                for m in range(1, stage - self.atomicModel.levels[0].stage + 1):
                    Z += 1
                    nDebye[i] += Z

        dEion = c2 * np.sqrt(self.atmos.ne / self.atmos.temperature)
        cNe_T = 0.5 * self.atmos.ne * (c1 / self.atmos.temperature)**1.5
        total = np.ones(self.atmos.Nspace)

        ground = self.atomicModel.levels[0]
        for i in range(1, self.Nlevel):
            dE = self.atomicModel.levels[i].E_SI - ground.E_SI
            gi0 = self.atomicModel.levels[i].g / ground.g
            dZ = self.atomicModel.levels[i].stage - ground.stage
            if debye:
                dE_kT = (dE - nDebye[i] * dEion) / (Const.KBOLTZMANN * self.atmos.temperature)
            else:
                dE_kT = dE / (Const.KBOLTZMANN * self.atmos.temperature)

            nst = gi0 * np.exp(-dE_kT)
            self.nstar[i, :] = nst
            for m in range(1, dZ + 1):
                self.nstar[i, :] /= cNe_T
            total += self.nstar[i]

        self.nstar[0] = self.ntotal / total

        for i in range(1, self.Nlevel):
            self.nstar[i] *= self.nstar[0]

        
    def setup_Gamma(self):
        self.Gamma = np.zeros((self.Nlevel, self.Nlevel, self.atmos.Nspace))

    def setup_wavelength(self, laIdx: int):
        self.eta = np.zeros(self.atmos.Nspace)
        self.gij = np.zeros((self.Ntrans, self.atmos.Nspace))
        self.wla = np.zeros((self.Ntrans, self.atmos.Nspace))
        self.V = np.zeros((self.Nlevel, self.atmos.Nspace))
        self.U = np.zeros((self.Nlevel, self.atmos.Nspace))
        self.chi = np.zeros((self.Nlevel, self.atmos.Nspace))

        hc_k = Const.HC / (Const.KBOLTZMANN * Const.NM_TO_M)
        h_4pi = 0.25 * Const.HPLANCK / np.pi
        hc_4pi = h_4pi * Const.CLIGHT
        for kr, t in enumerate(self.trans):
            if not t.active[laIdx]:
                continue

            if t.type == TransitionType.Line:
                self.gij[kr, :] = t.Bji / t.Bij
                self.wla[kr, :] = t.wlambda(t.lt(laIdx)) * t.wphi / hc_4pi
                # self.V[t.i, :] = hc_4pi * t.Bij * t.phi[t.lt(laIdx)]
                # self.V[t.j, :] = self.gij[kr] * self.V[t.i]
            else:
                self.gij[kr, :] = self.nstar[t.i] / self.nstar[t.j] \
                    * np.exp(-hc_k / self.spect.wavelength[laIdx] / self.atmos.temperature)
                self.wla[kr, :] = t.wlambda(t.lt(laIdx)) / self.spect.wavelength[laIdx] / h_4pi
                # self.V[t.i, :] = t.alpha[t.lt(laIdx)]
                # self.V[t.j, :] = self.gij[kr] * self.V[t.i, :]
            t.gij = self.gij[kr]

    def zero_angle_dependent_vars(self):
        self.eta.fill(0.0)
        self.V.fill(0.0)
        self.U.fill(0.0)
        self.chi.fill(0.0)

    def compute_collisions(self):
        self.C = np.zeros_like(self.Gamma)
        for col in self.atomicModel.collisions:
            col.compute_rates(self.atmos, self.nstar, self.C)
        # NOTE(cmo): RH sometimes returns C < 0 due to the spline interpolation. Want to see if it makes a difference (probably not) -- makes no noticeable difference to line shape -- Han is going to correct this
        self.C[self.C < 0.0] = 0.0
        

@njit
def planck(temp, wav):
    hc_Tkla = Const.HC / (Const.KBOLTZMANN * Const.NM_TO_M * wav) / temp
    twohnu3_c2 = (2.0 * Const.HC) / (Const.NM_TO_M * wav)**3

    return twohnu3_c2 / (np.exp(hc_Tkla) - 1.0)

@dataclass
class Background:
    chi: np.ndarray
    eta: np.ndarray

def background(atmos: Atmosphere, spect: SpectrumConfiguration) -> Background:
    Nspace = atmos.Nspace
    Nspect = spect.wavelength.shape[0]
    eos = witt()
    chi = np.zeros((Nspect, Nspace))
    eta = np.zeros((Nspect, Nspace))
    for k in range(Nspace):
        chi[:, k] = eos.contOpacity(atmos.temperature[k], atmos.pgas[k], atmos.pe[k], spect.wavelength*10) / Const.CM_TO_M

    # Need to find emissivity too -- use Kirchoff's law since bg is LTE i.e. eta_bg = B_nu(T) * k_nu

    # wavelength_m = spect.wavelength * 1e-9
    # cte = 2.0 * Const.KBOLTZMANN / wavelength_m**2
    for k in range(Nspace):
        eta[:, k] = planck(atmos.temperature[k], spect.wavelength) * chi[:, k]

    return Background(chi, eta)

@dataclass
class IPsi:
    I: np.ndarray
    PsiStar: np.ndarray

@njit
def w2(dtau):
    w = np.empty(2)
    if dtau < 5e-4:
        w[0] = dtau * (1.0 - 0.5*dtau)
        w[1] = dtau**2 * (0.5 - dtau / 3.0)
    elif dtau > 50.0:
        w[0] = 1.0
        w[1] = 1.0
    else:
        expdt = np.exp(-dtau)
        w[0] = 1.0 - expdt
        w[1] = w[0] - dtau * expdt
    return w

@njit
def piecewise_1d_impl(muz, toFrom, Istart, z, chi, S):
    Nspace = chi.shape[0]
    zmu = 0.5 / muz

    if toFrom:
        dk = -1
        kStart = Nspace - 1
        kEnd = 0
    else:
        dk = 1
        kStart = 0
        kEnd = Nspace - 1

    dtau_uw = zmu * (chi[kStart] + chi[kStart + dk]) * np.abs(z[kStart] - z[kStart + dk])
    dS_uw = (S[kStart] - S[kStart + dk]) / dtau_uw

    Iupw = Istart
    I = np.empty(Nspace)
    Psi = np.empty(Nspace)
    I[kStart] = Iupw
    Psi[kStart] = 0.0

    for k in range(kStart + dk, kEnd + dk, dk):
        w = w2(dtau_uw)

        if k != kEnd:
            dtau_dw = zmu * (chi[k] + chi[k+dk]) * np.abs(z[k] - z[k+dk])
            dS_dw = (S[k] - S[k+dk]) / dtau_dw
            # print(dS_uw, dtau_uw, Iupw, w)
            I[k] = (1.0 - w[0]) * Iupw + w[0] * S[k] + w[1] * dS_uw
            Psi[k] = w[0] - w[1] / dtau_uw
        else:
            I[k] = (1.0 - w[0]) * Iupw + w[0] * S[k] + w[1] * dS_uw
            Psi[k] = w[0] - w[1] / dtau_uw
        
        Iupw = I[k]
        dS_uw = dS_dw
        dtau_uw = dtau_dw

    return I, Psi

def piecewise_1d(atmos, mu, toFrom, wav, chi, S):
    zmu = 0.5 / atmos.muz[mu]
    z = atmos.height

    if toFrom:
        dk = -1
        kStart = atmos.Nspace - 1
        kEnd = 0
    else:
        dk = 1
        kStart = 0
        kEnd = atmos.Nspace - 1

    dtau_uw = zmu * (chi[kStart] + chi[kStart + dk]) * np.abs(z[kStart] - z[kStart + dk])

    if toFrom:
        if atmos.lowerBc == BoundaryCondition.Zero:
            Iupw = 0.0
        elif atmos.lowerBc == BoundaryCondition.Thermalised:
            Bnu = planck(atmos.temperature[-2:], wav)
            Iupw = Bnu[1] - (Bnu[0] - Bnu[1]) / dtau_uw
    else:
        if atmos.upperBc == BoundaryCondition.Zero:
            Iupw = 0.0
        elif atmos.upperBc == BoundaryCondition.Thermalised:
            Bnu = planck(atmos.temperature[:2], wav)
            Iupw = Bnu[0] - (Bnu[1] - Bnu[0]) / dtau_uw
    # Iupw = 0.0
    I, Psi = piecewise_1d_impl(atmos.muz[mu], toFrom, Iupw, z, chi, S)
    return IPsi(I, Psi/chi)


# @profile
def gamma_matrices(atmos: Atmosphere, spect: SpectrumConfiguration, activeAtoms: List[ComputationalAtom], background: Background):
    Nspace = atmos.Nspace
    Nrays = atmos.Nrays
    Nspect = spect.wavelength.shape[0]
    Iplus = np.zeros((Nspect, Nrays))

    for atom in activeAtoms:
        atom.setup_Gamma()
        atom.compute_collisions()
        atom.Gamma += atom.C #* 1e8
        # for t in atom.trans:
        #     t.compute_phi()

    for la, wave in enumerate(spect.wavelength):
        for atom in activeAtoms:
            atom.setup_wavelength(la)

        for mu in range(Nrays):
            for toFrom, sign in enumerate([-1.0, 1.0]):
                chiTot = np.zeros(Nspace)
                etaTot = np.zeros(Nspace)

                for atom in activeAtoms:
                    # TODO(cmo): Check when things need to be zero'd
                    # NOTE(cmo): HACK HACK HACK vvvv
                    # atom.setup_wavelength(la)
                    atom.zero_angle_dependent_vars()
                    for t in atom.trans:
                        if not t.active[la]:
                            continue

                        uv = t.uv(la, mu, toFrom)
                        chi = atom.n[t.i] * uv.Vij - atom.n[t.j] * uv.Vji
                        eta = atom.n[t.j] * uv.Uji

                        atom.chi[t.i] += chi
                        atom.chi[t.j] -= chi
                        atom.U[t.j] += uv.Uji
                        atom.V[t.i] += uv.Vij
                        atom.V[t.j] += uv.Vji
                        chiTot += chi
                        etaTot += eta
                        atom.eta += eta

                chi = chiTot + background.chi[la]
                s = (etaTot + background.eta[la]) / chi

                # print('\n\n\n----------\n')
                # print(wave)
                ip = piecewise_1d(atmos, mu, toFrom, wave, chi, s)
                Iplus[la, mu] = ip.I[0] 

                if np.any(ip.PsiStar * chi < 0) or np.any(ip.PsiStar * chi > 1.0):
                    print(la, wave)
                    print(ip.PsiStar)
                    raise ValueError('Formal solver exploded!!')

                for atom in activeAtoms:
                    Ieff = ip.I - ip.PsiStar * atom.eta

                    for kr, t in enumerate(atom.trans):
                        if not t.active[la]:
                            continue

                        wmu = 0.5 * atmos.wmu[mu]
                        wlamu = atom.wla[kr] * wmu

                        uv = t.uv(la, mu, toFrom)

                        integrand = (uv.Uji + uv.Vji * Ieff) - (ip.PsiStar * atom.chi[t.i] * atom.U[t.j])
                        atom.Gamma[t.i, t.j] += integrand * wlamu

                        # NOTE(cmo): The sign in this second term differs from RH due to us accumulating atom.chi[j] -= chi, rather than +=
                        integrand = (uv.Vij * Ieff) - (ip.PsiStar * atom.chi[t.j] * atom.U[t.i])
                        atom.Gamma[t.j, t.i] += integrand * wlamu

    for atom in activeAtoms:
        for k in range(Nspace):
            np.fill_diagonal(atom.Gamma[:,:,k], 0.0)
            for i in range(atom.Nlevel):
                GamDiag = np.sum(atom.Gamma[:, i, k])
                atom.Gamma[i, i, k] = -GamDiag

    return Iplus

# @profile
def stat_equil(atmos, activeAtoms):
    Nspace = atmos.Nspace
    maxRelChange = 0.0
    for atom in activeAtoms:
        Nlevel = atom.Nlevel
        for k in range(Nspace):
            iEliminate = np.argmax(atom.n[:, k])
            Gamma = np.copy(atom.Gamma[:, :, k])

            Gamma[iEliminate, :] = 1.0
            nk = np.zeros(Nlevel)
            nk[iEliminate] = atom.ntotal[k]

            nOld = np.copy(atom.n[:, k])
            # if k == 40:
            #     print(Gamma)
            #     print('\n---\n')
            #     print(nk)
            #     print('\n---\n')

            nNew = solve(Gamma, nk)

            # if k == 40:
            #     print(nNew)
            #     print('\n---\n')

            change = np.max(np.where(nNew != 0.0, np.abs(nNew - nOld) / nNew, 0.0))
            maxRelChange = max(maxRelChange, change)
            atom.n[:, k] = nNew

    return maxRelChange



