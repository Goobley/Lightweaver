"""
Simple EOS and background opacity package
Coded in python by J. de la Cruz Rodriguez (ISP-SU 2017)
"""
import numpy as np
from math import *
import os
import xdrlib
from numba import njit
from .utils import get_data_path


class dum: # Just a container for the PF
    nstage = 0
    npi = 0
    pf  = 0
    eion = 0

    def __init__(self):
        pass

# ----------------------------------------------------------------------------------------


class witt:
    """
    EOS described in Mihalas (1970) "Stellar atmospheres"
    It considers in detail H, H+, H- and H2, no other molecules.

    This particular implementation was taken from Wittmann's routines.
    Partition functions from Kurucz.

    Coded in python by J. de la Cruz Rodriguez (ISP-SU 2017)
    Messed with by C. Osborne (University of Glasgow 2019-2020)

    Don't blame Jaime for EOS bugs in Lightweaver!
    **N.B. Everything in here is CGS.**

    Dependencies: pf_Kurucz.input containing the partition function data.

    """
    # Some definitions (from NIST)

    BK = 1.3806488E-16          # Boltzmann [erg K]
    EE = 4.80320441E-10         # Electron charge
    HH = 6.62606957E-27         # Planck [erg s]
    PI = 3.14159265358979323846 # PI number
    CC = 2.99792458E10          # Speed of light [cm/s]
    AMU = 1.660538921E-24       # Atomic mass unit
    EV = 1.602176565E-12        # Electron Volt to erg
    CM1_TO_EV = HH*CC/EV        # CM^-1 to eV
    ME = 9.10938188E-28         # mass of electron

    saha_fac =  ((2.0 * PI * ME * BK) / (HH*HH))**1.5

    # Default abundances

    ABUND = 10.0**np.float64( [-0.04048,-1.07,-10.95,-10.89, -9.44, -3.48, -3.99,
                               -3.11, -7.48, -3.95,  -5.71, -4.46, -5.57, -4.49, -6.59, -4.83,
                               -6.54, -5.48, -6.82,  -5.68, -8.94, -7.05, -8.04, -6.37, -6.65,
                               -4.50, -7.12, -5.79,  -7.83, -7.44, -9.16, -8.63, -9.67, -8.69,
                               -9.41, -8.81, -9.44,  -9.14, -9.80, -9.54,-10.62,-10.12,-20.00,
                               -10.20,-10.92,-10.35, -11.10,-10.18,-10.58,-10.04,-11.04, -9.80,
                               -10.53, -9.81,-10.92,  -9.91,-10.82,-10.49,-11.33,-10.54,-20.00,
                               -11.04,-11.53,-10.92, -11.94,-10.94,-11.78,-11.11,-12.04,-10.96,
                               -11.28,-11.16,-11.91, -10.93,-11.77,-10.59,-10.69,-10.24,-11.03,
                               -10.95,-11.14,-10.19, -11.33,-20.00,-20.00,-20.00,-20.00,-20.00,
                               -20.00,-11.92,-20.00, -12.51,-20.00,-20.00,-20.00,-20.00,-20.00,
                               -20.00,-20.00])


    AMASS = np.float64([1.008,  4.003,  6.941,  9.012, 10.811, 12.011, 14.007, 15.999,
                        18.998, 20.179, 22.990, 24.305, 26.982, 28.086, 30.974, 32.060,
                        35.453, 39.948, 39.102, 40.080, 44.956, 47.900, 50.941, 51.996,
                        54.938, 55.847, 58.933, 58.710, 63.546, 65.370, 69.720, 72.590,
                        74.922, 78.960, 79.904, 83.800, 85.468, 87.620, 88.906, 91.220,
                        92.906, 95.940, 98.906,101.070,102.905,106.400,107.868,112.400,
                        114.820,118.690,121.750,127.600,126.905,131.300,132.905,137.340,
                        138.906,140.120,140.908,144.240,146.000,150.400,151.960,157.250,
                        158.925,162.500,164.930,167.260,168.934,170.040,174.970,178.490,
                        180.948,183.850,186.200,190.200,192.200,195.090,196.967,200.590,
                        204.370,207.190,208.981,210.000,210.000,222.000,223.000,226.025,
                        227.000,232.038,230.040,238.029,237.048,242.000,242.000,245.000,
                        248.000,252.000,253.000])


    # Energy of a 6 level H atom in erg and corresponding g-values

    eH = np.float64((0.0, 82258.211, 97491.219, 102822.76, 105290.508, 109677.617)) * HH * CC
    gH = np.float64((2.0, 8.0, 18.0, 32.0, 50.0, 1.0))
    # ----------------------------------------------------------------------------------------

    def init_pf_data(self, ifile, to_EV=True):
        #
        # Reads Kurucz's partition functions and ionization potentials from a file
        # Taken from RH (Uitenbroek 2001)
        #

        ff = open(ifile,'rb')
        f = ff.read()
        ff.close()

        data = xdrlib.Unpacker(f)
        npf = data.unpack_uint()

        nelem = 99

        self.tpf = np.float64(data.unpack_farray(npf, data.unpack_double))
        self.el = [None] * nelem

        for ii in range(nelem):
            pti = data.unpack_uint()
            nstage = data.unpack_uint()

            self.el[ii] = dum()

            self.el[ii].pf =  np.float64(data.unpack_farray(npf*nstage, data.unpack_double)).reshape((nstage, npf))
            self.el[ii].eion =  np.float64(data.unpack_farray(nstage, data.unpack_double)) * self.HH*self.CC


            self.el[ii].nstage = nstage
            self.el[ii].npf = pti

            if(to_EV): self.el[ii].eion /= self.EV

    # ----------------------------------------------------------------------------------------

    def acota(self, x, x0, x1):

        if(x < x0): x = x0
        if(x > x1): x = x1

        return x

    # ----------------------------------------------------------------------------------------

    def acotasig(self, x, x0, x1):
        if(x < 0):
            x = -x
            x = self.acota(x, x0, x1)
            x = -x
        else:
            x = self.acota(x, x0, x1)

        return x

    # ----------------------------------------------------------------------------------------

    def sign(self, a, b):
        return abs(a) * (b / abs(b))

    # ----------------------------------------------------------------------------------------

    def __init__(self, abund_init = [], verbose = False, prec = 1.e-5, pf_file='pf_Kurucz.input'):

        self.verbose = verbose
        self.prec = 1.e-5
        self.ncontr = 28 # number of species contributing as electron donnors (sorted)
        self.dtype = 'float64'

        # Replace default abundances ?
        nabund = len(abund_init)

        if(nabund > 0):
            self.ABUND[0:nabund] = abund_init
            if(self.verbose): print('witt::__init__: replacing default abundances with user provided values')

        self.abtot = self.ABUND.sum()
        self.ABUND /= self.abtot
        self.abtot = 1.0
        self.ab_others = self.ABUND[1::].sum() / self.ABUND[0]

        self.avw = (self.ABUND * self.AMASS).sum()

        self.muH = self.avw/ self.AMASS[0] / self.ABUND[0]
        self.rho_from_H = self.muH * self.AMASS[0]*self.AMU / self.BK

        self.avw *=  self.AMU

        # init arrays for later

        self.alfai = np.zeros(self.ncontr, dtype=self.dtype)
        self.chi1  = np.zeros(self.ncontr, dtype=self.dtype)
        self.chi2  = np.zeros(self.ncontr, dtype=self.dtype)
        self.u0    = np.zeros(self.ncontr, dtype=self.dtype)
        self.u1    = np.zeros(self.ncontr, dtype=self.dtype)
        self.u2    = np.zeros(self.ncontr, dtype=self.dtype)

        # Init PF

        DATA_PATH = get_data_path() + pf_file

        self.init_pf_data(DATA_PATH, True)



    # ----------------------------------------------------------------------------------------

    def nsaha(self, t, xne, u0, u1, eion):
        # operates with t, xne and eion in EV
        return 2.0 * self.saha_fac * (u1 / u0) * t**1.5 * exp(-eion*self.EV / (t*self.BK)) / xne

    # ----------------------------------------------------------------------------------------

    def saha(self, theta, eion, u1, u2, pe):
        # Operates with theta, eion in eV and pe
        return u2*exp(2.302585093*(9.0804625434325867-theta*eion))/(u1*pe*theta**2.5)

    # ----------------------------------------------------------------------------------------

    def init_pe_from_pg(self, t, pg):

        # Assume that only H is a electron donnor

        nu = self.ABUND[0]
        saha = 10.0**( -0.4771+2.5*log10(t)-log10(pg)-(13.6*5040.0/t))
        aaa = 1.0 + saha
        bbb = -(nu-1.0) * saha
        ccc = -saha*nu
        ybh=(-bbb+sqrt(bbb*bbb-4.*aaa*ccc))/(2.*aaa)
        pe=pg*ybh/(1.+ybh)

        return pe

    # ----------------------------------------------------------------------------------------

    def pe_from_pg(self, t, pg, get_fe = False):

        dif = 1.1
        pe = self.init_pe_from_pg(t, pg) # Init Pe assuming only H and increase it 10%
        ope = pe
        it = 0

        while((abs(dif) > self.prec) and (it < 250)):
            pe = (ope + pe)*0.5
            ope = pe

            pe,fe = self.pe_pg(t, pe, pg, get_fe=True)
            dif = 2.0 * abs(pe-ope) / (pe+ope)
            it += 1

        if(self.verbose): print("witt::pe_from_pg: convergence niter = {0}".format(it-1))

        if(get_fe): return pe, fe
        return pe

    # ----------------------------------------------------------------------------------------

    def pe_from_rho(self, t, rho):

        # fraction of atoms

        xna = rho / self.avw
        BKT = self.BK * t

        # now estimate Pgas and iterate

        if(t > 8000): a = 0.5;
        elif(t > 4000): a = 0.1;
        elif(t > 2000): a = 0.01;
        else: a = 0.001;

        xne = a * xna /(1.0 - a)
        Pgas = (xna+xne)*BKT


        # Iterate

        it = 0
        dif = 1.0

        while((it < 250) and (abs(dif) > self.prec)):
            oPgas = Pgas
            Pe = self.pe_from_pg(t, Pgas)
            xna_guessed = (Pgas-Pe)/BKT
            dif = abs(xna-xna_guessed)/xna

            Pgas  *= xna/xna_guessed

        return Pe#, Pgas

    # ----------------------------------------------------------------------------------------

    def pg_from_rho(self, temp, rho):

        xna = (rho / self.avw)

        if(temp > 8000): a = 0.5;
        elif(temp > 4000): a = 0.1;
        elif(temp > 2000): a = 0.01;
        else: a = 0.001;

        xne = a * xna /(1.0 - a)
        pgas = (xna+xne)*self.BK*temp

        Pe = self.pe_from_pg(temp, pgas)
        irho = self.rho_from_pe(temp, Pe)

        dif = 1.0
        it = 0
        while((dif >= self.prec) and (it<100)):
            Pe *= (1.0 + rho / irho)*0.5
            irho = self.rho_from_pe(temp, Pe)
            dif = np.abs((irho - rho) / (rho))
            it+=1

        pgas = self.pg_from_pe(temp, Pe)

        return pgas

    # ----------------------------------------------------------------------------------------

    def rho_from_pe(self,temp, pe):

        pg, fe_out = self.pg_from_pe(temp, pe, True)
        rho = pe * self.rho_from_H / (fe_out * temp)
        return rho

    # ----------------------------------------------------------------------------------------

    def rho_from_pg(self,temp, pg):

        pe, fe_out = self.pe_from_pg(temp, pg, True)
        rho = pe * self.rho_from_H / (fe_out * temp)

        return rho
    # ----------------------------------------------------------------------------------------


    def molecb(self, X):
        Y = np.empty(2, dtype=self.dtype); dy = np.empty(2, dtype=self.dtype)
        Y[0]=-11.206998+X*(2.7942767+X*(7.9196803E-2-X*2.4790744E-2)) # H2
        Y[1]=-12.533505+X*(4.9251644+X*(-5.6191273E-2+X*3.2687661E-3)) # H
        dx=(-X*X)/5040.

        dy[0]=dx*(2.7942767+X*(2*7.9196803e-2-X*3*2.4790744e-2))
        dy[1]=dx*(4.9251644+X*(-2*5.6191273e-2-X*3*3.2687661e-3))

        return Y, dy

    # ----------------------------------------------------------------------------------------

    def pe_pg(self, t, pe, pgas, get_fe=False):

        g1 = 0.; theta = 5040.0/t


        # Init some values

        if(pe < 0.0):
            pe = 1.e-15
            g4 = 0.0
            g5 = 0.0
        else:
            cmol, dcmol = self.molecb(theta)
            cmol[0] = self.acota(cmol[0], -30., 30.)
            cmol[1] = self.acota(cmol[1], -30., 30.)

            g4 = pe * 10.0**cmol[0]
            g5 = pe * 10.0**cmol[1]

        #
        # H first
        #
        u = self.partition_f(0, t, only=3)
        g2 = self.saha(theta, self.el[0].eion[0], u[0], u[1], pe) # p(h+)/p(h)
        g3 = self.saha(theta, 0.754, 1.0, u[0], pe)        # p(h)/p(h-)
        g3 = 1.0 / self.acota(g3, 1.e-30, 1.0e30)


        #
        # Now count electrons contributed by the first
        # two ionized stages of all contributing atoms
        #
        for ii in range(1, self.ncontr):
            alfai = self.ABUND[ii] / self.ABUND[0] # relative to H abundance
            u = self.partition_f(ii, t, only=3)

            a = self.saha(theta, self.el[ii].eion[0], u[0], u[1], pe)
            b = self.saha(theta, self.el[ii].eion[1], u[1], u[2], pe)

            c = 1.+a*(1.+b)
            g1 += alfai/c*a*(1.+2.*b)


        # All the math...

        a=1.+g2+g3
        b=2.*(1.+g2/g5*g4)
        c=g5
        d=g2-g3
        e=g2/g5*g4

        a = self.acotasig(a, 1.e-15, 1.e15);
        d = self.acotasig(d, 1.e-15, 1.e15);

        c1=c*b*b+a*d*b-e*a*a
        c2=2.0*a*e-d*b+a*b*g1
        c3=-(e+b*g1)
        f1=0.5*c2/c1
        f1=-f1+self.sign(1.,c1)*sqrt(f1*f1-c3/c1)
        f5=(1.-a*f1)/b
        f4=e*f5
        f3=g3*f1
        f2=g2*f1
        fe = self.acota(f2-f3+f4+g1, 1.e-30, 1.e30)
        phtot = pe/fe

        # Refinement by Wittmann, not sure this is needed in
        # double prec.

        if(f5 <= 1.e-4):
            diff = 1.0; const6=g5/pe*f1*f1; const7=f2-f3+g1; it = 0

            while((diff > 1.e-5) and (it < 5)):
                of5 = f5
                f5=phtot*const6
                f4=e*f5
                fe=const7+f4
                phtot=pe/fe
                diff = 0.5 * fabs(f5-of5) / (f5 + of5)
                it += 1

        # Recompute pe

        pe = pgas / (1.+(f1+f2+f3+f4+f5+self.ab_others)/fe)

        if(pe <= 0.0):
            pe = 1.e-15

        if(get_fe): return pe, fe
        return pe


    # ----------------------------------------------------------------------------------------

    def Boltzmann(self, t, u, glow, e_pot):
        # Gives the ratio between the total population of a ionization stage and a given level
        # The energy levels must be in Erg.
        return (glow / u) * exp(-(e_pot) / (self.BK*t))

    # ----------------------------------------------------------------------------------------

    def getH6pop(self, t, pgas, pe):


        # Solve ionization for H

        n, u = self.getXparts(0, t, pgas, pe, divide_by_u = False, return_u = True)


        # Define output array

        res = np.empty(6, dtype='float64')
        res[-1] = n[1] # number of protons

        ratios = np.empty(5, dtype='float64')

        # Solve level populations for the 5 first levels of H

        for ii in range(5):
            ratios[ii] = self.Boltzmann(t, u[0], self.gH[ii], self.eH[ii])


        # particle conservation (with 6 levels in H this line has no effect)

        #ratios /= ratios.sum()


        # Multiply the neutral H population by the ratios of level populations

        res[0:5] = n[0] * ratios


        return res


    # ----------------------------------------------------------------------------------------

    def _itep1(self, x, y, xx):
        """
        Linear interpolation routine
        input:
             x: array of input x values
             y: array of input y values
             xx: scalar x value where the interpolated value of "y must be calculated
        """

        if(xx <= x[0]): return y[0]
        elif(xx >= x[-1]): return y[-1]
        else:
            idx=np.where(x>xx)[0]
            p0 = idx[0]
            p1 = p0-1

            dx = x[p1]-x[p0]
            u1 = (xx-x[p0]) / dx
            u0 = 1.0 - u1

            return u0 * y[p0] + u1 * y[p1]


    # ----------------------------------------------------------------------------------------

    def partition_f(self, n, t, only=0):
        """
        Computes the partition function of element "n" for a given temperature

        input:
             n: Atomic number of the atom -1 (H is 0).
             t: Temperature in K.
        Keywords:
             only: (=int) only return this number first levels of the atom.
                   The default is to return all the levels


        """
        nn = self.el[n].nstage

        if(only > 0):
            nn = min(nn, only)
            res = np.zeros(only, dtype='float64')
        else:
            res = np.zeros(nn, dtype='float64')

        for ii in range(nn):
            res[ii] = self._itep1( self.tpf, self.el[n].pf[ii], t)

        return res

    # ----------------------------------------------------------------------------------------

    def pg_from_pe(self, t, pe, get_fe=False):

        pg, dum = self.gasc(t, pe)
        if(get_fe): return pg, dum[-1]

        return pg

    # ----------------------------------------------------------------------------------------

    def gasc(self, t, pe):

        pp = np.zeros(self.ncontr+6, dtype = self.dtype)

        theta = 5040. / t
        cmol, dmol = self.molecb(theta)

        g4 = 10.0**cmol[0]; g5 = 10.0**cmol[1]


        # First H

        u = self.partition_f(0, t)
        g2 = self.saha(theta, self.el[0].eion[0], u[0], u[1], pe) # p(h+)/p(h)
        g3 = 1.0 / self.saha(theta, 0.754, 1.0, u[0], pe)        # p(h)/p(h-)
        g1 = 0.0


        # Other contributions

        for ii in range(1, self.ncontr):
            alfai = self.ABUND[ii] / self.ABUND[0] # relative to H abundance
            u = self.partition_f(ii, t, only=3)

            a = self.saha(theta, self.el[ii].eion[0], u[0], u[1], pe)
            b = self.saha(theta, self.el[ii].eion[1], u[1], u[2], pe)

            c=1.+a*(1.+b) # fraction of n0/ntot

            pp[ii]=alfai/c # abund /ntot (partial pressure of neutral species ii)

            # 1*n1 + 2*n2 + ... j*n_j so we count how many electrons
            # come from each ionized species

            g1 += pp[ii]*a*(1.+2.*b)


        a=1.+g2+g3
        e=g2/g5*g4
        b=2.0*(1.0 + e)
        c=g5
        d=g2-g3
        c1=c*b*b+a*d*b-e*a*a
        c2=2.*a*e-d*b+a*b*g1
        c3=-(e+b*g1)
        f1=0.5*c2/c1
        f1=-f1+self.sign(1.0,c1)*sqrt(f1*f1-c3/c1)
        f5=(1.0-a*f1)/b

        f4=e*f5
        f3=g3*f1
        f2=g2*f1
        fe=f2-f3+f4+g1
        phtot=pe/fe

        # Refinement by Wittmann
        if(f5 <= 1.e-5):
            diff = 1.0; const6=g5/pe*f1*f1; const7=f2-f3+g1; it = 0

            while((diff > 1.e-5) and (it < 5)):
                of5 = f5
                f5=phtot*const6
                f4=e*f5
                fe=const7+f4
                phtot=pe/fe
                diff = 0.5 * fabs(f5-of5) / (f5 + of5)
                it += 1

        pg=pe*(1.0+(f1+f2+f3+f4+f5+self.ab_others)/fe)

        # Store the partial pressures of H too

        pp[self.ncontr+0] = f1    #  p(h)/p(h')
        pp[self.ncontr+1] = f2    #  p(h+)/p(h')
        pp[self.ncontr+2] = f5    #  p(h2)/p(h')
        pp[self.ncontr+3] = f3    #  p(h-)/p(h')
        pp[self.ncontr+4] = phtot #  p(h') (total hydrogen!)
        pp[self.ncontr+5] = fe #  p(h') (total hydrogen!)


        return pg, pp

    # ----------------------------------------------------------------------------------------

    def getXparts(self, iatom, t, pg, pe, divide_by_u = False, only = 0, return_u = False):

        # Precompute partial densities of atoms, electrons
        # and partial density of iatom with the abundance

        TBK = t * self.BK; xna = (pg-pe) / TBK; xne = pe / TBK
        n_tot = xna * self.ABUND[iatom] / self.abtot



        # Partition function

        u = self.partition_f(iatom, t, only=only)
        nLev = u.size


        # Solve saha and get the partial density of each ionized stage

        xpa = np.empty(nLev, dtype='float64')
        xpa[0] = 1.0

        for ii in range(1, nLev):
            xpa[ii] = self.nsaha(t, xne, u[ii-1], u[ii], self.el[iatom].eion[ii-1])

        for ii in range(nLev-1, 0, -1):
            xpa[0] = 1.0 + xpa[0]*xpa[ii]

        xpa[0] = 1.0 / xpa[0]

        for ii in range(1, nLev):
            xpa[ii] *= xpa[ii-1]


        # Now that we have the ratios between ionized stages, multiply by the partial
        # density of iatom. Divide by the partition function if needed

        if(divide_by_u):
            xpa[:] *= n_tot / u[:]
        else:
            xpa *= n_tot

        if(return_u): return xpa, u
        else: return xpa

    # ----------------------------------------------------------------------------------------

    def getBackgroundPartials(self, t, pg, pe, divide_by_u = True):

        n = np.empty(17, dtype='float64')
        tbk = t * self.BK

        # --- He/He+/He++ ---

        xpa = self.getXparts(1, t, pg, pe, divide_by_u = divide_by_u)
        n[3] = xpa[0]; n[4] = xpa[1]; n[5] = xpa[2]

        # --- C ---

        xpa = self.getXparts(5, t, pg, pe, divide_by_u = divide_by_u)
        n[6] = xpa[0]


        # --- Al ---

        xpa = self.getXparts(12, t, pg, pe, divide_by_u = divide_by_u)
        n[7] = xpa[0]


        # --- Si/Si+ ---

        xpa = self.getXparts(13, t, pg, pe, divide_by_u = divide_by_u)
        n[8] = xpa[0]; n[9] = xpa[1]


        # --- Ca/Ca+ ---

        xpa = self.getXparts(19, t, pg, pe, divide_by_u = divide_by_u)
        n[10] = xpa[0]; n[11] = xpa[1]


        # --- Mg/Mg+ ---

        xpa = self.getXparts(11, t, pg, pe, divide_by_u = divide_by_u)
        n[12] = xpa[0]; n[13] = xpa[1]


        # --- Fe ---

        xpa = self.getXparts(25, t, pg, pe, divide_by_u = divide_by_u)
        n[14] = xpa[0]


        # --- N ---

        xpa = self.getXparts(6, t, pg, pe, divide_by_u = divide_by_u)
        n[15] = xpa[0]


        # --- O ---

        xpa = self.getXparts(7, t, pg, pe, divide_by_u = divide_by_u)
        n[16] = xpa[0]


        # Get H- and other H densities

        if(divide_by_u): pfH = 0.5
        else: pfH = 1.0

        dum, pp = self.gasc(t, pe)
        n[0] = pp[self.ncontr+0]*pp[self.ncontr+4] / tbk * pfH # H / pf[H]
        n[1] = pp[self.ncontr+1]*pp[self.ncontr+4] / tbk       # H+/ 1.0
        n[2] = pp[self.ncontr+3]*pp[self.ncontr+4] / tbk       # H-/ 1.0


        return n

    # ----------------------------------------------------------------------------------------

    def contOpacity(self, iT, Pgas, Pe, w):

        # Some definitions

        TK = iT * self.BK; TKEV = TK / self.EV
        HTK = self.HH/TK; TLOG = log(iT);
        xna = (Pgas-Pe) / TK; xne = Pe / TK


        # Partial densities of background absorvers,
        # divided by the partition functions

        n = self.getBackgroundPartials(iT, Pgas, Pe, divide_by_u = True)


        # get background opacity

        opac, scat = cop(iT,  TKEV, TK, HTK, TLOG, xna, xne, w, n[0], n[1],
                              n[2], n[3],n[4], n[5], n[6], n[7], n[8], n[9], n[10], n[11],
                              n[12], n[13], n[14], n[15], n[16])

        return opac




# ----------------------------------------------------------------------------------------
#
# All the following functions are used to compute background opacities.
# The main call is done to function "cop", that calls all the other auxiliary
# functions.
#
# ----------------------------------------------------------------------------------------

@njit(cache=True)
def SEATON(FREQ0, XSECT, POWER, A, FREQ):
    return XSECT*(A+(1.-A)*(FREQ0/FREQ))* (FREQ0/FREQ)**(floor(2.*POWER+0.01)*0.5)

# ----------------------------------------------------------------------------------------

Z4LOG=np.float64((0.,1.20412,1.90849,2.40824,2.79588,3.11261))
A0 = np.float64((
    5.53,5.49,5.46,5.43,5.40,5.25,5.00,4.69,4.48,4.16,3.85,
    4.91,4.87,4.84,4.80,4.77,4.63,4.40,4.13,3.87,3.52,3.27,
    4.29,4.25,4.22,4.18,4.15,4.02,3.80,3.57,3.27,2.98,2.70,
    3.64,3.61,3.59,3.56,3.54,3.41,3.22,2.97,2.70,2.45,2.20,
    3.00,2.98,2.97,2.95,2.94,2.81,2.65,2.44,2.21,2.01,1.81,
    2.41,2.41,2.41,2.41,2.41,2.32,2.19,2.02,1.84,1.67,1.50,
    1.87,1.89,1.91,1.93,1.95,1.90,1.80,1.68,1.52,1.41,1.30,
    1.33,1.39,1.44,1.49,1.55,1.56,1.51,1.42,1.33,1.25,1.17,
    0.90,0.95,1.00,1.08,1.17,1.30,1.32,1.30,1.20,1.15,1.11,
    0.55,0.58,0.62,0.70,0.85,1.01,1.15,1.18,1.15,1.11,1.08,
    0.33,0.36,0.39,0.46,0.59,0.76,0.97,1.09,1.13,1.10,1.08,
    0.19,0.21,0.24,0.28,0.38,0.53,0.76,0.96,1.08,1.09,1.09)).reshape((12,11))


@njit(cache=True)
def COULFF(TLOG, FREQLG, NZ):
    GAMLOG=10.39638-TLOG/1.15129+Z4LOG[NZ-1]
    IGAM=min(int(GAMLOG+7.),10)
    if(IGAM<1): IGAM=1

    #  HVKTLG=2*log10(HVKT) #

    HVKTLG=(FREQLG-TLOG)/1.15129-20.63764
    IHVKT=min(int(HVKTLG+9.),11)
    if(IHVKT<1): IHVKT=1

    P=GAMLOG-(IGAM-7);
    Q=HVKTLG-(IHVKT-9);
    return (1.-P)*((1.-Q)*A0[IHVKT-1,IGAM-1]+Q*A0[IHVKT, IGAM-1])+P*((1.-Q)*A0[IHVKT-1,IGAM]+Q*A0[IHVKT,IGAM])


# ----------------------------------------------------------------------------------------

A1 = np.float64((0.9916,1.105,1.101,1.101,1.102,1.0986))
B1 = np.float64((2.719e3,-2.375e4,-9.863e3,-5.765e3,-3.909e3,-2.704e3))
C1 = np.float64((-2.268e10,4.077e8,1.035e8,4.593e7,2.371e7,1.229e7))

@njit(cache=True)
def COULX(N, freq, Z):
    n=(N+1.0)**2

    if(freq>=(Z*Z*3.28805e15/n)):
        FREQ1=freq*1.e-10
        CLX=0.2815/FREQ1/FREQ1/FREQ1/n/n/(N+1.0)*Z*Z*Z*Z

        if(N>=6):
            return CLX

        CLX*=(A1[N]+(B1[N]+C1[N]*(Z*Z/FREQ1))*(Z*Z/FREQ1))
        return CLX

    return 0.0

# ----------------------------------------------------------------------------------------

@njit(cache=True)
def HOP( XNE,  XH1,  XH2,  FREQ,  FREQLG, T,  TLOG,  TKEV,  STIM,  EHVKT):

    CONT = np.empty(8)
    BOLT = np.empty(8)

    FREQ3=(FREQ*1.E-10)**3
    CFREE=3.6919E-22/FREQ3

    n1 = (np.arange(8)+1.0)**2
    BOLT = np.exp(-13.595*(1.-1./n1)/TKEV)*2.*n1*XH1


    FREET=XNE*CFREE*XH2/sqrt(T)
    XR=XH1/13.595*TKEV
    BOLTEX=exp(-13.427/TKEV)*XR
    EXLIM=exp(-13.595/TKEV)*XR

    for N in range(8):
        CONT[N]=COULX(N,FREQ,1.0)

    C=0.2815/FREQ3;
    if(FREQ < 4.05933E13):
        BOLTEX=EXLIM/EHVKT
    H=(CONT[6]*BOLT[6]+CONT[7]*BOLT[7]+(BOLTEX-EXLIM)*C+
       COULFF(TLOG,FREQLG,1)*FREET)*STIM


    H += (CONT[0:6]*BOLT[0:6]).sum()*(1.-EHVKT)

    return H

# ----------------------------------------------------------------------------------------

@njit(cache=True)
def HRAYOP(XH1,  FREQ):

    WAVE=min(FREQ,2.463e15)
    WAVE=2.997925e18/WAVE
    WW=WAVE*WAVE
    WW2 = WW*WW
    SIG=(5.799e-13+1.422e-6/WW+2.784/(WW2))/(WW2)

    return SIG*XH1*2.0

# ----------------------------------------------------------------------------------------

@njit(cache=True)
def H2PLOP(XH1, XH2, FREQ, FREQLG, FREQ15, TKEV, STIM):

  if(FREQ > 3.28805E15):
      return 0.0

  FR=-3.0233E3+(3.7797E2+(-1.82496E1+(3.9207E-1-3.1672E-3*FREQLG)*
			  FREQLG)*FREQLG)*FREQLG
  ES=-7.342E-3+(-2.409+(1.028+(-0.4230+(0.1224-0.01351*FREQ15)*
			       FREQ15)*FREQ15)*FREQ15)*FREQ15
  return exp(-ES/TKEV+FR)*2.*XH1*XH2*STIM


# ----------------------------------------------------------------------------------------

@njit(cache=True)
def HMINOP( XH1,  XHMIN,  FREQ,  T, TKEV,  XNE,  EHVKT):

    FREQ1=FREQ*1.E-10
    B=(1.3727E-15+4.3748/FREQ)/FREQ1
    C=-2.5993E-7/FREQ1**2

    if(FREQ <= 1.8259E14): HMINBF=0.
    elif(FREQ >= 2.111E14): HMINBF=6.801E-10+(5.358E-3+(1.481E3+(-5.519E7+4.808E11/FREQ1)/FREQ1)/FREQ1)/FREQ1
    else: HMINBF=3.695E-6+(-1.251E-1+1.052E3/FREQ1)/FREQ1

    HMINFF=(B+C/T)*XH1*XNE*2.E-20


    #
    # We use the number density / partition function for H-.
    # The partition function for H- is 1
    #
    if(T < 7730.): HMIN=XHMIN
    else: HMIN=exp(0.7552/TKEV)/(2.*2.4148E15*T*sqrt(T))*XH1*XNE

    H=HMINBF*(1-EHVKT)*HMIN*1.E-10
    return H+HMINFF

# ----------------------------------------------------------------------------------------

G0 = np.float64((1.,3.,1.,9.,3.,3.,1.,9.,20.,3.))
HEFREQ0 = np.float64((5.9452090e15,1.1528440e15,0.9803331e15,.8761076e15,
		      0.8147100e15,0.4519048e15,0.4030971e15,.8321191e15,
		      0.3660215e15,0.3627891e15))
CHI0 = np.float64((0.,19.819,20.615,20.964,21.217,22.718,22.920,23.006,
		   23.073,23.086))

@njit(cache=True)
def HE1OP( XHE1,  XHE2,  XNE,  FREQ,  FREQLG, T,  TKEV,  TLOG,  EHVKT,  STIM):

    TRANS = np.zeros(10)
    BOLT=np.exp(-CHI0/TKEV)*G0*XHE1

    FREET=XNE*1.E-10*XHE2*1.E-10/sqrt(T)*1.E-10
    XRLOG=log(XHE1*(2./13.595)*TKEV)
    BOLTEX=exp(-23.730/TKEV+XRLOG)
    EXLIM=exp(-24.587/TKEV+XRLOG)
    FREQ3=(FREQ*1.E-10)**3
    CFREE=3.6919E8/FREQ3
    C=2.815E-1/FREQ3

    for NMIN in range(10):
        if(HEFREQ0[NMIN] <= FREQ): break

    dum = np.float64((33.32-2.*FREQLG, -390.026+(21.035-0.318*FREQLG)*FREQLG, 26.83-1.91*FREQLG, 61.21-2.9*FREQLG, 81.35-3.5*FREQLG, 12.69-1.54*FREQLG, 23.85-1.86*FREQLG, 49.30-2.60*FREQLG,85.20-3.69*FREQLG, 58.81-2.89*FREQLG ))

    TRANS[NMIN::] = np.exp(dum[NMIN::])


    EX = BOLTEX
    if(FREQ < 2.055E14): EX=EXLIM/EHVKT

    HE1=(EX-EXLIM)*C;
    HE1 += (TRANS*BOLT).sum()

    return (HE1+COULFF(TLOG,FREQLG,1)*FREET*CFREE)*STIM

# ----------------------------------------------------------------------------------------

@njit(cache=True)
def HE2OP(  XHE2,  XHE3,  XNE,  FREQ,  FREQLG, T,  TKEV,  TLOG,  EHVKT,  STIM):

    #
    #   REQUIRES FUNCTIONS COULX AND COULFF
    #   FREQUENCIES ARE 4X HYDROGEN,CHI ARE FOR ION POT=54.403
    #
    CONT = np.empty(9)

    N12 = (np.arange(9)+1.0)**2
    BOLT=np.exp(-(54.403-54.403/N12)/TKEV)*2.*N12*XHE2

    FREET=XNE*XHE3/sqrt(T)
    XR=XHE2/13.595*TKEV
    BOLTEX=exp(-53.859/TKEV)*XR
    EXLIM=exp(-54.403/TKEV)*XR

    for N in range(9): CONT[N]=COULX(N,FREQ,2.0)

    FREQ3=(FREQ*1.E-5)**3
    CFREE=3.6919E-07/FREQ3*4.0
    C=2.815E14*2.0*2.0/FREQ3
    EX=BOLTEX

    if(FREQ < 1.31522E14): EX=EXLIM/EHVKT
    HE2=(EX-EXLIM)*C

    HE2 += (CONT*BOLT).sum()
    HE2=(HE2+COULFF(TLOG,FREQLG,2)*CFREE*FREET)*STIM

    if(HE2 >= 1.E-20): return HE2
    else:              return 0.0


# ----------------------------------------------------------------------------------------

@njit(cache=True)
def HEMIOP( XHE1,  FREQ,  T,  XNE):
    A= 3.397E-26+(-5.216E-11+7.039E05/FREQ)/FREQ;
    B=-4.116E-22+( 1.067E-06+8.135E09/FREQ)/FREQ;
    C= 5.081E-17+(-8.724E-03-5.659E12/FREQ)/FREQ;
    return (A*T+B+C/T)*XNE*XHE1*1.E-20;

# ----------------------------------------------------------------------------------------

@njit(cache=True)
def HERAOP(XHE1, FREQ):
    WW=(2.997925E+03/min(FREQ*1.E-15,5.15))**2
    arg = 1.+(2.44E5+5.94E10/(WW-2.90E5))/WW
    SIG=5.484E-14/WW/WW*arg*arg
    return SIG*XHE1

# ----------------------------------------------------------------------------------------

PEACH0 = np.float64((-42.474, -42.350, -42.109, -41.795, -41.467, -41.159, -40.883,#,/*  1500 */
                     -41.808, -41.735, -41.582, -41.363, -41.115, -40.866, -40.631,#,/*  1550 */
                     -41.273, -41.223, -41.114, -40.951, -40.755, -40.549, -40.347,#,/*  1621 */
                     -45.583, -44.008, -42.957, -42.205, -41.639, -41.198, -40.841,#,/*  1622 */
                     -44.324, -42.747, -41.694, -40.939, -40.370, -39.925, -39.566,#,/*  2513 */
                     -50.969, -48.388, -46.630, -45.344, -44.355, -43.568, -42.924,#,/*  2514 */
                     -50.633, -48.026, -46.220, -44.859, -43.803, -42.957, -42.264,#,/*  3756 */
                     -53.028, -49.643, -47.367, -45.729, -44.491, -43.520, -42.736,#,/*  3757 */
                     -51.785, -48.352, -46.050, -44.393, -43.140, -42.157, -41.363,#,/*  6549 */
                     -52.285, -48.797, -46.453, -44.765, -43.486, -42.480, -41.668,#,/*  6550 */
                     -52.028, -48.540, -46.196, -44.507, -43.227, -42.222, -41.408,#,/*  7234 */
                     -52.384, -48.876, -46.513, -44.806, -43.509, -42.488, -41.660,#,/*  7235 */
                     -52.363, -48.856, -46.493, -44.786, -43.489, -42.467, -41.639,#,/*  7291 */
                     -54.704, -50.772, -48.107, -46.176, -44.707, -43.549, -42.611,#,/*  7292 */
                     -54.359, -50.349, -47.643, -45.685, -44.198, -43.027, -42.418)).reshape((15,7))
FREQMG = np.float64((1.9341452e15,1.8488510e15,1.1925797e15, 7.9804046e14,4.5772110e14,4.1440977e14,
                     4.1113514e14))
FLOG0 = np.float64((35.32123,35.19844,35.15334,34.71490,34.31318, 33.75728,33.65788,33.64994,33.43947))
TLG0=np.float64((8.29405,8.51719,8.69951,8.85367, 8.98720,9.10498,9.21034))

@njit(cache=True)
def Mg1OP(FREQ,  FREQLG,  T,  TLOG):
    NT=min(6,int(floor(T/1000.))-3)
    if(NT<1): NT=1

    DT=(TLOG-TLG0[NT-1])/(TLG0[NT]-TLG0[NT-1])
    for N in range(7):
        if(FREQ > FREQMG[N]): break

    D=(FREQLG-FLOG0[N])/(FLOG0[N+1]-FLOG0[N])
    if(N > 1): N=2*N-1
    D1=1.0-D
    XWL1=PEACH0[N+1,NT-1]*D + PEACH0[N,NT-1]*D1
    XWL2=PEACH0[N+1,NT  ]*D + PEACH0[N,NT  ]*D1

    return exp(XWL1*(1.-DT)+XWL2*DT)

# ----------------------------------------------------------------------------------------

@njit(cache=True)
def C1OP( FREQ,  TKEV):

  # CROSS-SECTION TIMES THE PARTITION FUNCTION
  C1240=5.*exp(-1.264/TKEV)
  C1444=exp(-2.683/TKEV)
  X1444=0.0
  X1240=0.0
  X1100=0.0

  if(FREQ >= 2.7254E15): X1100=SEATON(2.7254E15,1.219E-17,2.0E0,3.317E0,FREQ)
  if(FREQ >= 2.4196E15): X1240=SEATON(2.4196E15,1.030E-17,1.5E0,2.789E0,FREQ)
  if(FREQ >= 2.0761E15): X1444=SEATON(2.0761E15,9.590E-18,1.5E0,3.501E0,FREQ)

  return X1100*9.+X1240*C1240+X1444*C1444

# ----------------------------------------------------------------------------------------

@njit(cache=True)
def Al1OP(FREQ):
    if(FREQ > 1.443E15): return 2.1E-17*((1.443E15/FREQ)**3)*6.0
    else: return 0.0

# ----------------------------------------------------------------------------------------

PEACH1 = np.float64(( 38.136,38.138,38.140,38.141,38.143,38.144,38.144,38.145,38.145,#/* 1200 */
                      37.834,37.839,37.843,37.847,37.850,37.853,37.855,37.857,37.858,#/* 1400 */
                      37.898,37.898,37.897,37.897,37.897,37.896,37.895,37.895,37.894,#/* 1519 */
                      40.737,40.319,40.047,39.855,39.714,39.604,39.517,39.445,39.385,#/* 1520 */
                      40.581,40.164,39.893,39.702,39.561,39.452,39.366,39.295,39.235,#/* 1676 */
                      45.521,44.456,43.753,43.254,42.878,42.580,42.332,42.119,41.930,#/* 1677 */
                      45.520,44.455,43.752,43.251,42.871,42.569,42.315,42.094,41.896,#/* 1978 */
                      55.068,51.783,49.553,47.942,46.723,45.768,44.997,44.360,43.823,#/* 1979 */
                      53.868,50.369,48.031,46.355,45.092,44.104,43.308,42.652,42.100,#/* 5379 */
                      54.133,50.597,48.233,46.539,45.261,44.262,43.456,42.790,42.230,#/* 5380 */
                      54.051,50.514,48.150,46.454,45.176,44.175,43.368,42.702,42.141,#/* 5624 */
                      54.442,50.854,48.455,46.733,45.433,44.415,43.592,42.912,42.340,#/* 5625 */
                      54.320,50.722,48.313,46.583,45.277,44.251,43.423,42.738,42.160,#/* 6260 */
                      55.691,51.965,49.444,47.615,46.221,45.119,44.223,43.478,42.848,#/* 6261 */
                      55.661,51.933,49.412,47.582,46.188,45.085,44.189,43.445,42.813,#/* 6349 */
                      55.973,52.193,49.630,47.769,46.349,45.226,44.314,43.555,42.913,#/* 6350 */
                      55.922,52.141,49.577,47.715,46.295,45.172,44.259,43.500,42.858,#/* 6491 */
                      56.828,52.821,50.110,48.146,46.654,45.477,44.522,43.730,43.061,#/* 6492 */
                      56.657,52.653,49.944,47.983,46.491,45.315,44.360,43.569,42.901)).reshape((19,9))
FREQSI1 = np.float64((2.1413750e15,1.97231650e15,1.7879689e15,1.5152920e15,0.55723927e15,5.3295914e14,4.7886458e14,4.72164220e14,4.6185133e14))
FLOG1 = np.float64((35.45438,35.30022,35.21799,35.11986,34.95438, 33.95402,33.90947,33.80244,33.78835,33.76626, 33.70518))
TLG1 = np.float64((8.29405,8.51719,8.69951,8.85367,8.98720,9.10498,9.21034,9.30565,9.39266))

@njit(cache=True)
def Si1OP(FREQ, FREQLG, T, TLOG):
    NT=min(8,int(floor(T/1000.))-3)
    if(NT<1): NT=1
    DT=(TLOG-TLG1[NT-1])/(TLG1[NT]-TLG1[NT-1])

    for N in range(9):
        if(FREQ > FREQSI1[N]): break

    D=(FREQLG-FLOG1[N])/(FLOG1[N+1]-FLOG1[N])

    if(N>1): N=2*N-1
    DD=1.-D
    XWL1=PEACH1[N+1,NT-1]*D+PEACH1[N,NT-1]*DD
    XWL2=PEACH1[N+1,NT  ]*D+PEACH1[N,NT  ]*DD

    return exp(-(XWL1*(1.-DT)+XWL2*DT))*9.

# ----------------------------------------------------------------------------------------


G1 = np.float64((25.,35.,21.,15., 9.,35.,33.,21.,27.,49., 9.,21.,
                27., 9., 9.,25.,33.,15.,35., 3., 5.,11.,15.,13.,
                15., 9.,21.,15.,21.,25.,35., 9., 5.,45.,27.,21.,
                 15.,21.,15.,25.,21.,35., 5.,15.,45.,35.,55.,25.))
E1 = np.float64((500., 7500.,12500.,17500.,19000.,19500.,19500.,
		21000.,22000.,23000.,23000.,24000.,24000.,24500.,
		24500.,26000.,26500.,26500.,27000.,27500.,28500.,
		29000.,29500.,29500.,29500.,30000.,31500.,31500.,
		33500.,33500.,34000.,34500.,34500.,35000.,35500.,
		37000.,37000.,37000.,38500.,40000.,40000.,41000.,
		 41000.,43000.,43000.,43000.,43000.,44000.))
WNO1 = np.float64((63500.,58500.,53500.,59500.,45000.,44500.,44500.,
                  43000.,58000.,41000.,54000.,40000.,40000.,57500.,
                  55500.,38000.,57500.,57500.,37000.,54500.,53500.,
                  55000.,34500.,34500.,34500.,34000.,32500.,32500.,
                  32500.,32500.,32000.,29500.,29500.,31000.,30500.,
                  29000.,27000.,54000.,27500.,24000.,47000.,23000.,
                   44000.,42000.,42000.,21000.,42000.,42000.))

@njit(cache=True)
def Fe1OP(FREQ, HKT):
    WAVENO=FREQ/2.99792458E10
    if(WAVENO < 21000.): return 0.0
    BOLT=G1*np.exp(-E1*2.99792458e10*HKT);
    XXX=((WNO1+3000.-WAVENO)/WNO1/.1)


    XSECT = np.zeros(48)
    idx = np.where(WNO1 <WAVENO)[0]
    XSECT[idx] = 3.e-18/(1.+XXX[idx]**4)

    return (XSECT*BOLT).sum()

# ----------------------------------------------------------------------------------------

@njit(cache=True)
def COOLOP(XC1,XMg1,XAl1,XSi1,XFe1,STIM,FREQ,FREQLG,T,TLOG,TKEV,HKT):
    #     Si I, Mg I, Al I, C I, Fe I

    return ( C1OP(FREQ,TKEV          )*XC1 +
	     Mg1OP(FREQ,FREQLG,T,TLOG)*XMg1+
	     Al1OP(FREQ              )*XAl1+
	     Si1OP(FREQ,FREQLG,T,TLOG)*XSi1+
             Fe1OP(FREQ,HKT          )*XFe1)*STIM

# ----------------------------------------------------------------------------------------

@njit(cache=True)
def N1OP( FREQ,  TKEV):

    # CROSS-SECTION TIMES PARTITION FUNCTION

  C1130=6.*exp(-3.575/TKEV)
  C1020=10.*exp(-2.384/TKEV)
  X1130=0.
  X1020=0.
  X853=0.

  if(FREQ >= 3.517915E15): X853 =SEATON(3.517915E15,1.142E-17,2.0E0,4.29E0,FREQ)
  if(FREQ >= 2.941534E15): X1020=SEATON(2.941534E15,4.410E-18,1.5E0,3.85E0,FREQ)
  if(FREQ >= 2.653317E15): X1130=SEATON(2.653317E15,4.200E-18,1.5E0,4.34E0,FREQ)

  return X853*4.+X1020*C1020+X1130*C1130

# ----------------------------------------------------------------------------------------

@njit(cache=True)
def O1OP( FREQ):
    if(FREQ >= 3.28805E15): return 9.*SEATON(3.28805E15,2.94E-18,1.E0,2.66E0,FREQ)
    else: return 0.0

# ----------------------------------------------------------------------------------------

@njit(cache=True)
def Mg2OP( FREQ,  TKEV):

  #    CROSS-SECTION TIMES PARTITION FUNCTION

  C1169=6.*exp(-4.43/TKEV)
  X1169=0.0
  X824=0.0

  if(FREQ >= 3.635492E15): X824 =SEATON(3.635492E15,1.40E-19,4.E0,6.7E0,FREQ)
  if(FREQ >= 2.564306E15): X1169=5.11E-19*(2.564306E15/FREQ)**3

  return X824*2.+X1169*C1169

# ----------------------------------------------------------------------------------------

PEACH2 = np.float64((-43.8941, -43.8941, -43.8941, -43.8941, -43.8941, -43.8941,#/*    500 */
                     -42.2444, -42.2444, -42.2444, -42.2444, -42.2444, -42.2444,#/*    600 */
                     -40.6054, -40.6054, -40.6054, -40.6054, -40.6054, -40.6054,#/*    759 */
                     -54.2389, -52.2906, -50.8799, -49.8033, -48.9485, -48.2490,#/*    760 */
                     -50.4108, -48.4892, -47.1090, -46.0672, -45.2510, -44.5933,#/*   1905 */
                     -52.0936, -50.0741, -48.5999, -47.4676, -46.5649, -45.8246,#/*   1906 */
                     -51.9548, -49.9371, -48.4647, -47.3340, -46.4333, -45.6947,#/*   1975 */
                     -54.2407, -51.7319, -49.9178, -48.5395, -47.4529, -46.5709,#/*   1976 */
                     -52.7355, -50.2218, -48.4059, -47.0267, -45.9402, -45.0592,#/*   3245 */
                     -53.5387, -50.9189, -49.0200, -47.5750, -46.4341, -45.5082,#/*   3246 */
                     -53.2417, -50.6234, -48.7252, -47.2810, -46.1410, -45.2153,#/*   3576 */
                     -53.5097, -50.8535, -48.9263, -47.4586, -46.2994, -45.3581,#/*   3577 */
                     -54.0561, -51.2365, -49.1980, -47.6497, -46.4302, -45.4414,#/*   3900 */
                     -53.8469, -51.0256, -48.9860, -47.4368, -46.2162, -45.2266)).reshape((14,6))

FREQSI2 = np.float64((4.9965417e15,3.9466738e15,1.5736321e15,1.5171539e15,9.2378947e14,8.3825004e14,7.6869872e14))
FLOG2 = np.float64((36.32984,36.14752,35.91165,34.99216,34.95561,34.45941,34.36234,34.27572,34.20161))
TLG2 = np.float64((9.21034,9.39266,9.54681,9.68034,9.79813,9.90349))

@njit(cache=True)
def Si2OP( FREQ,  FREQLG,  T,  TLOG):

    NT=min(5,int(floor(T/2000.))-4)
    if(NT<1): NT=1
    DT=(TLOG-TLG2[NT-1])/(TLG2[NT]-TLG2[NT-1])

    for N in range(7):
        if(FREQ>FREQSI2[N]):
            break

    D=(FREQLG-FLOG2[N])/(FLOG2[N+1]-FLOG2[N])

    if(N>1): N=2*N-2
    if(N==13): N=12

    D1=1.-D
    XWL1=PEACH2[N+1,NT-1]*D+PEACH2[N,NT-1]*D1
    XWL2=PEACH2[N+1,NT  ]*D+PEACH2[N,NT  ]*D1

    return exp(XWL1*(1.-DT)+XWL2*DT)*6.

# ----------------------------------------------------------------------------------------

@njit(cache=True)
def Ca2OP( FREQ,  TKEV):

    C1218=10.*exp(-1.697/TKEV)
    C1420=6.*exp(-3.142/TKEV)
    X1044=0.; X1218=0.; X1420=0.

    if(FREQ>=2.870454e15):
        XXX=(2.870454e15/FREQ)**3
        X1044=1.08e-19*XXX

    if(FREQ>=2.460127e15): X1218=1.64e-17*sqrt(2.460127e15/FREQ)
    if(FREQ>=2.110779e15): X1420=SEATON(2.110779e15,4.13e-18,3.,0.69, FREQ)

    return X1044+X1218*C1218+X1420*C1420

# ----------------------------------------------------------------------------------------

@njit(cache=True)
def LUKEOP( XN1,  XO1,  XMg2,  XSi2,  XCa2, STIM,  FREQ,  FREQLG,  T,  TLOG,  TKEV):

    #    N I, O I, Si II, Mg II, Ca II

    return ( N1OP(FREQ,TKEV          )*XN1 +
	     O1OP(FREQ               )*XO1 +
	     Mg2OP(FREQ,TKEV         )*XMg2+
	     Si2OP(FREQ,FREQLG,T,TLOG)*XSi2+
	     Ca2OP(FREQ,TKEV         )*XCa2)*STIM

# ----------------------------------------------------------------------------------------

@njit(cache=True)
def HOTOP():
    return 0.0 # dummy function, should fill this

# ----------------------------------------------------------------------------------------

@njit(cache=True)
def ELECOP( XNE):
    return 0.6653E-24*XNE

# ----------------------------------------------------------------------------------------

@njit(cache=True)
def H2RAOP(  XH1,  FREQ,  T,  TKEV,  TLOG):


  WW=(2.997925E18/min(FREQ,2.922E15))**2
  WW2 = WW*WW

  SIG=(8.14E-13+1.28e-6/WW+1.61e0/WW2)/WW2
  ARG=4.477/TKEV-4.6628E1+(1.8031E-3+(-5.023E-7+(8.1424E-11-5.0501E-15*T)*T)*T)*T-1.5*TLOG
  H1=XH1*2.0

  if(ARG > -80.0): return exp(ARG)*H1*H1*SIG
  else: return 0.0

# ----------------------------------------------------------------------------------------

@njit(cache=True)
def cop( T,  TKEV,  TK,  HKT,  TLOG, XNA,  XNE,  WLGRID, H1,  H2,  HMIN,  HE1,  HE2,
	 HE3,  C1,  AL1,  SI1,  SI2,CA1,  CA2,  MG1,  MG2,  FE1, N1,  O1):

    nW = len(WLGRID)
    OPACITY = np.zeros(nW)
    SCATTER = np.zeros(nW)

    for iWL in range(nW):
        FREQ=2.997925E18/WLGRID[iWL]
        FREQLG=log(FREQ)
        FREQ15=FREQ*1.E-15
        EHVKT=exp(-FREQ*HKT)
        STIM=1.0-EHVKT

        ACOOL = 0.0; ALUKE = 0.0

        AHYD = HOP(XNE,H1,H2,FREQ,FREQLG,T,TLOG,TKEV,STIM,EHVKT)
        AH2P = H2PLOP(H1,H2,FREQ,FREQLG,FREQ15,TKEV,STIM)
        AHMIN = HMINOP(H1,HMIN,FREQ,T,TKEV,XNE,EHVKT)
        SIGH = HRAYOP(H1,FREQ)
        AHE1 = HE1OP(HE1,HE2,XNE,FREQ,FREQLG,T,TKEV,TLOG,EHVKT,STIM)
        AHE2 = HE2OP(HE2, HE3,XNE,FREQ,FREQLG,T, TKEV, TLOG, EHVKT,STIM)
        AHEMIN = HEMIOP(HE1,FREQ,T,XNE)
        SIGHE = HERAOP(HE1,FREQ)

        if(T < 12000.):
            ACOOL = COOLOP(C1 ,MG1, AL1,SI1, FE1,STIM,FREQ,FREQLG,T,TLOG,TKEV,HKT)
        if(T < 30000.):
            ALUKE = LUKEOP( N1, O1, MG2, SI2, CA2,STIM,FREQ,FREQLG,T,TLOG,TKEV)

        AHOT = HOTOP()
        SIGEL = ELECOP(XNE)
        SIGH2 = H2RAOP(H1,FREQ,T,TKEV,TLOG)

        A=AHYD+AHMIN+AH2P+AHE1+AHE2+AHEMIN+ACOOL+ALUKE+AHOT
        B=SIGH+SIGHE+SIGEL+SIGH2

        OPACITY[iWL]=A+B
        SCATTER[iWL]=B

    return OPACITY, SCATTER

# ----------------------------------------------------------------------------------------

