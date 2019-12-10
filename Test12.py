from Falc80 import Falc80
from AllOfTheAtoms import H_6_atom, H_6_CRD_atom, H_3_atom, C_atom, O_atom, OI_ord_atom, Si_atom, Al_atom, CaII_atom, Fe_atom, FeI_atom, He_9_atom, He_atom, He_large_atom, MgII_atom, N_atom, Na_atom, S_atom
from Atmosphere import Atmosphere, ScaleType
from AtomicSet import RadiativeSet
from AtomicTable import get_global_atomic_table
from Molecule import MolecularTable
from CAtmosphere import LwContext
from PyProto import InitialSolution
from dataclasses import dataclass
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor, wait
from tqdm import tqdm
from Utils import NgOptions
from astropy.io import fits



def iterate_ctx(ctx, prd=True, Nscatter=3, NmaxIter=500):
    for i in range(NmaxIter):
        dJ = ctx.formal_sol_gamma_matrices()
        if i < Nscatter:
            continue
        delta = ctx.stat_equil()
        if prd:
            dRho = ctx.prd_redistribute(maxIter=5)

        if dJ < 3e-3 and delta < 1e-3:
            print(i)
            print('----------')
            return

# wave = np.linspace(853.9444, 854.9444, 1001)
wave = np.linspace(392, 398, 10001)
def synth_8542(atmos, conserve, useNe, stokes=False):
    # atmos = Atmosphere(ScaleType.Tau500, depthScale=10**data['tau'], temperature=data['temperature'], vlos=data['vlos']/1e2, vturb=data['vturb']/1e2)
    # atmos = Atmosphere(ScaleType.Geometric, depthScale=data['height']/1e2, temperature=data['temperature'], vlos=data['vlos']/1e2, vturb=data['vturb']/1e2)
    # atmos = Atmosphere(ScaleType.ColumnMass, depthScale=data['cmass'], temperature=data['temperature'], vlos=data['vlos'], vturb=data['vturb'], ne=data['ne'], nHTot=data['nHTot'])
    # atmos = Atmosphere(ScaleType.Geometric, depthScale=data['height'], temperature=data['temperature'], vlos=data['vlos'], vturb=data['vturb'], ne=data['ne'], nHTot=data['nHTot'])
    atmos.convert_scales()
    atmos.quadrature(5)
    aSet = RadiativeSet([H_3_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe_atom(), He_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
    aSet.set_active('H', 'Ca')
    spect = aSet.compute_wavelength_grid()

    molPaths = ['../Molecules/' + m + '.molecule' for m in ['H2']]
    mols = MolecularTable(molPaths)

    if useNe:
        eqPops = aSet.compute_eq_pops(mols, atmos)
    else:
        eqPops = aSet.iterate_lte_ne_eq_pops(mols, atmos)
    ctx = LwContext(atmos, spect, eqPops, ngOptions=NgOptions(0,0,0), hprd=True, conserveCharge=conserve)
    iterate_ctx(ctx, prd=True)
    eqPops.update_lte_atoms_Hmin_pops(atmos)
    Iwave = ctx.compute_rays(wave, [atmos.muz[-1]], stokes=stokes)
    return Iwave

def add_B(atmos):
    atmos.B = np.ones(atmos.Nspace) * 1.0
    atmos.gammaB = np.ones(atmos.Nspace) * np.pi * 0.25
    atmos.chiB = np.zeros(atmos.Nspace)


atmosCons = Falc80()
# add_B(atmosCons)
IwaveCons = synth_8542(atmosCons, conserve=True, useNe=False, stokes=False)
atmosLte = Falc80()
# add_B(atmosLte)
IwaveLte = synth_8542(atmosLte, conserve=False, useNe=False, stokes=False)
atmosFal = Falc80()
# add_B(atmosFal)
Iwave = synth_8542(atmosFal, conserve=False, useNe=True, stokes=False)

# atmos = Falc80()
# add_B(atmos)
# atmos.convert_scales()
# atmos.quadrature(5)
# aSet = RadiativeSet([H_3_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe_atom(), He_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
# aSet.set_active('H', 'Ca')
# spect = aSet.compute_wavelength_grid()

# molPaths = ['../Molecules/' + m + '.molecule' for m in ['H2']]
# mols = MolecularTable(molPaths)
# eqPops = aSet.compute_eq_pops(mols, atmos)
# ctx = LwContext(atmos, spect, eqPops, ngOptions=NgOptions(0,0,0), hprd=False, conserveCharge=False)
# iterate_ctx(ctx, prd=False)
# ctx.single_stokes_fs()
# s = ctx.spect
