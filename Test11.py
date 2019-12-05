from Falc80 import Falc80
from AllOfTheAtoms import H_6_atom, H_6_CRD_atom, C_atom, O_atom, OI_ord_atom, Si_atom, Al_atom, CaII_atom, Fe_atom, FeI_atom, He_9_atom, He_atom, He_large_atom, MgII_atom, N_atom, Na_atom, S_atom
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


# rho = np.copy(np.load('en024048_hion/atmos/rho.npy')[::-1, :,:])
# Bifrost is right-handed with z up, so positive velocity is blue-shift, same as our system
z = np.copy(np.load('en024048_hion/atmos/z.npy')[::-1]) * 1e6
uz = np.copy(np.load('en024048_hion/atmos/uz.npy')[::-1, :,:])
temperature = np.copy(np.load('en024048_hion/atmos/temp.npy')[::-1, :,:])
nHTot = np.copy(np.load('en024048_hion/atmos/nHTot.npy')[::-1,:,:])
ne = np.copy(np.load('en024048_hion/atmos/ne.npy')[::-1,:,:])




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

wave = np.linspace(853.9444, 854.9444, 1001)
def synth_8542(data, conserve, useNe):
    # atmos = Atmosphere(ScaleType.Tau500, depthScale=10**data['tau'], temperature=data['temperature'], vlos=data['vlos']/1e2, vturb=data['vturb']/1e2)
    # atmos = Atmosphere(ScaleType.Geometric, depthScale=data['height']/1e2, temperature=data['temperature'], vlos=data['vlos']/1e2, vturb=data['vturb']/1e2)
    # atmos = Atmosphere(ScaleType.ColumnMass, depthScale=data['cmass'], temperature=data['temperature'], vlos=data['vlos'], vturb=data['vturb'], ne=data['ne'], nHTot=data['nHTot'])
    atmos = Atmosphere(ScaleType.Geometric, depthScale=data['height'], temperature=data['temperature'], vlos=data['vlos'], vturb=data['vturb'], ne=data['ne'], nHTot=data['nHTot'])
    atmos.convert_scales()
    atmos.quadrature(5)
    aSet = RadiativeSet([H_6_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe_atom(), He_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
    aSet.set_active('H', 'Ca')
    spect = aSet.compute_wavelength_grid()

    molPaths = ['../Molecules/' + m + '.molecule' for m in ['H2']]
    mols = MolecularTable(molPaths)

    if useNe:
        eqPops = aSet.compute_eq_pops(mols, atmos)
    else:
        eqPops = aSet.iterate_lte_ne_eq_pops(mols, atmos)
    ctx = LwContext(atmos, spect, eqPops, ngOptions=NgOptions(0,0,0), hprd=False, conserveCharge=conserve)
    iterate_ctx(ctx, prd=False)
    Iwave = ctx.compute_rays(wave, [1.0])
    return Iwave


datasets = []
for x in range(20, 25):
    for y in range(35, 40):
        data = {'height': z, 'temperature': temperature[:,x,y], 'vlos': uz[:,x,y], 'vturb': 3000*np.ones(temperature.shape[0]), 'ne': ne[:,x,y], 'nHTot': nHTot[:,x,y]}
        datasets.append(data)

with ProcessPoolExecutor() as executor:
    futures = [executor.submit(synth_8542, d, False, True) for d in datasets]
    wait(futures)
    spectra = [f.result() for f in futures]
