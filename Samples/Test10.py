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


cube = fits.getdata('Bifrost/BIFROST_en024048_bin_1x1_101_300_101_300_vt=5_resampled_153.fits').astype('<f8')
cube[8,:,:,:] = 3e5
x = 0
y = 0

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
def synth_8542(data, conserve):
    # atmos = Atmosphere(ScaleType.Tau500, depthScale=10**data['tau'], temperature=data['temperature'], vlos=data['vlos']/1e2, vturb=data['vturb']/1e2)
    atmos = Atmosphere(ScaleType.Geometric, depthScale=data['height']/1e2, temperature=data['temperature'], vlos=data['vlos']/1e2, vturb=data['vturb']/1e2)
    atmos.convert_scales()
    atmos.quadrature(5)
    aSet = RadiativeSet([H_6_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe_atom(), He_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
    aSet.set_active('H', 'Ca')
    spect = aSet.compute_wavelength_grid()

    molPaths = ['../Molecules/' + m + '.molecule' for m in ['H2']]
    mols = MolecularTable(molPaths)

    eqPops = aSet.iterate_lte_ne_eq_pops(mols, atmos)
    ctx = LwContext(atmos, spect, eqPops, ngOptions=NgOptions(0,0,0), hprd=False, initSol=InitialSolution.Lte, conserveCharge=conserve)
    iterate_ctx(ctx, prd=False)
    Iwave = ctx.compute_rays(wave, [1.0])
    return Iwave


datasets = []
for x in range(20, 25):
    for y in range(35, 40):
        data = {'tau': cube[0,x,y,:], 'height': cube[1,x,y,:],'temperature': cube[2,x,y,:], 'vlos': cube[9,x,y,:], 'vturb': cube[8,x,y,:]}
        datasets.append(data)

with ProcessPoolExecutor() as executor:
    futures = [executor.submit(synth_8542, d, False) for d in datasets]
    wait(futures)
    spectra = [f.result() for f in futures]
