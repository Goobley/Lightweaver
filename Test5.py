from Falc80 import Falc80
from AllOfTheAtoms import H_6_atom, H_6_CRD_atom, C_atom, O_atom, Si_atom, Al_atom, CaII_atom, Fe_atom, He_9_atom, He_atom, He_large_atom, MgII_atom, N_atom, Na_atom, S_atom
# TODO(cmo): Find out why the other atoms weren't generated
from AtomicTable import AtomicTable
from AtomicSet import RadiativeSet
from Molecule import MolecularTable
from CAtmosphere import LwContext
from PyProto import InitialSolution
import matplotlib.pyplot as plt
import time
import numpy as np

from dataclasses import dataclass

def plot_zeeman_components(z):
    plt.plot(z.shift[z.alpha != 0], -1.0 * z.strength[z.alpha != 0], 'r+')
    plt.plot(z.shift[z.alpha == 0], z.strength[z.alpha == 0], 'b+')

@dataclass
class NgOptions:
    Norder: int = 3
    Nperiod: int = 20
    Ndelay: int = 20

atmos = Falc80()
atmos.B = np.ones(atmos.Nspace) * 1.0
atmos.gammaB = np.ones(atmos.Nspace) * np.pi * 0.25
atmos.chiB = np.zeros(atmos.Nspace)
at = AtomicTable()
atmos.convert_scales(at)
atmos.quadrature(5)
prevNe = np.copy(atmos.ne)
# atmos.vlos = np.random.randn(atmos.Nspace) * 8000
# atmos.temperature += np.random.randn(atmos.Nspace) * 0.12 * atmos.temperature
# atmos.ne[:] = np.array([1.25175889e+16, 1.30957836e+16, 1.37876773e+16, 1.49296515e+16,
#        1.64977754e+16, 1.75560532e+16, 1.89211874e+16, 2.07697755e+16,
#        2.33685059e+16, 2.51289836e+16, 2.72147474e+16, 2.98557771e+16,
#        3.33518787e+16, 3.76177962e+16, 4.26568864e+16, 4.63790118e+16,
#        4.96925462e+16, 5.05198797e+16, 4.90478321e+16, 4.78229543e+16,
#        4.81103149e+16, 4.85192724e+16, 4.82049382e+16, 4.68478529e+16,
#        4.54932276e+16, 4.49560766e+16, 4.48167295e+16, 4.49186340e+16,
#        4.53023806e+16, 4.56086636e+16, 4.67365121e+16, 4.83351469e+16,
#        4.89563047e+16, 4.93076772e+16, 5.07887337e+16, 5.32721761e+16,
#        5.77667488e+16, 6.54945670e+16, 7.74114942e+16, 9.17381687e+16,
#        1.13597997e+17, 1.41282639e+17, 1.68054861e+17, 1.95075150e+17,
#        2.18392687e+17, 2.26012490e+17, 2.17223183e+17, 1.97084888e+17,
#        1.70254433e+17, 1.37845245e+17, 1.05946748e+17, 9.90834748e+16,
#        1.24576027e+17, 1.66009455e+17, 2.17850816e+17, 2.91019181e+17,
#        4.08217220e+17, 6.30216892e+17, 9.69781188e+17, 1.48644806e+18,
#        2.26813311e+18, 3.44814113e+18, 4.27827137e+18, 5.35808625e+18,
#        6.85944978e+18, 8.99373997e+18, 1.23736070e+19, 1.84556469e+19,
#        2.63188078e+19, 3.89400749e+19, 5.31061034e+19, 7.48552895e+19,
#        1.08376257e+20, 1.70384012e+20, 2.77757695e+20, 4.44333046e+20,
#        6.88176843e+20, 1.04536485e+21, 1.54060077e+21, 2.21092194e+21,
#        2.97822709e+21, 3.87008177e+21])
# atmos.ne[:] = atmos.nHTot

# aSet = RadiativeSet([H_6_CRD_atom(), He_large_atom()], set([]))
aSet = RadiativeSet([H_6_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe_atom(), He_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()], set([]))
# aSet = RadiativeSet([H_6_atom(), CaII_atom(), Al_atom()], set([]))
aSet.set_active('H', 'Ca')
spect = aSet.compute_wavelength_grid()
# spect = aSet.compute_wavelength_grid(np.linspace(10, 200, 100))

# [(47, 266), (266, 632), (470, 974), (632, 1170), (812, 1248)]
# molPaths = ['../Molecules/' + m + '.molecule' for m in ['H2', 'H2+', 'C2', 'N2', 'O2', 'CH', 'CO', 'CN', 'NH', 'NO', 'OH', 'H2O']] 
molPaths = ['../Molecules/' + m + '.molecule' for m in ['H2']]
mols = MolecularTable(molPaths, at)

# eqPops = aSet.compute_eq_pops(mols, atmos)
eqPops = aSet.iterate_lte_ne_eq_pops(mols, atmos)
# eqPops['H2'][:] = 0.0
# eqPops['H-'][:] = 0.0
# prevH = np.copy(eqPops['Ca'])
ctx = LwContext(atmos, spect, aSet, eqPops, at, ngOptions=NgOptions(0,0,0), initSol=InitialSolution.Lte)
# input()
start = time.time()
# newH = np.copy(eqPops['Ca'])
delta = 1.0
dJ = 1.0
dRho = 1.0
it = 0
updateBgCount = 0
changedBgCount = 0
changedBg = False
# ctx.configure_hprd_coeffs()
maxOuter = 100
for thing in range(maxOuter):
    for it in range(1000):
        # it += 1
        dJ = ctx.gamma_matrices_formal_sol()
        if it >= 3:
                # delta = 1.0
            # if it % 3 == 4:
            #     delta = ctx.stat_equil(True, True, False)
            # if it % 3 == 1:
            #     delta = ctx.stat_equil(False, False, True)
            # else:
            # if (dJ < 1e-2 and delta < 1e-1) or (dJ < 1e-1 and delta < 1e-2):
            print(changedBgCount)
            if False and dJ * delta < 5e-3:
                delta = ctx.stat_equil(True, True, False, True)
                if changedBgCount >= 3:
                    changedBg = True
                changedBgCount += 1
            else:
                delta = ctx.stat_equil(True, False, False, False)
                changedBgCount = 0
                changedBg = False
            # changedBg = True
            
            # delta = ctx.stat_equil(True, False, True)
            # delta = ctx.stat_equil(False, False, False)
        # if it >= 1:
        #     dRho = ctx.prd_redistribute()
        if delta < 1e-2:
            eqPops.update_lte_atoms_Hmin_pops(atmos)
            ctx.background.update_background()
            break
        # dJ = ctx.gamma_matrices_formal_sol()
        # delta = ctx.stat_equil(False, False, False, True)
        # if delta < 1e-4:
            # break
        # if updateBgCount > 3:
        #     break
        # dJ = ctx.gamma_matrices_formal_sol()
        # delta = ctx.stat_equil(True, True, True, False)
        # updateBgCount += 1
    if thing != maxOuter-1:
        dJ = ctx.gamma_matrices_formal_sol()
        delta = ctx.stat_equil(True, True, True, True)
        if dJ < 1e-2 and delta < 1e-2:
            break

print(thing)
print(it)

for it in range(3):
    dJ = ctx.gamma_matrices_formal_sol()
    
# print(ctx.activeAtoms[0].trans[4].phi[20,0,0,0])
s = ctx.spect
oldI = np.copy(s.I)
# ctx.single_stokes_fs()
# # plt.plot(s.wavelength, s.I[:, -1] / oldI[:, -1], '+')
# plt.plot(s.wavelength, oldI[:, -1])
# plt.plot(s.wavelength, s.I[:, -1])
# plt.ylim(0.95, 1.05)
# plt.show()
# dJStokes = ctx.single_stokes_fs()
# print(ctx.activeAtoms[0].trans[4].phi[20,0,0,0])
end = time.time()
print('%e s' % (end-start))

plt.plot(s.wavelength, s.I[:,-1])
from helita.sim.rh import Rhout
rh = Rhout('/Users/goobley/VanillaRh/rhf1d/run/')
plt.plot(rh.wave, rh.imu[-1, :], '--')