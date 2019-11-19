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
    Norder: int = 2
    Nperiod: int = 8
    Ndelay: int = 20

atmos = Falc80()
# atmos.B = np.ones(atmos.Nspace) * 1.0
# atmos.gammaB = np.ones(atmos.Nspace) * np.pi * 0.25
# atmos.chiB = np.zeros(atmos.Nspace)
at = AtomicTable()
atmos.convert_scales(at)
atmos.quadrature(5)

# aSet = RadiativeSet([H_6_CRD_atom(), He_large_atom()], set([]))
aSet = RadiativeSet([H_6_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe_atom(), He_large_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()], set([]))
# aSet = RadiativeSet([H_6_atom(), CaII_atom(), Al_atom()], set([]))
aSet.set_active('H', 'He')
spect = aSet.compute_wavelength_grid()
# spect = aSet.compute_wavelength_grid(np.linspace(10, 200, 100))

# [(47, 266), (266, 632), (470, 974), (632, 1170), (812, 1248)]
# molPaths = ['../Molecules/' + m + '.molecule' for m in ['H2', 'H2+', 'C2', 'N2', 'O2', 'CH', 'CO', 'CN', 'NH', 'NO', 'OH', 'H2O']] 
molPaths = ['../Molecules/' + m + '.molecule' for m in ['H2']]
start = time.time()
mols = MolecularTable(molPaths, at)

eqPops = aSet.compute_eq_pops(mols, atmos)
# prevH = np.copy(eqPops['Ca'])
ctx = LwContext(atmos, spect, aSet, eqPops, at, ngOptions=NgOptions(0,0,0))
# newH = np.copy(eqPops['Ca'])
delta = 1.0
dJ = 1.0
dRho = 1.0
it = 0
# ctx.configure_hprd_coeffs()
for it in range(303):
    # it += 1
    dJ = ctx.gamma_matrices_formal_sol()
    if it >= 3:
        delta = ctx.stat_equil()
    # if it >= 1:
    #     dRho = ctx.prd_redistribute()
    if delta < 1e-2:
        break

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