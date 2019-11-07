from Falc80 import Falc80
from AllOfTheAtoms import H_6_atom, C_atom, O_atom, Si_atom, Al_atom, CaII_atom, Fe_atom, He_9_atom, MgII_atom, N_atom, Na_atom, S_atom
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
    Nperiod: int = 3
    Ndelay: int = 12

atmos = Falc80()
atmos.B = np.ones(atmos.Nspace) * 1.0
atmos.gammaB = np.ones(atmos.Nspace) * np.pi * 0.25
atmos.chiB = np.zeros(atmos.Nspace)
at = AtomicTable()
atmos.convert_scales(at)
atmos.quadrature(5)

# NOTE(cmo): adding FeAtom seems to make us explode... Need to fix
aSet = RadiativeSet([H_6_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe_atom(), He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()], set([]))
aSet.set_active('Ca', 'Mg', 'H')
spect = aSet.compute_wavelength_grid(np.linspace(150, 600, 500))

molPaths = ['../Molecules/' + m + '.molecule' for m in ['H2', 'H2+', 'C2', 'N2', 'O2', 'CH', 'CO', 'CN', 'NH', 'NO', 'OH', 'H2O']]
# molPaths = ['../Molecules/' + m + '.molecule' for m in ['H2', 'H2+', 'CH', 'OH']]
start = time.time()
mols = MolecularTable(molPaths, at)

eqPops = aSet.compute_eq_pops(mols, atmos)
eqPops['H-'][:] = np.array([4.67794609e+02, 5.47693681e+02, 6.55528291e+02, 8.65609530e+02,
      1.22730835e+03, 1.52680715e+03, 1.98916083e+03, 2.77256718e+03,
      4.24750588e+03, 5.54341861e+03, 7.44283531e+03, 1.05093918e+04,
      1.59466616e+04, 2.51981523e+04, 4.13008797e+04, 7.06415001e+04,
      1.27124700e+05, 2.03044366e+05, 2.87883960e+05, 4.36296816e+05,
      6.38340525e+05, 8.24612644e+05, 1.03740993e+06, 1.25115833e+06,
      1.47433388e+06, 1.64884009e+06, 1.76083452e+06, 1.79993483e+06,
      1.79719092e+06, 1.91107118e+06, 2.16001000e+06, 2.74213094e+06,
      3.83424075e+06, 5.48443518e+06, 7.73744187e+06, 1.02264141e+07,
      1.44033483e+07, 2.22729948e+07, 3.70605076e+07, 6.02443281e+07,
      1.14634053e+08, 2.37496499e+08, 4.71414342e+08, 1.00673360e+09,
      2.18585966e+09, 4.70384573e+09, 8.19172447e+09, 1.30586211e+10,
      1.92423676e+10, 2.91796123e+10, 4.96675353e+10, 1.18703492e+11,
      3.01332036e+11, 6.31806616e+11, 1.20099820e+12, 2.22842972e+12,
      4.43544397e+12, 1.01441945e+13, 2.27323572e+13, 4.99089483e+13,
      1.07318072e+14, 2.25942528e+14, 3.24406555e+14, 4.59193055e+14,
      6.44056691e+14, 9.05417550e+14, 1.30020158e+15, 1.96191406e+15,
      2.69498240e+15, 3.83119209e+15, 5.00891484e+15, 6.70350090e+15,
      9.13850447e+15, 1.31272364e+16, 1.92683091e+16, 2.77765323e+16,
      3.90576523e+16, 5.38468848e+16, 7.23138198e+16, 9.50952884e+16,
      1.19609859e+17, 1.46480089e+17])

ctx = LwContext(atmos, spect, aSet, eqPops, at, ngOptions=NgOptions(0,0,0))
delta = 1.0
dJ = 1.0
it = 0
for it in range(200):
    # it += 1
    dJ = ctx.gamma_matrices_formal_sol()
    if dJ < 1e-2:
        break
    delta = ctx.stat_equil()
# print(ctx.activeAtoms[0].trans[4].phi[20,0,0,0])
s = ctx.spect
oldI = np.copy(s.I)
ctx.single_stokes_fs()
# plt.plot(s.wavelength, s.I[:, -1] / oldI[:, -1], '+')
plt.plot(s.wavelength, oldI[:, -1])
plt.plot(s.wavelength, s.I[:, -1])
# plt.ylim(0.95, 1.05)
# plt.show()
# dJStokes = ctx.single_stokes_fs()
# print(ctx.activeAtoms[0].trans[4].phi[20,0,0,0])
end = time.time()
print('%e s' % (end-start))