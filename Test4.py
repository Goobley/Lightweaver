import numpy as np
from Falc80 import Falc80
from AtomicTable import AtomicTable
from CAtmosphere import LwContext
from AtomicSet import RadiativeSet
from AllOfTheAtoms import CaIIatom
from PyProto import background
import time


atmos = Falc80()
at = AtomicTable()
atmos.convert_scales(at)
atmos.quadrature(5)
aSet = RadiativeSet([CaIIatom()], [])
aSet.set_active('Ca')
# spect = aSet.compute_wavelength_grid(np.linspace(150, 600, 500))
spect = aSet.compute_wavelength_grid()

np.seterr(invalid='raise', divide='raise')

bg = background(atmos, spect)
activeAtoms = [a for a in aSet.activeAtoms]
ctx = LwContext(atmos, spect, activeAtoms, bg, at)
# ctx.gamma_matrices_formal_sol()
# delta = ctx.stat_equil()
# print("delta: %e"%delta)

start = time.time()
delta = 1.0
it = 0
while delta > 1e-2 and it < 200:
    it += 1
    ctx.gamma_matrices_formal_sol()
    delta = ctx.stat_equil()
    print(delta, it)
end = time.time()
print('%.2e'%(end-start))
# ctx.gamma_matrices_formal_sol()
# Iplus = gamma_matrices(atmos, spect, activeAtoms, bg)
# delta = stat_equil(atmos, activeAtoms)