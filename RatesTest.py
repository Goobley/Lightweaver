import numpy as np
from AllOfTheAtoms import CaIIatom
from AtomicTable import AtomicTable
from AtomicSet import RadiativeSet
from PyProto import ComputationalAtom
from Falc80 import Falc80

atmos = Falc80()
atomicTable = AtomicTable()
atmos.convert_scales(atomicTable)
atmos.quadrature(Nrays=5)

ca = CaIIatom()
radSet = RadiativeSet([ca], [])
radSet.set_active('Ca')
spect = radSet.compute_wavelength_grid()
caComp = ComputationalAtom(ca, atmos, atomicTable, spect)

caComp.compute_collisions()

# For a particular collisional rate computation (x in the collisions list of ca)
# Cmat = np.zeros((caComp.Nlevel, caComp.Nlevel, atmos.Nspace))
# ca.collisions[x].compute_rates(atmos, caComp.nstar, Cmat)
# These have now been added to Cmat