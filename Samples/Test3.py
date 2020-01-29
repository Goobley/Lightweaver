from Falc80 import Falc80
from AtomicTable import AtomicTable
from CAtmosphere import CAtmosphere

atmos = Falc80()
atomicTable = AtomicTable()
atmos.convert_scales(atomicTable)
atmos.quadrature(5)
cAtmos = CAtmosphere(atmos)

