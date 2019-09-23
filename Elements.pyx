import numpy as np
cimport numpy as np
np.import_array()


cdef extern from "../RhCoreData.h":
    struct Element:
        char ID[3]
        int abundance_set
        int Nstage
        int Nmolecule
        double weight
        double abund
        double* ionpot
        double** pf
        double** n    
    
    struct AtomicTable:
        double* Tpf
        Element* elements
        int Nelem
        int Npf
        double totalAbund
        double avgMolWeight
        double weightPerH

    int read_abundance(AtomicTable* table, double metallicity, const char* abundInput, const char* pfPath)

cdef class PyElement:
    cdef Element* celement
    cdef int Npf

    @staticmethod
    cdef new(Element* ele, int Npf):
        this = PyElement()
        this.Npf = Npf
        this.celement = ele
        return this

    @property
    def ID(self):
        return <bytes> self.celement.ID

    @property
    def abundance_set(self):
        return self.celement.abundance_set

    @property
    def Nstage(self):
        return self.celement.Nstage

    @property
    def Nmolecule(self):
        return self.celement.Nmolecule

    @property
    def weight(self):
        return self.celement.weight

    @property
    def abund(self):
        return self.celement.abund

    @property
    def ionpot(self):
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.celement.Nstage
        ndarray = np.PyArray_SimpleNewFromData(1, &shape[0],
                                               np.NPY_FLOAT64, <void*>self.celement.ionpot)
        return ndarray

    @property
    def pf(self):
        cdef np.npy_intp shape[2]
        shape[0] = <np.npy_intp> self.celement.Nstage
        shape[1] = <np.npy_intp> self.Npf
        ndarray = np.PyArray_SimpleNewFromData(2, &shape[0],
                                               np.NPY_FLOAT64, <void*>self.celement.pf[0])
        return ndarray


cdef class PyAtomicTable:
    cdef AtomicTable ctable

    def __init__(self, metallicity, abundInput, pfPath):
        # self.ctable = AtomicTable()
        # self.ctable = AtomicTable()
        cdef bytes abund = abundInput.encode()
        cdef char* abundStr = abund
        cdef bytes pf = pfPath.encode()
        cdef char* pfStr = pf

        read_abundance(&(self.ctable), metallicity, abundStr, pfStr)

    @property
    def Nelem(self):
        return self.ctable.Nelem

    @property
    def Npf(self):
        return self.ctable.Npf

    @property
    def totalAbund(self):
        return self.ctable.totalAbund

    @property
    def avgMolWeight(self):
        return self.ctable.avgMolWeight

    @property
    def weightPerH(self):
        return self.ctable.weightPerH

    # This is slooooooooooow ~ 1.5 us, other method is 89 ns
    # @property
    # def Tpf(self):
    #     cdef double[:] view = <double[:self.ctable.Npf]> self.ctable.Tpf
    #     return np.asarray(view)

    @property
    def Tpf(self):
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.ctable.Npf
        ndarray = np.PyArray_SimpleNewFromData(1, &shape[0],
                                               np.NPY_FLOAT64, <void*>self.ctable.Tpf)
        return ndarray

    @property
    def elements(self):
        eles = [PyElement.new(&self.ctable.elements[i], self.ctable.Npf) for i in range(self.ctable.Nelem)]
        return eles
