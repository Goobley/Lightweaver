import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, calloc, free

cdef extern from "../RhCoreData.h":
    struct Element:
        char ID[3]
        int abundance_set
        int Nstage
        # int* mol_index # Can actually chuck mol_index and Nmolecule -- they're only used in the informational printout at the end of ChemEq
        # int Nmolecule
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

cdef convert_element(ele, Element* cEle):
    name = ele.name.encode()
    cEle.ID[0] = name[0]
    cEle.ID[1] = name[1]
    cEle.ID[2] = '\0'

    # cEle.Nmolecule = 0
    # cEle.mol_index = NULL
    cEle.abundance_set = True
    cEle.weight = ele.weight
    cEle.abund = ele.abundance

    ele.ionpot = np.ascontiguousarray(ele.ionpot)
    cdef np.ndarray[np.double_t, ndim=1] ionpot = np.ascontiguousarray(ele.ionpot)
    cEle.ionpot = &ionpot[0]

    cEle.Nstage = ele.pf.shape[0] 

    ele.pf = np.ascontiguousarray(ele.pf)
    cdef np.ndarray[np.double_t, ndim=2] pf = np.ascontiguousarray(ele.pf)
    cEle.pf = <double**> malloc(cEle.Nstage * sizeof(double*))
    for i in range(cEle.Nstage):
        cEle.pf[i] = <double*> &pf[i, 0]

    cEle.n = NULL


cdef class CAtomicTable:
    cdef AtomicTable* atomicTable

    def __init__(self, table):
        # TODO(cmo): This should actually take a LtePopulations, rather than an AtomicTable, so it can set the LtePops
        self.atomicTable = <AtomicTable*> malloc(sizeof(AtomicTable))
        self.atomicTable.totalAbund = table.totalAbundance
        self.atomicTable.weightPerH = table.weightPerH
        self.atomicTable.avgMolWeight = table.avgMolWeight

        cdef int Npf = len(table.Tpf)
        table.Tpf = np.ascontiguousarray(table.Tpf)
        cdef np.ndarray[np.double_t, ndim=1] Tpf = np.ascontiguousarray(table.Tpf)
        self.atomicTable.Tpf = &Tpf[0]
        self.atomicTable.Npf = Npf

        cdef int Nelem = len(table.elements)
        self.atomicTable.elements = <Element*> malloc(Nelem * sizeof(Element))
        self.atomicTable.Nelem = Nelem

        cdef Element* cEle
        for i, ele in enumerate(table.elements):
            cEle = &self.atomicTable.elements[i]
            convert_element(ele, cEle)

    def __dealloc__(self):
        for i in range(self.atomicTable.Nelem):
            free(self.atomicTable.elements[i].pf)
            free(self.atomicTable.elements[i].n)
        free(self.atomicTable.elements)
        free(self.atomicTable)

    # cpdef print_idx(self, index):
    #     cdef Element* ele = &self.atomicTable.elements[index]
    #     print("ID: ", ele.ID[0], ele.ID[1])
    #     print("ionpot: ")
    #     for i in range(ele.Nstage):
    #         print(ele.ionpot[i])

    #     for i in range(ele.Nstage):
    #         print('---------------')
    #         for j in range(self.atomicTable.Npf):
    #             print(ele.pf[i][j])

    

