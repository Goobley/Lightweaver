from AtomicModel import AtomicModel, LineType, ExplicitContinuum, HydrogenicContinuum
import Constants as Const

from copy import deepcopy
import numpy as np
from numpy.polynomial.legendre import leggauss
cimport numpy as np
from libc.stdlib cimport malloc, calloc, free

ctypedef int bool_t
ctypedef void Ng
ctypedef void FixedTransition

cdef extern from "../RhCoreData.h":
    struct AtomicLine:
        bool_t polarizable
        bool_t PRD

        int i
        int j
        int Nlambda
        int Nblue
        int Ncomponent
        int Nxrd
        int nLine

        double lambda0
        double* wavelength
        double g_Lande_eff
        double Aji
        double Bji
        double Bij
        double* Rij
        double* Rji
        double** phi
        double** phi_Q
        double** phi_U
        double** phi_V
        double** psi_Q
        double** psi_U
        double** psi_V
        double* wphi
        double* Qelast
        double Grad
        # double qcore
        # double qwing
        double** rho_prd
        double* c_shift
        double* c_fraction
        double** gII
        int** id0
        int** id1
        double** frac
        Ng* Ng_prd
        Atom* atom
        AtomicLine** xrd
    
    struct AtomicContinuum:
        bool_t hydrogenic
        int i
        int j
        int Nlambda
        int Nblue
        int nCont
        double lambda0
        double* wavelength
        double alpha0
        double* alpha
        double* Rij
        double* Rji
        Atom *atom

    struct RhAccumulate:
        double** gij
        double** Vij
        double** wla
        double** chi_up
        double** chi_down
        double** Uji_down
        double*  eta
        double** Gamma
        double** RjiLine
        double** RijLine
        double** RjiCont
        double** RijCont
        bool_t* lineRatesDirty

    struct Atom:
        bool_t active
        bool_t NLTEpops
        int Nlevel
        int Nline
        int Ncont
        int Nfixed
        int Nprd
        int* stage
        int activeIndex
        double*  g
        double*  E
        double** C
        double*  vbroad
        double** n
        double** nstar
        double*  ntotal
        double** Gamma
        AtomicLine *line
        AtomicContinuum *continuum
        FixedTransition *ft
        Ng* Ng_n
        RhAccumulate* accumulate

    enum angleset:
        SET_VERTICAL
        SET_GL
        SET_A2
        SET_A4
        SET_A6
        SET_A8
        SET_B4
        SET_B6
        SET_B8
        NO_SET

    struct AngleSet:
        angleset set
        int Ninclination, Nazimuth

    struct Atmosphere:
        bool_t H_LTE
        bool_t Stokes
        bool_t hydrostatic
        int Ndim
        int* N
        int Nspace
        int Nrays
        int Natom
        int Nmolecule
        int NPRDactive
        int Nactiveatom
        int Nactivemol
        double* T
        double* ne
        double* vturb
        double** v_los # // Moved from spectrum because it feels out of place -- Should probably go to Geometry though
        double gravity
        double vmicro_char
        double lambda_ref
        double* wmu
        double* nHtot
#   double** nH; // nH is only used during atmospheric setup -- can probably go.
        double* nHmin
        double* B
        double* gamma_B
        double* chi_B
        double B_char
        double** cos_gamma
        double** cos_2chi
        double** sin_2chi
        AngleSet angleSet
        Atom* H
        Atom** atoms
        Atom** activeatoms
        Molecule* H2
        Molecule* OH
        Molecule* CH
        Molecule* molecules
        Molecule** activemols
        AtomicTable* atomicTable

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

    enum boundcond:
        ZERO
        THERMALIZED
        IRRADIATED
        REFLECTIVE

    enum mass_scale:
        GEOMETRIC
        COLUMN_MASS
        TAU500

    enum vertical:
        TOP
        BOTTOM

    struct Geometry:
        enum mass_scale scale
        enum boundcond vboundary[2]
        int Ndep
        int Nrays
        double* height
        double* cmass
        double* tau_ref
        double* mux
        double* muy
        double* muz
        double* wmu
        double* vel
        double** Itop
        double** Ibottom

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
 
cdef init_atom(Atom* atom):
    atom.active = False
    atom.NLTEpops = False
    atom.Nlevel = 0
    atom.Nline = 0
    atom.Ncont = 0
    atom.Nfixed = 0
    atom.Nprd = 0
    atom.stage = NULL
    atom.activeIndex = 0

    atom.g = NULL
    atom.E = NULL

    atom.C = NULL
    atom.vbroad = NULL
    atom.n = NULL
    atom.nstar = NULL
    atom.Gamma = NULL
    atom.line = NULL
    atom.continuum = NULL
    atom.ft = NULL
    atom.Ng_n = NULL
    atom.accumulate = NULL

cdef init_atomic_line(AtomicLine* line):
    line.polarizable = False
    line.PRD = False

    line.i = 0
    line.j = 0
    line.Nlambda = 0
    line.Nblue = 0
    line.Ncomponent = 0
    line.Nxrd = 0
    line.nLine = 0

    line.lambda0 = 0.0
    line.wavelength = NULL
    line.g_Lande_eff = 0.0
    line.Aji = 0.0
    line.Bji = 0.0
    line.Bij = 0.0
    line.Rij = NULL
    line.Rji = NULL
    line.phi = NULL
    line.phi_Q = NULL
    line.phi_U = NULL
    line.phi_V = NULL
    line.psi_Q = NULL
    line.psi_U = NULL
    line.psi_V = NULL
    line.wphi = NULL
    line.Qelast = NULL
    line.Grad = 0.0
    # line.qcore = 0.0
    # line.qwing = 0.0
    line.rho_prd = NULL
    line.c_shift = NULL
    line.c_fraction = NULL
    line.gII = NULL
    line.id0 = NULL
    line.id1 = NULL
    line.frac = NULL
    line.Ng_prd = NULL
    line.atom = NULL
    line.xrd = NULL


cdef init_atomic_continuum(AtomicContinuum* cont):
    cont.hydrogenic = True
    cont.i = 0
    cont.j = 0
    cont.Nlambda = 0
    cont.Nblue = 0
    cont.nCont = 0
    cont.lambda0 = 0.0
    cont.wavelength = NULL
    cont.alpha0 = 0.0
    cont.alpha = NULL
    cont.Rij = NULL
    cont.Rji = NULL
    cont.atom = NULL

cdef init_accumulate(RhAccumulate* acc):
    acc.gij = NULL
    acc.Vij = NULL
    acc.wla = NULL
    acc.chi_up = NULL
    acc.chi_down = NULL
    acc.Uji_down = NULL
    acc.eta = NULL
    acc.Gamma = NULL
    acc.RjiLine = NULL
    acc.RijLine = NULL
    acc.RjiCont = NULL
    acc.RijCont = NULL
    acc.lineRatesDirty = NULL

cdef free_accumulate(RhAccumulate* acc):
    free(<void*> acc.gij)
    free(<void*> acc.Vij)
    free(<void*> acc.wla)
    free(<void*> acc.chi_up)
    free(<void*> acc.chi_down)
    free(<void*> acc.Uji_down)
    free(<void*> acc.eta)
    free(<void*> acc.Gamma)
    free(<void*> acc.RjiLine)
    free(<void*> acc.RijLine)
    free(<void*> acc.RjiCont)
    free(<void*> acc.RijCont)
    free(<void*> acc.lineRatesDirty)


cdef init_atmosphere(Atmosphere* atmos):
    atmos.H_LTE = False
    atmos.Stokes = False
    atmos.hydrostatic = False
    atmos.Ndim = 0
    atmos.N = NULL
    atmos.Nspace = 0
    atmos.Nrays = 0
    atmos.Natom = 0
    atmos.Nmolecule = 0
    atmos.NPRDactive = 0
    atmos.Nactiveatom = 0
    atmos.Nactivemol = 0
    atmos.T = NULL
    atmos.ne = NULL
    atmos.vturb = NULL
    atmos.v_los = NULL
    atmos.gravity = 0.0
    atmos.vmicro_char = 0.0
    atmos.lambda_ref = 0.0
    atmos.wmu = NULL
    atmos.nHtot = NULL
    atmos.nHmin = NULL
    atmos.B = NULL
    atmos.gamma_B = NULL
    atmos.chi_B = NULL
    atmos.B_char = NULL
    atmos.cos_gamma = NULL
    atmos.cos_2chi = NULL
    atmos.sin_2chi = NULL
    atmos.angleSet.set = NO_SET
    atmos.angleSet.Nazimuth = 0
    atmos.angleSet.Ninclination = 0
    atmos.H = NULL
    atmos.atoms = NULL
    atmos.activeatoms = NULL
    atmos.H2 = NULL
    atmos.OH = NULL
    atmos.CH = NULL
    atmos.molecules = NULL
    atmos.activemols = NULL
    atmos.atomicTable = NULL

cdef init_geometry(Geometry* geo):
    geo.scale = mass_scale
    geo.vboundary[0] = ZERO
    geo.vboundary[1] = THERMALIZED
    geo.Ndep = 0
    geo.Nrays = 0
    geo.height = NULL
    geo.cmass = NULL
    geo.tau_ref = NULL
    geo.mux = NULL
    geo.muy = NULL
    geo.muz = NULL
    geo.wmu = NULL
    geo.vel = NULL
    geo.Itop = NULL
    geo.Ibottom = NULL

cdef class ComputationalAtomicContinuum:
    cdef AtomicContinuum* cCont

    @staticmethod
    cdef new(ComputationalAtom atom, cont, AtomicContinuum* cCont, options):
        self = ComputationalAtomicContinuum()
        self.atomicModel = atom
        self.continuumModel = cont
        self.cCont = cCont

        init_atomic_continuum(cCont)
        cCont.atom = &atom.cAtom

        cCont.i = cont.i
        cCont.j = cont.j
        cCont.nCont = atom.atomicModel.continua.index(cont)
        cCont.lambda0 = cont.lambda0

        if type(cont) is ExplicitContinuum:
            cCont.hydrogenic = False
        
        self.wavelength = np.ascontiguousarray(cont.wavelength)
        cdef int Nlambda = self.wavelength.shape[0]
        cCont.Nlambda = Nlambda
        self.alpha = np.ascontiguousarray(cont.alpha)
        cdef np.ndarray[np.double_t, ndim=1] ptr
        ptr = np.ascontiguousarray(self.wavelength)
        cCont.wavelength = &ptr[0]
        ptr = np.ascontiguousarray(self.alpha)
        cCont.alpha = &ptr[0]

        if atom.active:
            Nspace = atom.atmos.depthScale.shape[0]
            self.Rij = np.ascontiguousarray(np.zeros(Nspace))
            self.Rji = np.ascontiguousarray(np.zeros(Nspace))

            ptr = np.ascontiguousarray(self.Rij)
            cCont.Rij = &ptr[0]
            ptr = np.ascontiguousarray(self.Rji)
            cCont.Rji = &ptr[0]

        @property
        def i(self):
            return self.cCont.i

        @property
        def i(self):
            return self.cCont.j



cdef class ComputationalAtomicLine:
    cdef AtomicLine* cLine

    # def __dealloc__(self):
    #     free(<void*> self.cLine.c_shift)
    #     free(<void*> self.cLine.c_fraction)

    @staticmethod
    cdef new(ComputationalAtom atom, line, AtomicLine* cLine, options):
        self = ComputationalAtomicLine()
        self.atomicModel = atom
        self.lineModel = line
        self.cLine = cLine

        init_atomic_line(cLine)

        cLine.atom = &atom.cAtom

        cLine.i = line.i
        cLine.j = line.j
        cLine.nLine = atom.atomicModel.lines.index(line)
        cLine.Nlambda = line.Nlambda
        cLine.Grad = line.gRad
        cLine.g_Lande_eff = line.gLandeEff
        
        cLine.Aji = line.Aji
        cLine.Bji = line.Bji
        cLine.Bij = line.Bij

        cLine.lambda0 = line.lambda0

        if line.type == LineType.PRD and options.PRD.enable:
            cLine.PRD = True
            atom.Nprd += 1

        cLine.Ncomponent = 1
        cLine.c_shift = <double*> malloc(sizeof(double))
        cLine.c_fraction = <double*> malloc(sizeof(double))
        cLine.c_shift[0] = 0.0
        cLine.c_fraction[0] = 1.0

        cdef np.ndarray[np.double_t, ndim=1] wavelength
        if options.stokes:
            self.wavelength = np.ascontiguousarray(line.polarized_wavelength(options.stokes.b_char))
            wavelength = np.ascontiguousarray(self.wavelength)

            cLine.polarizable = np.any(line.wavelength != self.wavelength)
            cLine.wavelength = &wavelength[0]
            cLine.Nlambda = self.wavelength.shape[0]
        else:
            self.wavelength = np.ascontiguousarray(line.wavelength)
            wavelength = np.ascontiguousarray(self.wavelength)
            cLine.wavelength = &wavelength[0]
            cLine.Nlambda = self.wavelength.shape[0]

        cdef np.ndarray[np.double_t, ndim=1] ptr
        if atom.active:
            Nspace = atom.atmos.depthScale.shape[0]
            self.Rij = np.ascontiguousarray(np.zeros(Nspace))
            self.Rji = np.ascontiguousarray(np.zeros(Nspace))

            ptr = np.ascontiguousarray(self.Rij)
            cLine.Rij = &ptr[0]
            ptr = np.ascontiguousarray(self.Rji)
            cLine.Rji = &ptr[0]
        
            # if options.XRD.enable and len(line.xrd) > 0:

        @property
        def PRD(self):
            return self.cLine.PRD

        @property
        def i(self):
            return self.cLine.i

        @property
        def i(self):
            return self.cLine.j

        @property
        def Aji(self):
            return self.cLine.Aji

        @property
        def Bji(self):
            return self.cLine.Bji

        @property
        def Bij(self):
            return self.cLine.Bij

        
cdef class ComputationalAtom:
    # TODO(cmo): This should really be a pointer, since the atoms are allocated in a flat list
    # No -- we'll change the flat list to a pointer of pointers
    cdef Atom cAtom
    cdef int Nthread

    def __dealloc__(self):
        for i in range(self.cAtom.Nline):
            free(<void*> self.cAtom.line[i].c_shift)
            free(<void*> self.cAtom.line[i].c_fraction)
            free(<void*> self.cAtom.line[i].xrd)
            free(<void*> self.cAtom.line)
            free(<void*> self.cAtom.continuum)
            free(<void*> self.cAtom.n)
            free(<void*> self.cAtom.nstar)
            free(<void*> self.cAtom.C)

            if self.Nthread > 1:
                for i in range(self.Nthread):
                    free_accumulate(&self.cAtom.accumulate[i])

    def __init__(cAtom, atom, atmos, active, atomicTable, options):
        init_atom(&self.cAtom)
        self.atomicModel = atom
        self.atmos = atmos
        self.active = active
        self.atomicTable = atomicTable
        self.Nthread = options.Nthread

        atomicTable[atom.name].atom = self

        cdef int Nspace = atmos.depthScale.shape[0]
        cdef int Nlevel = len(atom.levels)
        self.nstar = np.ascontiguousarray(np.zeros((Nlevel, Nspace)))
        self.ntotal = np.ascontiguousarray(atomicTable[atom.name].abundance * atmos.nHTot)
        
        vtherm = 2.0 * Const.KBOLTZMANN / (Const.AMU * atomicTable[atom.name].weight)
        self.vbroad = np.ascontiguousarray(np.sqrt(vtherm * atmos.temperature + atmos.vturb**2))

        self.cAtom.active = active
        self.cAtom.Nlevel = Nlevel
        cdef np.ndarray[np.double_t, ndim=1] ptr
        self.g = np.ascontiguousarray(np.zeros(Nlevel))
        self.E = np.ascontiguousarray(np.zeros(Nlevel))
        for i, l in enumerate(atom.levels):
            self.g[i] = l.g
            self.E[i] = l.E_SI

        ptr = np.ascontiguousarray(self.g)
        self.cAtom.g = <double*> &ptr[0]
        ptr = np.ascontiguousarray(self.E)
        self.cAtom.E = <double*> &ptr[0]

        cdef np.ndarray[np.double_t, ndim=2] ptr2 = np.ascontiguousarray(self.nstar)
        self.cAtom.nstar = <double**> malloc(Nlevel * sizeof(double*))
        # As we used i in the enumerate before, Cython treats it as a python object here and won't do the cast-y stuff. Hence idx -- I'm not sure that's the reasoning actually. It seems in part linked to the type of the argument of the range
        for idx in range(Nlevel):
            self.cAtom.nstar[idx] = <double*> &ptr2[idx, 0]
        
        self.ntotal = np.ascontiguousarray(self.ntotal)
        ptr = np.ascontiguousarray(self.ntotal)
        self.cAtom.ntotal = &ptr[0]

        self.vbroad = np.ascontiguousarray(self.vbroad)
        ptr = np.ascontiguousarray(self.vbroad)
        self.cAtom.vbroad = &ptr[0]

        # TODO(cmo): Copy the levels, lines etc to this object (deepcopy), then add the extras like radiative rates to the entries in those arrays, leaving the model untouched. Then copy from those new models to the C Models. Given that we need to keep variables like Nblue updated later, we can use a setter property on this object to update them.

        cdef int Nline = len(atom.lines)
        self.cAtom.Nline = Nline
        self.cAtom.line = <AtomicLine*> malloc(Nline * sizeof(AtomicLine))
        cdef AtomicLine* cLine = NULL
        self.lines = []
        for i, l in enumerate(atom.lines):
            cLine = &self.cAtom.line[i]
            self.lines.append(ComputationalAtomicLine.new(self, l, cLine, options))
        
        if options.xrd.enable:
            for i, l in enumerate(atom.lines):
                if len(l.xrd) > 0:
                    length = len(l.xrd)
                    self.cAtom.line[i].Nxrd = length
                    self.cAtom.line[i].xrd = <AtomicLine**> malloc(length * sizeof(AtomicLine*))
                    for x in l.xrd:
                        xIdx = atom.lines.index(x)
                        self.cAtom.line[i].xrd[i] = &self.cAtom.line[xIdx]


        cdef int Ncont = len(atom.continua)
        self.cAtom.Ncont = Ncont
        self.cAtom.continuum = <AtomicContinuum*> malloc(Nline * sizeof(AtomicContinuum))
        cdef AtomicContinuum* cCont = NULL
        self.continua = []
        for i, l in enumerate(atom.lines):
            cCont = &self.cAtom.continuum[i]
            self.continua.append(ComputationalAtomicContinuum.new(self, l, cCont, options))

        self.collisions = deepcopy(atom.collisions)

        if self.active:
            self.cAtom.n = <double**> malloc(Nlevel * sizeof(double*))
            self.n = np.ascontiguousarray(np.zeros((Nlevel, Nspace)))
            ptr2 = np.ascontiguousarray(self.n)

            for idx in range(Nlevel):
                self.cAtom.n[idx] = <double*> &ptr2[idx, 0]

            if self.Nthread > 1:
                self.cAtom.accumulate = <RhAccumulate*> malloc(self.Nthread * sizeof(RhAccumulate))
                for i in range(self.Nthread):
                    init_accumulate(&self.cAtom.accumulate[i])


            self.C = np.ascontiguousarray(np.zeros((Nlevel*Nlevel, Nspace)))
            ptr2 = np.ascontiguousarray(self.C)
            self.cAtom.C = <double**> malloc(Nlevel*Nlevel*sizeof(double*))
            for idx in range(Nlevel*Nlevel):
                self.cAtom.C[idx] = <double*> &ptr2[idx, 0]

        else:
            self.n = self.nstar
            self.cAtom.n = self.cAtom.nstar

cdef class ComputationalAtmosphere:
    cdef Atmosphere cAtmos
    cdef Geometry cGeo

    def __dealloc__(self):
        free(<void*> self.cAtmos.N)

    def __init__(self, atmos, atoms, nRays, **kwargs):
        init_atmosphere(&self.cAtmos)
        init_geometry(&self.cGeo)
        # assume that the incoming atmosphere is already in the right units
        # Also assume that all depth scales are filled in? Yes for now
        self.atmos = atmos
        self.atoms = atoms

        # Set up the atmosphere and geometry structures based on the the inputs.
        # What order between atoms and atmosphere?
        # If we go atom->atmosphere we only have to do one pass (CAtoms are already in place)
        # However atmosphere->atom->atmosphere may be tidier
        # Upon reflection, I think atoms->atmosphere will be easiest to pull off
        cdef int Nspace = atmos.depthScale.shape[0]
        self.cAtmos.Ndim = 1
        self.cAtmos.N = <int*> malloc(sizeof(int))
        self.cAtmos[0] = Nspace
        self.geo.Ndep = Nspace

        self.cAtmos.gravity = atmos.gravity
        self.cAtmos.lambda_ref = 500.0

        cdef np.ndarray[np.double_t, ndim=1] ptr
        self.tau_ref = np.ascontiguousarray(atmos.tau_ref)
        ptr = np.ascontiguousarray(self.tau_ref)
        self.cAtmos.tau_ref = <double*> &ptr[0]

        self.cmass = np.ascontiguousarray(atmos.cmass)
        ptr = np.ascontiguousarray(self.cmass)
        self.cAtmos.cmass = <double*> &ptr[0]

        self.height = np.ascontiguousarray(atmos.height)
        ptr = np.ascontiguousarray(self.height)
        self.cAtmos.height = <double*> &ptr[0]

        self.temperature = np.ascontiguousarray(atmos.temperature)
        ptr = np.ascontiguousarray(self.temperature)
        self.cAtmos.T = <double*> &ptr[0]

        self.ne = np.ascontiguousarray(atmos.ne)
        ptr = np.ascontiguousarray(self.ne)
        self.cAtmos.ne = <double*> &ptr[0]

        self.height = np.ascontiguousarray(atmos.ne)
        ptr = np.ascontiguousarray(self.height)
        self.cAtmos.height = <double*> &ptr[0]
        
        self.vturb = np.ascontiguousarray(atmos.vturb)
        ptr = np.ascontiguousarray(self.vturb)
        self.cAtmos.vturb = <double*> &ptr[0]

        self.v_los = np.ascontiguousarray(atmos.v_los)
        ptr = np.ascontiguousarray(self.v_los)
        self.cGeo.v = <double*> &ptr[0]

        # Copy properties, set up atmosphere.

        if nRays > 1:
            # Get quadrature
            self.nRays = nRays
            x, w = leggauss(nRays)
            mid, halfWidth = 0.5, 0.5
            x = mid + halfWidth * x
            w *= halfWidth

            self.muz = np.ascontiguousarray(x)
            self.muy = np.ascontiguousarray(np.zeros_like(x))
            self.mux = np.ascontiguousarray(np.sqrt(1.0 - x**2))
            self.wmu = np.ascontiguousarray(w)

            self.cGeo.Nrays = self.nRays
            self.cAtmos.Nrays = self.nRays

            ptr = np.ascontiguousarray(self.muz)
            self.cGeo.muz = <double*> &ptr[0]
            ptr = np.ascontiguousarray(self.muy)
            self.cGeo.muy = <double*> &ptr[0]
            ptr = np.ascontiguousarray(self.mux)
            self.cGeo.mux = <double*> &ptr[0]
            ptr = np.ascontiguousarray(self.wmu)
            self.cGeo.wmu = <double*> &ptr[0]
            self.cAtmos.wmu = self.cGeo.wmu

        # TODO(cmo): Fix ray handling
        elif nRays == 1:
            raise ValueError("Needs special handling for one ray")
        else:
            raise ValueError("Unsupported nRays=%d"%nRays)

        if atmos.B is not None:
            self.cAtmos.Stokes = True
            raise ValueError("Not yet supporting magnetic atmospheres")

        self.nHTot = np.ascontiguousarray(atmos.nHTot)
        ptr = np.ascontiguousarray(self.nHTot)
        self.cAtmos.nHtot = <double*> &ptr[0]

        # Only supporting most basic BCs for now
        # TODO(cmo): Fix BC handling
        self.cGeo.vboundary[TOP] = ZERO
        self.cGeo.vboundary[BOTTOM] = THERMALIZED

        # Construct ComputationalAtoms for each atom -- we need to be passed Options from the caller then
        # Construct a ComputationalMolecule for each molecume
        # Get an AtomicTable from somewhere (caller? i.e. Context)
        # The context configures this guy, sets up the RLKlines, then does the sort lambda and initial solution, and allocates the background arrays



# TODO(cmo): RLK binding
# TODO(cmo): Basic molecule handling
# TODO(cmo): RhContext

