#include "LightweaverAmalgamated.cpp"

struct ArenaAllocator
{
    u8* data = nullptr;
    u8* fillPoint = nullptr;
    i64 maxCapacity = 0;

    ArenaAllocator(i64 capacity)
    {
        data = (u8*)calloc(capacity, sizeof(u8));
        if (!data)
            assert(false);
        fillPoint = data;
        maxCapacity = capacity;
    }

    ~ArenaAllocator()
    {
        release();
    }

    template <typename T>
    T* allocate(i64 count = 1)
    {
        u8* ptr = fillPoint;
        fillPoint += sizeof(T) * count;
        if (fillPoint - data > maxCapacity)
            assert(false);

        return (T*)ptr;
    }

    void release()
    {
        free(data);
    }


    F64View alloc1d(i64 dim0)
    {
        f64* store = allocate<f64>(dim0);
        F64View result(store, dim0);
        return result;
    }

    F64View2D alloc2d(i64 dim0, i64 dim1)
    {
        f64* store = allocate<f64>(dim0 * dim1);
        F64View2D result(store, dim0, dim1);
        return result;
    }

    F64View3D alloc3d(i64 dim0, i64 dim1, i64 dim2)
    {
        f64* store = allocate<f64>(dim0 * dim1 * dim2);
        F64View3D result(store, dim0, dim1, dim2);
        return result;
    }

    F64View4D alloc4d(i64 dim0, i64 dim1, i64 dim2, i64 dim3)
    {
        f64* store = allocate<f64>(dim0 * dim1 * dim2 * dim3);
        F64View4D result(store, dim0, dim1, dim2, dim3);
        return result;
    }
};

void init_atmos(ArenaAllocator* store, Atmosphere* atmos, const std::string& path)
{
    FILE* data = fopen(path.c_str(), "rb");
    if (!data)
        assert(false);


    fread(&atmos->Nspace, sizeof(int), 1, data);
    fread(&atmos->Nrays, sizeof(int), 1, data);
    printf("Nz: %d, Nrays: %d\n", atmos->Nspace, atmos->Nrays);
    atmos->Ndim = 1;
    atmos->Nx = atmos->Nspace;
    atmos->Ny = 0;
    atmos->Nz = 0;
    atmos->Noutgoing = 1;

    atmos->z = store->alloc1d(atmos->Nspace);
    fread(atmos->z.data, sizeof(f64), atmos->Nspace, data);
    atmos->height = store->alloc1d(atmos->Nspace);
    fread(atmos->height.data, sizeof(f64), atmos->Nspace, data);
    atmos->temperature = store->alloc1d(atmos->Nspace);
    fread(atmos->temperature.data, sizeof(f64), atmos->Nspace, data);
    atmos->ne = store->alloc1d(atmos->Nspace);
    fread(atmos->ne.data, sizeof(f64), atmos->Nspace, data);
    atmos->vz = store->alloc1d(atmos->Nspace);
    fread(atmos->vz.data, sizeof(f64), atmos->Nspace, data);
    atmos->vlosMu = store->alloc2d(atmos->Nspace, atmos->Nrays);
    fread(atmos->vlosMu.data, sizeof(f64), atmos->Nspace * atmos->Nrays, data);
    atmos->vturb = store->alloc1d(atmos->Nspace);
    fread(atmos->vturb.data, sizeof(f64), atmos->Nspace, data);
    atmos->nHTot = store->alloc1d(atmos->Nspace);
    fread(atmos->nHTot.data, sizeof(f64), atmos->Nspace, data);
    atmos->muz = store->alloc1d(atmos->Nspace);
    fread(atmos->muz.data, sizeof(f64), atmos->Nrays, data);
    atmos->wmu = store->alloc1d(atmos->Nspace);
    fread(atmos->wmu.data, sizeof(f64), atmos->Nrays, data);

    atmos->zLowerBc.type = RadiationBc::THERMALISED;
    atmos->zUpperBc.type = RadiationBc::ZERO;

    fclose(data);
}

void init_spect(ArenaAllocator* store, Spectrum* spect,
                const Atmosphere& atmos, const std::string& path)
{
    FILE* data = fopen(path.c_str(), "rb");
    if (!data)
        assert(false);

    int Nspect;
    fread(&Nspect, sizeof(int), 1, data);
    printf("%d\n", Nspect);
    spect->wavelength = store->alloc1d(Nspect);
    fread(spect->wavelength.data, sizeof(f64), Nspect, data);
    spect->I = store->alloc3d(Nspect, atmos.Nrays, 1);
    fread(spect->I.data, sizeof(f64), Nspect * atmos.Nrays * 1, data);
    spect->J = store->alloc2d(Nspect, atmos.Nspace);
    fread(spect->J.data, sizeof(f64), Nspect * atmos.Nspace, data);

    fclose(data);
}

void init_atoms(ArenaAllocator* store, std::vector<Atom*>* atomStore,
                Atmosphere* atmos, const Spectrum& spect,
                const std::string& path, bool detailed)
{
    FILE* data = fopen(path.c_str(), "rb");
    if (!data)
        assert(false);

    int Natom;
    fread (&Natom, sizeof(int), 1, data);

    for (int a = 0; a < Natom; ++a)
    {
        Atom* atom = store->allocate<Atom>();
        atom->trans = decltype(atom->trans)();
        atom->atmos = atmos;

        int Nlevel;
        fread(&Nlevel, sizeof(int), 1, data);
        int Ntrans;
        fread(&Ntrans, sizeof(int), 1, data);
        atom->Nlevel = Nlevel;
        atom->Ntrans = Ntrans;

        atom->n = store->alloc2d(Nlevel, atmos->Nspace);
        fread(atom->n.data, sizeof(f64), Nlevel * atmos->Nspace, data);
        atom->nStar = store->alloc2d(Nlevel, atmos->Nspace);
        fread(atom->nStar.data, sizeof(f64), Nlevel * atmos->Nspace, data);

        atom->vBroad = store->alloc1d(atmos->Nspace);
        fread(atom->vBroad.data, sizeof(f64), atmos->Nspace, data);
        atom->nTotal = store->alloc1d(atmos->Nspace);
        fread(atom->nTotal.data, sizeof(f64), atmos->Nspace, data);

        atom->stages = store->alloc1d(Nlevel);
        fread(atom->stages.data, sizeof(f64), Nlevel, data);

        if (!detailed)
        {
            atom->Gamma = store->alloc3d(Nlevel, Nlevel, atmos->Nspace);
            fread(atom->Gamma.data, sizeof(f64), Nlevel * Nlevel * atmos->Nspace, data);
            atom->C = store->alloc3d(Nlevel, Nlevel, atmos->Nspace);
            fread(atom->C.data, sizeof(f64), Nlevel * Nlevel * atmos->Nspace, data);

            // NOTE(cmo): These are only caching storage.
            atom->U = store->alloc2d(Nlevel, atmos->Nspace);
            atom->eta = store->alloc1d(atmos->Nspace);
            atom->chi = store->alloc2d(Nlevel, atmos->Nspace);
        }

        char deadbeef[4];
        fread(deadbeef, sizeof(char), 4, data);

        for (int t = 0; t < Ntrans; ++t)
        {
            char typecode;
            fread(&typecode, sizeof(char), 1, data);

            enum TransitionType type = If typecode == 'L' Then
                TransitionType::LINE  Else TransitionType::CONTINUUM End;

            Transition* trans = store->allocate<Transition>();
            trans->type = type;
            fread(&trans->i, sizeof(int), 1, data);
            fread(&trans->j, sizeof(int), 1, data);
            fread(&trans->Nblue, sizeof(int), 1, data);
            int NtransLambda;
            fread(&NtransLambda, sizeof(int), 1, data);

            fread(&trans->lambda0, sizeof(f64), 1, data);
            // TODO(cmo): Ignoring polarisation for now
            trans->polarised = false;

            trans->wavelength = store->alloc1d(NtransLambda);
            fread(trans->wavelength.data, sizeof(f64), NtransLambda, data);

            int Nlambda = spect.wavelength.shape(0);
            i8* activePtr = store->allocate<i8>(Nlambda);
            trans->active = BoolView((bool*)activePtr, Nlambda);
            fread(trans->active.data, sizeof(bool), Nlambda, data);

            trans->Rij = store->alloc1d(atmos->Nspace);
            fread(trans->Rij.data, sizeof(f64), atmos->Nspace, data);
            trans->Rji = store->alloc1d(atmos->Nspace);
            fread(trans->Rji.data, sizeof(f64), atmos->Nspace, data);

            if (type == TransitionType::LINE)
            {
                trans->dopplerWidth = Constants::CLight / trans->lambda0;
                fread(&trans->Aji, sizeof(f64), 1, data);
                fread(&trans->Bji, sizeof(f64), 1, data);
                fread(&trans->Bij, sizeof(f64), 1, data);

                trans->Qelast = store->alloc1d(atmos->Nspace);
                fread(trans->Qelast.data, sizeof(f64), atmos->Nspace, data);
                trans->aDamp = store->alloc1d(atmos->Nspace);
                fread(trans->aDamp.data, sizeof(f64), atmos->Nspace, data);

                trans->phi = store->alloc4d(NtransLambda, atmos->Nrays,
                                            2, atmos->Nspace);
                fread(trans->aDamp.data, sizeof(f64),
                      NtransLambda * atmos->Nrays * 2 * atmos->Nspace, data);
                trans->wphi = store->alloc1d(atmos->Nspace);
                fread(trans->wphi.data, sizeof(f64), atmos->Nspace, data);
            }
            else
            {
                trans->dopplerWidth = 1.0;
                trans->alpha = store->alloc1d(NtransLambda);
                fread(trans->alpha.data, sizeof(f64), NtransLambda, data);
            }
            atom->trans.push_back(trans);
        }

        atom->ng = Ng(0, 0, 0, atom->n.flatten());
        atomStore->push_back(atom);
    }
    fclose(data);
}

void init_background(ArenaAllocator* store, Background* bg,
                     const std::string& path)
{
    FILE* data = fopen(path.c_str(), "rb");
    if (!data)
        assert(false);

    int Nlambda;
    fread(&Nlambda, sizeof(Nlambda), 1, data);
    int Nspace;
    fread(&Nspace, sizeof(Nspace), 1, data);

    bg->chi = store->alloc2d(Nlambda, Nspace);
    fread(bg->chi.data, sizeof(f64), Nlambda * Nspace, data);
    bg->eta = store->alloc2d(Nlambda, Nspace);
    fread(bg->eta.data, sizeof(f64), Nlambda * Nspace, data);
    bg->sca = store->alloc2d(Nlambda, Nspace);
    fread(bg->sca.data, sizeof(f64), Nlambda * Nspace, data);
}


int main(void)
{
    ArenaAllocator dataStore(8LL * 1024LL * 1024LL * 1024LL);
    Atmosphere atmos{};
    init_atmos(&dataStore, &atmos, "FalAtmosBlob");
    Spectrum spect{};
    init_spect(&dataStore, &spect, atmos, "FalSpectBlob");

    bool detailed;
    std::vector<Atom*> activeAtoms;
    init_atoms(&dataStore, &activeAtoms, &atmos,
               spect, "FalActiveAtomsBlob", detailed=false);
    std::vector<Atom*> detailedAtoms;
    init_atoms(&dataStore, &activeAtoms, &atmos,
               spect, "FalDetailedAtomsBlob", detailed=true);

    Background bg{};
    init_background(&dataStore, &bg, "FalBackgroundBlob");

    FormalSolverManager fsManager;

    Context ctx{};
    ctx.atmos = &atmos;
    ctx.spect = &spect;
    ctx.background = &bg;
    ctx.activeAtoms = activeAtoms;
    ctx.detailedAtoms = detailedAtoms;
    ctx.Nthreads = 4;
    // ctx.formalSolver = fsManager.formalSolvers[0]; // linear
    ctx.formalSolver = fsManager.formalSolvers[1]; // Bezier
    ctx.initialise_threads();
    ctx.depthData = nullptr;

    int Nscatter = 10;
    for (int i = 0; i < 500; ++i)
    {
        f64 dJ = formal_sol_gamma_matrices(ctx);
        // printf("%.2e\n", dJ);
    }


    return 0;
}
