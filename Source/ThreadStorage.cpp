#include "ThreadStorage.hpp"
#include "Lightweaver.hpp"

namespace LwInternal
{

TransitionStorageFactory::TransitionStorageFactory(Transition* t) : trans(t)
{}

Transition* TransitionStorageFactory::copy_transition()
{
    if (!trans)
        return nullptr;

    tStorage.emplace_back(std::make_unique<TransitionStorage>());
    auto& ts = *tStorage.back().get();
    Transition* t = &ts.trans;

    ts.Rij = F64Arr(0.0, trans->Rij.shape(0));
    t->Rij = ts.Rij;
    ts.Rji = F64Arr(0.0, trans->Rji.shape(0));
    t->Rji = ts.Rji;

    t->type = trans->type;
    t->Nblue = trans->Nblue;
    t->i = trans->i;
    t->j = trans->j;
    t->polarised = trans->polarised;
    t->wavelength = trans->wavelength;
    t->lambda0 = trans->lambda0;
    t->dopplerWidth = trans->dopplerWidth;
    t->active = trans->active;
    
    if (t->type == LINE)
    {
        t->Aji = trans->Aji;
        t->Bji = trans->Bji;
        t->Bij = trans->Bij;
        t->Qelast = trans->Qelast;
        t->aDamp = trans->aDamp;
        t->phi = trans->phi;
        t->wphi = trans->wphi;
        t->phiQ = trans->phiQ;
        t->phiU = trans->phiU;
        t->phiV = trans->phiV;
        t->psiQ = trans->psiQ;
        t->psiU = trans->psiU;
        t->psiV = trans->psiV;

        if (trans->rhoPrd)
        {
            t->rhoPrd = trans->rhoPrd;
            if (trans->gII)
                t->gII = trans->gII;

            if (trans->hPrdCoeffs)
                t->hPrdCoeffs = trans->hPrdCoeffs;
        }

    }
    else
    {
        t->alpha = trans->alpha;
    }

    return t;
}

void TransitionStorageFactory::erase(Transition* trans)
{
    auto storageEntry = std::find_if(std::begin(tStorage), 
                                     std::end(tStorage),
                                    [trans](const auto& other)
                                    {
                                        return trans == &other->trans;
                                    });
    if (storageEntry == std::end(tStorage))
        return;

    tStorage.erase(storageEntry);
}

void TransitionStorageFactory::accumulate_rates()
{
    const int Nspace = trans->Rij.shape(0);
    trans->zero_rates();
    for (const auto& t : tStorage)
    {
        for (int k = 0; k < Nspace; ++k)
        {
            trans->Rij(k) += t->Rij(k);
            trans->Rji(k) += t->Rji(k);
        }
    }
}

void TransitionStorageFactory::accumulate_rates(const std::vector<size_t>& indices)
{
    const int Nspace = trans->Rij.shape(0);
    trans->zero_rates();
    for (auto i : indices)
    { 
        const auto& t = tStorage[i];
        for (int k = 0; k < Nspace; ++k)
        {
            trans->Rij(k) += t->Rij(k);
            trans->Rji(k) += t->Rji(k);
        }
    }
}

void TransitionStorageFactory::accumulate_prd_rates()
{
    if (!trans->rhoPrd)
        return;

    accumulate_rates();
}

void TransitionStorageFactory::accumulate_prd_rates(const std::vector<size_t>& indices)
{
    if (!trans->rhoPrd)
        return;

    accumulate_rates(indices);
}

AtomStorageFactory::AtomStorageFactory(Atom* a, bool detail) 
    : atom(a),         
      detailedStatic(detail)
{
    tStorage.reserve(atom->trans.size());
    for (auto t : atom->trans)
        tStorage.emplace_back(TransitionStorageFactory(t));
}

Atom* AtomStorageFactory::copy_atom()
{
    if (!atom)
        return nullptr;

    aStorage.emplace_back(std::make_unique<AtomStorage>());
    auto& as = *aStorage.back().get();
    Atom* a = &as.atom;
    a->atmos = atom->atmos;
    a->n = atom->n;
    a->nStar = atom->nStar;
    a->vBroad = atom->vBroad;
    a->nTotal = atom->nTotal;
    a->C = atom->C;
    a->Nlevel = atom->Nlevel;
    a->Ntrans = atom->Ntrans;
    const int Nlevel = a->Nlevel;
    const int Nspace = a->atmos->Nspace;

    if (a->Ntrans > 0)
    {
        as.gij = F64Arr2D(0.0, a->Ntrans, Nspace);
        a->gij = as.gij;
        as.wla = F64Arr2D(0.0, a->Ntrans, Nspace);
        a->wla = as.wla;
    }

    a->trans.reserve(a->Ntrans);
    for (auto& t : tStorage)
        a->trans.emplace_back(t.copy_transition());

    if (detailedStatic)
        return a;

    as.Gamma = F64Arr3D(0.0, Nlevel, Nlevel, Nspace);
    a->Gamma = as.Gamma;
    as.V = F64Arr2D(0.0, Nlevel, Nspace);
    a->V = as.V;
    as.U = F64Arr2D(0.0, Nlevel, Nspace);
    a->U = as.U;
    as.eta = F64Arr(0.0, Nspace);
    a->eta = as.eta;
    as.chi = F64Arr2D(0.0, Nlevel, Nspace);
    a->chi = as.chi;

    return a;
}

void AtomStorageFactory::erase(Atom* atom)
{
    auto storageEntry = std::find_if(std::begin(aStorage), 
                                     std::end(aStorage),
                                    [atom](const auto& other)
                                    {
                                        return atom == &other->atom;
                                    });
    if (storageEntry == std::end(aStorage))
        return;

    for (auto& t : tStorage)
    {
        auto& transToDelete = storageEntry->get()->atom.trans;
        for (auto& tt : transToDelete)
            t.erase(tt);
    }
    aStorage.erase(storageEntry);
}

void AtomStorageFactory::accumulate_Gamma_rates()
{
    for (auto& t : tStorage)
        t.accumulate_rates();

    if (detailedStatic)
        return;

    for (const auto& a : aStorage)
        for (int i = 0; i < atom->Nlevel; ++i)
            for (int j = 0; j < atom->Nlevel; ++j)
                for (int k = 0; k < atom->atmos->Nspace; ++k)
                    atom->Gamma(i, j, k) += a->Gamma(i, j, k);

}

void AtomStorageFactory::accumulate_Gamma_rates(const std::vector<size_t>& indices)
{
    for (auto& t : tStorage)
        t.accumulate_rates(indices);

    if (detailedStatic)
        return;

    for (auto& i : indices)
    {
        auto& a = aStorage[i];
        for (int i = 0; i < atom->Nlevel; ++i)
            for (int j = 0; j < atom->Nlevel; ++j)
                for (int k = 0; k < atom->atmos->Nspace; ++k)
                    atom->Gamma(i, j, k) += a->Gamma(i, j, k);
    }

}

void AtomStorageFactory::accumulate_prd_rates()
{
    for (auto& t : tStorage)
        t.accumulate_prd_rates();
}

void AtomStorageFactory::accumulate_prd_rates(const std::vector<size_t>& indices)
{
    for (auto& t : tStorage)
        t.accumulate_prd_rates(indices);
}

void IntensityCoreFactory::initialise(Context* ctx)
{
    atmos = ctx->atmos;
    spect = ctx->spect;
    background = ctx->background;
    if (ctx->Nthreads <= 1)
        return;

    bool detailedStatic = false;
    activeAtoms.reserve(ctx->activeAtoms.size());
    for (auto a : ctx->activeAtoms)
    {
        activeAtoms.emplace_back(AtomStorageFactory(a, detailedStatic=false));
    }

    detailedAtoms.reserve(ctx->detailedAtoms.size());
    for (auto a : ctx->detailedAtoms)
    {
        detailedAtoms.emplace_back(AtomStorageFactory(a, detailedStatic=true));
    }
}

IntensityCoreData* IntensityCoreFactory::new_intensity_core(bool psiOperator)
{
    const int Nspace = atmos->Nspace;
    arrayStorage.emplace_back(std::make_unique<IntensityCoreStorage>(Nspace));
    auto& as = *arrayStorage.back();
    auto& fd = as.formal;
    auto& iCore = as.core;

    JasPack(iCore, atmos, spect, background);
    iCore.fd = &fd;
    fd.atmos = atmos;
    fd.chi = as.chiTot;
    fd.S = as.S;
    fd.I = as.I;
    if (psiOperator)
        fd.Psi = as.PsiStar;

    iCore.JDag = &as.JDag;
    iCore.chiTot = as.chiTot;
    iCore.etaTot = as.etaTot;
    iCore.Uji = as.Uji;
    iCore.Vij = as.Vij;
    iCore.Vji = as.Vji;
    iCore.S = as.S;
    iCore.I = as.I;
    iCore.Ieff = as.Ieff;
    if (psiOperator)
        iCore.PsiStar = as.PsiStar;

    as.activeAtoms.reserve(activeAtoms.size());
    for (auto& atom : activeAtoms)
    {
        as.activeAtoms.emplace_back(atom.copy_atom());
    }
    iCore.activeAtoms = &as.activeAtoms;
    
    as.detailedAtoms.reserve(detailedAtoms.size());
    for (auto& atom : detailedAtoms)
        as.detailedAtoms.emplace_back(atom.copy_atom());
    iCore.detailedAtoms = &as.detailedAtoms;

    return &iCore;
}

void IntensityCoreFactory::erase(IntensityCoreData* core)
{
    auto storageEntry = std::find_if(std::begin(arrayStorage), 
                                     std::end(arrayStorage),
                                    [core](const auto& other)
                                    {
                                        return core == &other->core;
                                    });
    if (storageEntry == std::end(arrayStorage))
        return;

    for (auto& a : activeAtoms)
    {
        auto& atomsToDelete = storageEntry->get()->activeAtoms;
        for (auto& aa : atomsToDelete)
            a.erase(aa);
    }
    for (auto& a : detailedAtoms)
    {
        auto& atomsToDelete = storageEntry->get()->detailedAtoms;
        for (auto& aa : atomsToDelete)
            a.erase(aa);
    }

    arrayStorage.erase(storageEntry);
}

void IntensityCoreFactory::accumulate_Gamma_rates()
{
    for (auto& a : activeAtoms)
        a.accumulate_Gamma_rates();
    for (auto& a : detailedAtoms)
        a.accumulate_Gamma_rates();
}

void IntensityCoreFactory::accumulate_Gamma_rates(const std::vector<size_t>& indices)
{
    for (auto& a : activeAtoms)
        a.accumulate_Gamma_rates(indices);
    for (auto& a : detailedAtoms)
        a.accumulate_Gamma_rates(indices);
}

void IntensityCoreFactory::accumulate_prd_rates()
{
    for (auto& a : activeAtoms)
        a.accumulate_prd_rates();
}

void IntensityCoreFactory::accumulate_prd_rates(const std::vector<size_t>& indices)
{
    for (auto& a : activeAtoms)
        a.accumulate_prd_rates(indices);
}

void IterationCores::initialise(IntensityCoreFactory* fac, int Nthreads)
{
    factory = fac;
    cores.reserve(Nthreads);
    indices.reserve(Nthreads);
    for (int t = 0; t < Nthreads; ++t)
    {
        cores.emplace_back(factory->new_intensity_core(true));
        indices.emplace_back(factory->arrayStorage.size()-1);
    }
}

IterationCores::~IterationCores()
{
    if (!factory)
        return;

    for (const auto& c : cores)
        factory->erase(c);
}

void IterationCores::accumulate_Gamma_rates()
{
    factory->accumulate_Gamma_rates(indices);
}

void IterationCores::accumulate_prd_rates()
{
    factory->accumulate_prd_rates(indices);
}

void ThreadData::initialise(Context* ctx)
{
    threadDataFactory.initialise(ctx);
    if (ctx->Nthreads <= 1)
        return;

    if (schedMemory)
        assert(false && "Tried to re initialise_threads for a Context");

    sched_size memNeeded;
    scheduler_init(&sched, &memNeeded, ctx->Nthreads, nullptr);
    schedMemory = calloc(memNeeded, 1);
    scheduler_start(&sched, schedMemory);

    intensityCores.initialise(&threadDataFactory, ctx->Nthreads);
}

}