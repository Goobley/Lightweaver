#include "ThreadStorage.hpp"
#include "Lightweaver.hpp"
#include <algorithm>

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

void AtomStorageFactory::accumulate_Gamma_rates_parallel(scheduler* s)
{
    const int Njobs = tStorage.size();
    auto acc_task = [](void* userdata, scheduler* s,
                     sched_task_partition p, sched_uint threadId)
    {
        for (int j = p.start; j < p.end; ++j)
        {
            auto& data = ((TransitionStorageFactory*)userdata)[j];
            data.accumulate_rates();
        }
    };


    {
        sched_task accumulation;
        scheduler_add(s, &accumulation, acc_task, (void*)tStorage.data(), Njobs, 1);
        if (!detailedStatic)
        {
            for (auto& a : aStorage)
            {
                for (int i = 0; i < atom->Nlevel; ++i)
                    for (int j = 0; j < atom->Nlevel; ++j)
                        for (int k = 0; k < atom->atmos->Nspace; ++k)
                            atom->Gamma(i, j, k) += a->Gamma(i, j, k);
            }
        }
        scheduler_join(s, &accumulation);
    }

}

void AtomStorageFactory::accumulate_Gamma_rates_parallel(scheduler* s,
                                                         const std::vector<size_t>& indices)
{
    struct AccData
    {
        TransitionStorageFactory* trans;
        const std::vector<size_t>& indices;
    };
    const int Njobs = tStorage.size();
    std::vector<AccData> taskData;
    taskData.reserve(Njobs);
    for (int j = 0; j < Njobs; ++j)
        taskData.emplace_back(AccData{&tStorage[j], indices});

    auto acc_task = [](void* userdata, scheduler* s,
                     sched_task_partition p, sched_uint threadId)
    {
        for (int j = p.start; j < p.end; ++j)
        {
            auto& data = ((AccData*)userdata)[j];
            data.trans->accumulate_rates(data.indices);
        }
    };


    {
        sched_task accumulation;
        scheduler_add(s, &accumulation, acc_task, (void*)taskData.data(), Njobs, 1);
        if (!detailedStatic)
        {
            for (auto& i : indices)
            {
                auto& a = aStorage[i];
                for (int i = 0; i < atom->Nlevel; ++i)
                    for (int j = 0; j < atom->Nlevel; ++j)
                        for (int k = 0; k < atom->atmos->Nspace; ++k)
                            atom->Gamma(i, j, k) += a->Gamma(i, j, k);
            }
        }
        scheduler_join(s, &accumulation);
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
    depthData = ctx->depthData;
    formal_solver = ctx->formalSolver.solver;
    interp = ctx->interpFn;
    // if (ctx->Nthreads <= 1)
    //     return;

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

    JasPack(iCore, atmos, spect, background, depthData);
    iCore.fd = &fd;
    fd.atmos = atmos;
    fd.chi = as.chiTot;
    fd.S = as.S;
    fd.I = as.I;
    fd.interp = interp.interp_2d;
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

    iCore.formal_solver = formal_solver;

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

void IntensityCoreFactory::accumulate_Gamma_rates_parallel(Context& ctx)
{
    struct AccData
    {
        AtomStorageFactory* atom;
    };
    std::vector<AccData> taskData;
    const int Njobs = activeAtoms.size() + detailedAtoms.size();
    taskData.reserve(Njobs);
    for (int j = 0; j < activeAtoms.size(); ++j)
        taskData.emplace_back(AccData{&activeAtoms[j]});
    for (int j = 0; j < detailedAtoms.size(); ++j)
        taskData.emplace_back(AccData{&detailedAtoms[j]});

    auto acc_task = [](void* userdata, scheduler* s,
                       sched_task_partition p, sched_uint threadId)
    {
        for (i64 j = p.start; j < p.end; ++j)
        {
            auto& data = ((AccData*)userdata)[j];
            data.atom->accumulate_Gamma_rates_parallel(s);
        }
    };

    {
        sched_task accumulation;
        scheduler_add(&ctx.threading.sched, &accumulation, acc_task,
                      (void*)taskData.data(), Njobs, 1);
        scheduler_join(&ctx.threading.sched, &accumulation);
    }
}

void IntensityCoreFactory::accumulate_Gamma_rates_parallel(Context& ctx,
                                                           const std::vector<size_t>& indices)
{
    struct AccData
    {
        AtomStorageFactory* atom;
        const std::vector<size_t>& indices;
    };
    std::vector<AccData> taskData;
    const int Njobs = activeAtoms.size() + detailedAtoms.size();
    taskData.reserve(Njobs);
    for (int j = 0; j < activeAtoms.size(); ++j)
        taskData.emplace_back(AccData{&activeAtoms[j], indices});
    for (int j = 0; j < detailedAtoms.size(); ++j)
        taskData.emplace_back(AccData{&detailedAtoms[j], indices});

    auto acc_task = [](void* userdata, scheduler* s,
                       sched_task_partition p, sched_uint threadId)
    {
        for (i64 j = p.start; j < p.end; ++j)
        {
            auto& data = ((AccData*)userdata)[j];
            data.atom->accumulate_Gamma_rates_parallel(s, data.indices);
        }
    };

    {
        sched_task accumulation;
        scheduler_add(&ctx.threading.sched, &accumulation, acc_task,
                      (void*)taskData.data(), Njobs, 1);
        scheduler_join(&ctx.threading.sched, &accumulation);
    }
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

void IntensityCoreFactory::clear()
{
    arrayStorage.clear();
    detailedAtoms.clear();
    activeAtoms.clear();
    atmos = nullptr;
    spect = nullptr;
    background = nullptr;
    depthData = nullptr;
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

void IterationCores::accumulate_Gamma_rates_parallel(Context& ctx)
{
    factory->accumulate_Gamma_rates_parallel(ctx, indices);
}

void IterationCores::accumulate_prd_rates()
{
    factory->accumulate_prd_rates(indices);
}

void IterationCores::clear()
{
    cores.clear();
    indices.clear();
    factory = nullptr;
}

void ThreadData::initialise(Context* ctx)
{
    threadDataFactory.initialise(ctx);
    if (ctx->Nthreads <= 1)
        return;

    if (schedMemory)
    {
        throw std::runtime_error("Tried to re- initialise_threads for a Context");
    }

    sched_size memNeeded;
    scheduler_init(&sched, &memNeeded, ctx->Nthreads, nullptr);
    schedMemory = calloc(memNeeded, 1);
    scheduler_start(&sched, schedMemory);

    intensityCores.initialise(&threadDataFactory, ctx->Nthreads);
    for (Atom* a : ctx->activeAtoms)
    {
        for (Transition* t : a->trans)
        {
            if (t->type == TransitionType::LINE)
            {
                t->bound_parallel_compute_phi = [this, t](const Atmosphere& atmos,
                                                       F64View aDamp, F64View vBroad)
                {
                    t->compute_phi_parallel(this, atmos, aDamp, vBroad);
                };
            }
        }
    }
    for (Atom* a : ctx->detailedAtoms)
    {
        for (Transition* t : a->trans)
        {
            if (t->type == TransitionType::LINE)
            {
                t->bound_parallel_compute_phi = [this, t](const Atmosphere& atmos,
                                                       F64View aDamp, F64View vBroad)
                {
                    t->compute_phi_parallel(this, atmos, aDamp, vBroad);
                };
            }
        }
    }
}

void ThreadData::clear(Context* ctx)
{
    for (Atom* a : ctx->activeAtoms)
    {
        for (Transition* t : a->trans)
        {
            if (t->type == TransitionType::LINE)
            {
                t->bound_parallel_compute_phi = std::function<void(const Atmosphere&, F64View, F64View)>();
            }
        }
    }
    for (Atom* a : ctx->detailedAtoms)
    {
        for (Transition* t : a->trans)
        {
            if (t->type == TransitionType::LINE)
            {
                t->bound_parallel_compute_phi = std::function<void(const Atmosphere&, F64View, F64View)>();
            }
        }
    }

    if (schedMemory)
    {
        scheduler_stop(&sched, 1);
        free(schedMemory);
        schedMemory = nullptr;
    }
    intensityCores.clear();
    threadDataFactory.clear();
}

}