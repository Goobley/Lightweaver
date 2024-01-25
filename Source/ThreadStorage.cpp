#include "ThreadStorage.hpp"
#include "Lightweaver.hpp"
#include <algorithm>

namespace LwInternal
{

TransitionStorageFactory::TransitionStorageFactory(Transition* t,
                                                   PerTransFns perFns)
    : trans(t),
      methodFns(perFns)
{}

Transition* TransitionStorageFactory::copy_transition()
{
    if (!trans)
        return nullptr;

    tStorage.emplace_back(std::make_unique<TransitionStorage>());
    auto& ts = *tStorage.back().get();
    Transition* t = &ts.trans;
    ts.free_method_scratch = methodFns.free_per;

    ts.Rij = F64Arr(0.0, trans->Rij.shape(0));
    t->Rij = ts.Rij;
    ts.Rji = F64Arr(0.0, trans->Rji.shape(0));
    t->Rji = ts.Rji;

    t->type = trans->type;
    t->Nblue = trans->Nblue;
    t->Nred = trans->Nred;
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
            t->rhoPrd = trans->rhoPrd;
        t->hPrdCoeffs = trans->hPrdCoeffs;
        t->prdData = &trans->prdStorage;

    }
    else
    {
        t->alpha = trans->alpha;
    }

    if (methodFns.alloc_per)
        methodFns.alloc_per(t);

    return t;
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
        }
        for (int k = 0; k < Nspace; ++k)
        {
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

AtomStorageFactory::AtomStorageFactory(Atom* a, bool detail, bool wlaStorage,
                                       bool defaultAtomStorage,
                                       int fsWidthSimd, PerAtomTransFns perFns)
    : atom(a),
      detailedStatic(detail),
      wlaGijStorage(wlaStorage),
      defaultPerAtomStorage(defaultAtomStorage),
      fsWidth(fsWidthSimd),
      methodFns(perFns.perAtom)
{
    tStorage.reserve(atom->trans.size());
    for (auto t : atom->trans)
        tStorage.emplace_back(TransitionStorageFactory(t, perFns.perTrans));
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

    a->trans.reserve(a->Ntrans);
    for (auto& t : tStorage)
        a->trans.emplace_back(t.copy_transition());

    if ((!detailedStatic) && defaultPerAtomStorage)
    {
        as.Gamma = F64Arr3D(0.0, Nlevel, Nlevel, Nspace);
        a->Gamma = as.Gamma;
    }

    a->init_scratch(Nspace, detailedStatic, wlaGijStorage, defaultPerAtomStorage);

    if (methodFns.alloc_per)
        methodFns.alloc_per(a, detailedStatic);

    return a;
}

void AtomStorageFactory::accumulate_Gamma()
{
    if (detailedStatic || !defaultPerAtomStorage)
        return;

    // NOTE(cmo): Main thread gamma isn't zero'd in this function because it's
    // pre-filled with C in the middle layer.
    auto mainGammaFlat = atom->Gamma.flatten();
    for (const auto& a : aStorage)
    {
        auto gammaFlat = a->Gamma.flatten();
        // NOTE(cmo): I'm just going to flatten this and put my faith in the
        // compiler.
        for (i64 i = 0; i < mainGammaFlat.shape(0); ++i)
            mainGammaFlat(i) += gammaFlat(i);
    }
}

void AtomStorageFactory::accumulate_Gamma_rates()
{
    for (auto& t : tStorage)
        t.accumulate_rates();

    accumulate_Gamma();
}

void AtomStorageFactory::accumulate_rates()
{
    for (auto& t : tStorage)
        t.accumulate_rates();
}

void AtomStorageFactory::accumulate_prd_rates()
{
    for (auto& t : tStorage)
        t.accumulate_prd_rates();
}

void IntensityCoreFactory::initialise(Context* ctx)
{
    atmos = ctx->atmos;
    spect = ctx->spect;
    background = ctx->background;
    depthData = ctx->depthData;
    fsWidth = ctx->formalSolver.width;
    formal_solver = ctx->formalSolver.solver;
    interp = ctx->interpFn;

    PerAtomTransFns methodScratchFns{ PerAtomFns  { ctx->iterFns.alloc_per_atom,
                                                    ctx->iterFns.free_per_atom },
                                      PerTransFns { ctx->iterFns.alloc_per_trans,
                                                    ctx->iterFns.free_per_trans } };

    bool detailedStatic = false;
    bool wlaGijStorage = ctx->iterFns.defaultWlaGijStorage;
    bool defaultPerAtomStorage = ctx->iterFns.defaultPerAtomStorage;
    activeAtoms.reserve(ctx->activeAtoms.size());
    for (auto a : ctx->activeAtoms)
    {
        activeAtoms.emplace_back(AtomStorageFactory(a, detailedStatic, wlaGijStorage,
                                  defaultPerAtomStorage,
                                  ctx->formalSolver.width, methodScratchFns));
    }

    detailedAtoms.reserve(ctx->detailedAtoms.size());
    detailedStatic = true;
    for (auto a : ctx->detailedAtoms)
    {
        detailedAtoms.emplace_back(AtomStorageFactory(a, detailedStatic, wlaGijStorage,
                                    defaultPerAtomStorage,
                                    ctx->formalSolver.width, methodScratchFns));
    }
}

IntensityCoreData* IntensityCoreFactory::single_thread_intensity_core()
{
    const int Nspace = atmos->Nspace;
    arrayStorage.emplace_back(std::make_unique<IntensityCoreStorage>(Nspace, 0));
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
    iCore.PsiStar = as.PsiStar;
    iCore.JRest = spect->JRest;

    as.activeAtoms.reserve(activeAtoms.size());
    for (auto& atom : activeAtoms)
    {
        as.activeAtoms.emplace_back(atom.atom);
    }
    iCore.activeAtoms = &as.activeAtoms;

    as.detailedAtoms.reserve(detailedAtoms.size());
    for (auto& atom : detailedAtoms)
        as.detailedAtoms.emplace_back(atom.atom);
    iCore.detailedAtoms = &as.detailedAtoms;

    iCore.formal_solver = formal_solver;

    return &iCore;
}

IntensityCoreData* IntensityCoreFactory::new_intensity_core()
{
    const int Nspace = atmos->Nspace;
    const int NhPrd = If spect->JRest Then spect->JRest.shape(0) Else 0 End;
    arrayStorage.emplace_back(std::make_unique<IntensityCoreStorage>(Nspace, NhPrd));
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
    iCore.PsiStar = as.PsiStar;
    iCore.JRest = as.JRest;

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

void IntensityCoreFactory::accumulate_JRest()
{
    if (!spect->JRest || arrayStorage.size() == 1)
        return;

    auto JRestFlat = spect->JRest.flatten();
    JRestFlat.fill(0.0);
    for (auto& iCore : arrayStorage)
    {
        auto JRestOther = iCore->JRest.flatten();
        for (int i = 0; i < JRestFlat.shape(0); ++i)
        {
            JRestFlat(i) += JRestOther(i);
        }
    }
}

void IntensityCoreFactory::accumulate_Gamma_rates()
{
    for (auto& a : activeAtoms)
        a.accumulate_Gamma_rates();
    for (auto& a : detailedAtoms)
        a.accumulate_Gamma_rates();
    accumulate_JRest();
}

void IntensityCoreFactory::accumulate_Gamma_rates_parallel(Context& ctx)
{
    struct AccData
    {
        AtomStorageFactory* atom;
        int mode;
    };
    std::vector<AccData> taskData;
    taskData.reserve(2 * (activeAtoms.size() + detailedAtoms.size()));
    for (int j = 0; j < activeAtoms.size(); ++j)
    {
        taskData.emplace_back(AccData{ &activeAtoms[j], 0 });
        taskData.emplace_back(AccData{ &activeAtoms[j], 1});
    }
    for (int j = 0; j < detailedAtoms.size(); ++j)
    {
        taskData.emplace_back(AccData{ &detailedAtoms[j], 0 });
    }

    auto acc_task = [](void* userdata, enki::TaskScheduler* s,
                       enki::TaskSetPartition p, u32 threadId)
    {
        for (i64 j = p.start; j < p.end; ++j)
        {
            auto& data = ((AccData*)userdata)[j];
            switch (data.mode)
            {
                case 0:
                {
                    data.atom->accumulate_rates();
                } break;

                case 1:
                {
                    data.atom->accumulate_Gamma();
                } break;

                default:
                {
                    assert(false);
                }
            }
        }
    };

    {
        enki::TaskScheduler* sched = &ctx.threading.sched;
        LwTaskSet accumulation(taskData.data(), sched, taskData.size(),
                               1, acc_task);
        sched->AddTaskSetToPipe(&accumulation);
        accumulate_JRest();
        sched->WaitforTask(&accumulation);
    }
}

void IntensityCoreFactory::accumulate_prd_rates(bool includeDetailedAtoms)
{
    for (auto& a : activeAtoms)
        a.accumulate_prd_rates();
    if (includeDetailedAtoms)
    {
        for (auto& a : detailedAtoms)
            a.accumulate_prd_rates();
    }
    accumulate_JRest();
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
    if (Nthreads == 1)
    {
        cores.emplace_back(factory->single_thread_intensity_core());
        indices.emplace_back(factory->arrayStorage.size()-1);
    }
    else
    {
        for (int t = 0; t < Nthreads; ++t)
        {
            cores.emplace_back(factory->new_intensity_core());
            indices.emplace_back(factory->arrayStorage.size()-1);
        }
    }
}

IterationCores::~IterationCores()
{
    if (!factory)
        return;

    factory->clear();
}

void IterationCores::accumulate_Gamma_rates()
{
    factory->accumulate_Gamma_rates();
}

void IterationCores::accumulate_Gamma_rates_parallel(Context& ctx)
{
    // NOTE(cmo): Very approximate check to see if it's worth threading.
    if ((ctx.atmos->Nspace <= 1024)
        || (ctx.activeAtoms.size() + ctx.detailedAtoms.size()) < 8)
    {
        factory->accumulate_Gamma_rates();
    }
    else
    {
        factory->accumulate_Gamma_rates_parallel(ctx);
    }
}

void IterationCores::accumulate_prd_rates(bool includeDetailedAtoms)
{
    factory->accumulate_prd_rates(includeDetailedAtoms);
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

    if (ctx->iterFns.alloc_global_scratch)
    {
        ctx->iterFns.alloc_global_scratch(ctx);
        clear_global_scratch = [ctx](){
            ctx->iterFns.free_global_scratch(ctx);
            // NOTE(cmo): Clear self after a call to avoid any double free
            // behaviour
            ctx->threading.clear_global_scratch = {};
        };
    }

    // NOTE(cmo): Allocate a core even for single threaded use.
    intensityCores.initialise(&threadDataFactory, ctx->Nthreads);

    if (ctx->Nthreads <= 1)
        return;

    if (sched.GetNumTaskThreads() > 0)
    {
        throw std::runtime_error("Tried to re- initialise_threads for a Context");
    }

    sched.Initialize(ctx->Nthreads);

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

    sched.WaitforAllAndShutdown();
    if (clear_global_scratch)
        clear_global_scratch();
    intensityCores.clear();
    threadDataFactory.clear();
}

}
