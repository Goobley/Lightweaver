#include "Lightweaver.hpp"
#include "TaskSetWrapper.hpp"
#include <atomic>
#include <list>
#include <stdexcept>

void stat_eq_impl(Atom* atomIn, ExtraParams params,
                  int spaceStart, int spaceEnd)
{
    auto& atom = *atomIn;
    const int Nlevel = atom.Nlevel;
    if (spaceStart < 0 && spaceEnd < 0)
    {
        spaceStart = 0;
        spaceEnd = atom.n.shape(1);
    }


    auto nk = F64Arr(Nlevel);
    auto Gamma = F64Arr2D(Nlevel, Nlevel);

    for (int k = spaceStart; k < spaceEnd; ++k)
    {
        for (int i = 0; i < Nlevel; ++i)
        {
            nk(i) = atom.n(i, k);
            for (int j = 0; j < Nlevel; ++j)
                Gamma(i, j) = atom.Gamma(i, j, k);
        }

        int iEliminate = 0;
        f64 nMax = 0.0;
        for (int i = 0; i < Nlevel; ++i)
            nMax = max_idx(nMax, nk(i), iEliminate, i);

        for (int i = 0; i < Nlevel; ++i)
        {
            Gamma(iEliminate, i) = 1.0;
            nk(i) = 0.0;
        }
        nk(iEliminate) = atom.nTotal(k);

        solve_lin_eq(Gamma, nk);
        for (int i = 0; i < Nlevel; ++i)
            atom.n(i, k) = nk(i);
    }
}

void stat_eq(Context& ctx,  Atom* atom, ExtraParams params,
             int spaceStart, int spaceEnd) 
{
    return ctx.iterFns.stat_eq(atom, params, spaceStart, spaceEnd);
}

void parallel_stat_eq(Context* ctx, int chunkSize, ExtraParams params)
{
    const int Natom = ctx->activeAtoms.size();
    // NOTE(cmo): Run single threaded if chunkSize is 0
    if (chunkSize <= 0 || ctx->atmos->Nspace <= chunkSize)
    {
        for (Atom* atom : ctx->activeAtoms)
        {
            stat_eq_impl(atom, params);
        }
        return;
    }

    struct UpdateData
    {
        Atom* atom;
        ExtraParams* params;
        std::atomic<bool> exceptionThrown;
    };

    std::vector<UpdateData> threadData = std::vector<UpdateData>(Natom);
    std::list<LwTaskSet> atomTasks; // NOTE(cmo): This type is only move-constructible with a throwing constructor, so list is much easier to work with.
    for (int a = 0; a < Natom; ++a)
    {
        threadData[a].atom = ctx->activeAtoms[a];
        threadData[a].exceptionThrown = false;
        threadData[a].params = &params;
    }


    auto stat_eq_handler = [](void* data, enki::TaskScheduler* s,
                              enki::TaskSetPartition p, u32 threadId)
    {
        UpdateData* d = (UpdateData*)data;
        try
        {
            stat_eq_impl(d->atom, *d->params, p.start, p.end);
        }
        catch (const std::runtime_error& e)
        {
            d->exceptionThrown = true;
        }
    };

    enki::TaskScheduler* s = &ctx->threading.sched;
    for (int a = 0; a < Natom; ++a)
    {
        atomTasks.emplace_back(&threadData[a], s, ctx->atmos->Nspace,
                               chunkSize, stat_eq_handler);
    }

    for (auto& task : atomTasks)
        s->AddTaskSetToPipe(&task);
    for (auto& task : atomTasks)
        s->WaitforTask(&task);

    bool throwNeeded = false;
    for (int a = 0; a < Natom; ++a)
        if (threadData[a].exceptionThrown)
            throwNeeded = true;

    if (throwNeeded)
        throw std::runtime_error("Singular Matrix");
}

void time_dependent_update_impl(Atom* atomIn, F64View2D nOld, f64 dt,
                                ExtraParams params, int spaceStart, int spaceEnd)
{
    auto& atom  = *atomIn;
    const int Nlevel = atom.Nlevel;
    if (spaceStart < 0 && spaceEnd < 0)
    {
        spaceStart = 0;
        spaceEnd = atom.n.shape(1);
    }

    auto nk = F64Arr(Nlevel);
    auto Gamma = F64Arr2D(Nlevel, Nlevel);

    for (int k = spaceStart; k < spaceEnd; ++k)
    {
        for (int i = 0; i < Nlevel; ++i)
        {
            nk(i) = nOld(i, k);
            for (int j = 0; j < Nlevel; ++j)
                Gamma(i, j) = -atom.Gamma(i, j, k) * dt;
            Gamma(i, i) = 1.0 - atom.Gamma(i, i, k) * dt;
        }

        solve_lin_eq(Gamma, nk);

        for (int i = 0; i < Nlevel; ++i)
        {
            atom.n(i, k) = nk(i);
        }
    }
}

void time_dependent_update(Context& ctx, Atom* atomIn, F64View2D nOld, f64 dt,
                           ExtraParams params, int spaceStart, int spaceEnd)
{
    return ctx.iterFns.time_dep_update(atomIn, nOld, dt, params, spaceStart, spaceEnd);
}

void parallel_time_dep_update(Context* ctx, const std::vector<F64View2D>& oldPops,
                              f64 dt, int chunkSize, ExtraParams params)
{
    const int Natom = ctx->activeAtoms.size();
    // NOTE(cmo): Run single threaded if chunkSize is 0
    if (chunkSize <= 0 || ctx->atmos->Nspace <= chunkSize)
    {
        for (int a = 0; a < Natom; ++a)
        {
            time_dependent_update_impl(ctx->activeAtoms[a], oldPops[a], dt, params);
        }
        return;
    }

    struct UpdateData
    {
        Atom* atom;
        F64View2D nOld;
        f64 dt;
        ExtraParams* params;
        std::atomic<bool> exceptionThrown;
    };

    std::vector<UpdateData> threadData = std::vector<UpdateData>(Natom);
    std::list<LwTaskSet> atomTasks;
    for (int a = 0; a < Natom; ++a)
    {
        threadData[a].atom = ctx->activeAtoms[a];
        threadData[a].nOld = oldPops[a];
        threadData[a].dt = dt;
        threadData[a].params = &params;
        threadData[a].exceptionThrown = false;
    }

    auto update_handler = [](void* data, enki::TaskScheduler* s,
                             enki::TaskSetPartition p, u32 threadId)
    {
        UpdateData* d = (UpdateData*)data;
        try
        {
            time_dependent_update_impl(d->atom, d->nOld, d->dt, *d->params, p.start, p.end);
        }
        catch (const std::runtime_error& e)
        {
            d->exceptionThrown = true;
        }
    };

    enki::TaskScheduler* s = &ctx->threading.sched;
    for (int a = 0; a < Natom; ++a)
    {
        atomTasks.emplace_back(&threadData[a], s, ctx->atmos->Nspace,
                               chunkSize, update_handler);
    }

    for (auto& task : atomTasks)
        s->AddTaskSetToPipe(&task);
    for (auto& task : atomTasks)
        s->WaitforTask(&task);

    bool throwNeeded = false;
    for (int a = 0; a < Natom; ++a)
        if (threadData[a].exceptionThrown)
            throwNeeded = true;

    if (throwNeeded)
        throw std::runtime_error("Singular Matrix");
}

// NOTE(cmo): F and Ftd have opposite sign compared to what is shown in the
// paper, but they are inverted later in the nr_post_update
void F(int k, f64 ne, f64 backgroundNe, const std::vector<Atom*>& atoms, F64View F)
{
    const int Neqn = F.shape(0);
    int start = 0;
    F.fill(0.0);
    F(Neqn-1) = ne;
    for (int a = 0; a < atoms.size(); ++a)
    {
        const Atom* atom = atoms[a];
        for (int l = 0; l < atom->Nlevel; ++l)
        {
            F(start+l) = 0.0;
            for (int ll = 0; ll < atom->Nlevel; ++ll)
                F(start+l) -= atom->Gamma(l, ll, k) * atom->n(ll, k);
        }
        f64 nTotCur = 0.0;
        for (int ll = 0; ll < atom->Nlevel; ++ll)
            nTotCur += atom->n(ll, k);
        F(start + atom->Nlevel - 1) = nTotCur - atom->nTotal(k);
        f64 eleContrib = 0.0;
        for (int ll = 0; ll < atom->Nlevel; ++ll)
            eleContrib += atom->stages(ll) * atom->n(ll, k);
        F(Neqn - 1) -= eleContrib;
        start += atom->Nlevel;
    }
    F(Neqn - 1) -= backgroundNe;
}

void Ftd(int k, const NrTimeDependentData& timeDepData,
         f64 ne, f64 backgroundNe, const std::vector<Atom*>& atoms, F64View F)
{
    const int Neqn = F.shape(0);
    JasUnpack(timeDepData, dt, nPrev);
    int start = 0;
    F.fill(0.0);
    F(Neqn-1) = ne;
    const f64 theta = 1.0;
    for (int a = 0; a < atoms.size(); ++a)
    {
        const Atom* atom = atoms[a];
        for (int l = 0; l < atom->Nlevel; ++l)
        {
            F(start+l) = 0.0;
            for (int ll = 0; ll < atom->Nlevel; ++ll)
                F(start+l) += atom->Gamma(l, ll, k) * atom->n(ll, k);

            F(start+l) *= theta * dt;
            F(start+l) -= atom->n(l, k) - nPrev[a](l, k);
        }
        f64 nTotCur = 0.0;
        for (int ll = 0; ll < atom->Nlevel; ++ll)
            nTotCur += atom->n(ll, k);
        F(start + atom->Nlevel - 1) = nTotCur - atom->nTotal(k);
        f64 eleContrib = 0.0;
        for (int ll = 0; ll < atom->Nlevel; ++ll)
            eleContrib += atom->stages(ll) * atom->n(ll, k);
        F(Neqn - 1) -= eleContrib;
        start += atom->Nlevel;
    }
    F(Neqn - 1) -= backgroundNe;
}

void nr_post_update_impl(Context& ctx, std::vector<Atom*>* atoms,
                         const std::vector<F64View3D>& dC,
                         F64View backgroundNe,
                         const NrTimeDependentData& timeDepData,
                         f64 crswVal,
                         ExtraParams params,
                         int spaceStart, int spaceEnd)
{
    const bool fdCollisionRates = dC.size() > 0;
    const bool timeDep = (timeDepData.nPrev.size() != 0);
    int Nlevel = 0;
    for (Atom* atom : *atoms)
        Nlevel += atom->Nlevel;
    const int Neqn = Nlevel + 1;
    if (spaceStart < 0 && spaceEnd < 0)
    {
        spaceStart = 0;
        spaceEnd = ctx.atmos->Nspace;
    }

    F64Arr2D dF(0.0, Neqn, Neqn);
    F64Arr Fnew(0.0, Neqn);
    F64Arr Fg(0.0, Neqn);

    const f64 theta = 1.0;
    for (int k = spaceStart; k < spaceEnd; ++k)
    {
        dF.fill(0.0);
        if (timeDep)
            Ftd(k, timeDepData, ctx.atmos->ne(k), backgroundNe(k), *atoms, Fg);
        else
            F(k, ctx.atmos->ne(k), backgroundNe(k), *atoms, Fg);

        int start = 0;
        for (int a = 0; a < atoms->size(); ++a)
        {
            Atom* atom = (*atoms)[a];
            for (int l = 0; l < atom->Nlevel; ++l)
                for (int ll = 0; ll < atom->Nlevel; ++ll)
                    dF(start+l, start+ll) = -atom->Gamma(l, ll, k);

            if (timeDep)
            {
                for (int l = 0; l < atom->Nlevel; ++l)
                    for (int ll = 0; ll < atom->Nlevel; ++ll)
                        dF(start+l, start+ll) *= -theta * timeDepData.dt;
                for (int l = 0; l < atom->Nlevel; ++l)
                    dF(start+l, start+l) -= 1.0;
            }

            for (int tIdx = 0; tIdx < atom->Ntrans; ++tIdx)
            {
                Transition* t = atom->trans[tIdx];
                if (t->type == TransitionType::CONTINUUM)
                {
                    f64 preconRji = atom->Gamma(t->i, t->j, k) - crswVal * atom->C(t->i, t->j, k);
                    f64 entry = -(preconRji / ctx.atmos->ne(k)) * atom->n(t->j, k);
                    if (timeDep)
                        entry *= -theta * timeDepData.dt;
                    dF(start + t->i, Neqn-1) += entry;
                }
            }

            if (fdCollisionRates)
            {
                for (int i = 0; i < atom->Nlevel; ++i)
                {
                    f64 entry = 0.0;
                    for (int ll = 0; ll < atom->Nlevel; ++ll)
                        entry -= dC[a](i, ll, k) * atom->n(ll, k);
                    if (timeDep)
                        entry *= -theta * timeDepData.dt;
                    dF(start + i, Neqn-1) += entry;
                }
            }

            dF(start + atom->Nlevel-1).fill(0.0);
            for (int ll = 0; ll < atom->Nlevel; ++ll)
            {
                dF(start + atom->Nlevel-1, start+ll) = 1.0;
                dF(Neqn-1, start+ll) = -atom->stages(ll);
            }
            start += atom->Nlevel;
        }
        dF(Neqn-1, Neqn-1) = 1.0;
        for (int i = 0; i < Neqn; ++i)
            Fg(i) *= -1.0;

        solve_lin_eq(dF, Fg);

        start = 0;
        for (int a = 0; a < atoms->size(); ++a)
        {
            Atom* atom = (*atoms)[a];
            for (int ll = 0; ll < atom->Nlevel; ++ll)
            {
                atom->n(ll, k) += Fg(start + ll);
            }
            start += atom->Nlevel;
        }
        ctx.atmos->ne(k) += Fg(Neqn-1);
    }
}

void parallel_nr_post_update(Context& ctx, std::vector<Atom*>* atoms,
                             const std::vector<F64View3D>& dC,
                             F64View backgroundNe,
                             const NrTimeDependentData& timeDepData,
                             f64 crswVal,
                             int chunkSize, ExtraParams params)
{
    if (chunkSize <= 0 || ctx.atmos->Nspace <= chunkSize)
    {
        nr_post_update_impl(ctx, atoms, dC, backgroundNe, timeDepData, crswVal, params);
        return;
    }

    const int Nthreads = ctx.Nthreads;
    struct UpdateData
    {
        Context* ctx;
        std::vector<Atom*>* atoms;
        const std::vector<F64View3D>* dC;
        F64View backgroundNe;
        const NrTimeDependentData* timeDepData;
        f64 crswVal;
        ExtraParams* params;
        bool exceptionThrown;
    };

    // NOTE(cmo): Fine to pass pointers to local variables due to shared address
    // space, providing the data lives for the thread (or here, job) lifetime,
    // which we can guarantee.
    // https://docs.microsoft.com/en-us/windows/win32/procthread/creating-threads
    // POSIX guaranteed too https://stackoverflow.com/a/49795116/3847013o
    std::vector<UpdateData> threadData = std::vector<UpdateData>(Nthreads);
    for (int t = 0; t < Nthreads; ++t)
    {
        threadData[t].ctx = &ctx;
        threadData[t].atoms = atoms;
        threadData[t].dC = &dC;
        threadData[t].backgroundNe = backgroundNe;
        threadData[t].timeDepData = &timeDepData;
        threadData[t].crswVal = crswVal;
        threadData[t].params = &params;
        threadData[t].exceptionThrown = false;
    }

    auto update_handler = [](void* data, enki::TaskScheduler* s,
                             enki::TaskSetPartition p, u32 threadId)
    {
        UpdateData* d = &((UpdateData*)data)[threadId];
        try
        {
            nr_post_update_impl(*d->ctx, d->atoms, *d->dC, d->backgroundNe,
                           *d->timeDepData, d->crswVal, *d->params, p.start, p.end);
        }
        catch (const std::runtime_error& e)
        {
            d->exceptionThrown = true;
        }
    };

    {
        enki::TaskScheduler* sched = &ctx.threading.sched;
        LwTaskSet nrUpdate(threadData.data(), sched, ctx.atmos->Nspace,
                           chunkSize, update_handler);
        sched->AddTaskSetToPipe(&nrUpdate);
        sched->WaitforTaskSet(&nrUpdate);
    }

    bool throwNeeded = false;
    for (int t = 0; t < Nthreads; ++t)
        if (threadData[t].exceptionThrown)
            throwNeeded = true;

    if (throwNeeded)
        throw std::runtime_error("Singular Matrix");

}

void nr_post_update(Context& ctx, std::vector<Atom*>* atoms,
                    const std::vector<F64View3D>& dC,
                    F64View backgroundNe,
                    const NrTimeDependentData& timeDepData,
                    f64 crswVal,
                    ExtraParams params,
                    int spaceStart, int spaceEnd)
{
    return ctx.iterFns.nr_post_update(ctx, atoms, dC, backgroundNe, timeDepData,
                                      crswVal, params, spaceStart, spaceEnd);
}