#include "Lightweaver.hpp"

void stat_eq(Atom* atomIn, int spaceStart, int spaceEnd)
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

void parallel_stat_eq(Context* ctx, int chunkSize)
{
    const int Natom = ctx->activeAtoms.size();
    // NOTE(cmo): Run single threaded if chunkSize is 0
    if (chunkSize <= 0 || ctx->atmos->Nspace <= chunkSize)
    {
        for (Atom* atom : ctx->activeAtoms)
        {
            stat_eq(atom);
        }
        return;
    }

    struct UpdateData
    {
        Atom* atom;
        bool exceptionThrown;
    };
    UpdateData* threadData = (UpdateData*)malloc(Natom * sizeof(UpdateData));
    sched_task* atomTasks = (sched_task*)malloc(Natom * sizeof(sched_task));
    for (int a = 0; a < Natom; ++a)
    {
        threadData[a].atom = ctx->activeAtoms[a];
        threadData[a].exceptionThrown = false;
    }


    auto stat_eq_handler = [](void* data, scheduler* s,
                              sched_task_partition p, sched_uint threadId)
    {
        UpdateData* d = (UpdateData*)data;
        try
        {
            stat_eq(d->atom, p.start, p.end);
        }
        catch (const std::runtime_error& e)
        {
            d->exceptionThrown = true;
        }
    };

    for (int a = 0; a < Natom; ++a)
        scheduler_add(&ctx->threading.sched, &atomTasks[a], stat_eq_handler,
                        (void*)(&threadData[a]), ctx->atmos->Nspace, chunkSize);
    for (int a = 0; a < Natom; ++a)
        scheduler_join(&ctx->threading.sched, &atomTasks[a]);

    free(atomTasks);

    bool throwNeeded = false;
    for (int a = 0; a < Natom; ++a)
        if (threadData[a].exceptionThrown)
            throwNeeded = true;

    free(threadData);

    if (throwNeeded)
        throw std::runtime_error("Singular Matrix");
}

void time_dependent_update(Atom* atomIn, F64View2D nOld, f64 dt,
                           int spaceStart, int spaceEnd)
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

void parallel_time_dep_update(Context* ctx, const std::vector<F64View2D>& oldPops,
                              f64 dt, int chunkSize)
{
    const int Natom = ctx->activeAtoms.size();
    // NOTE(cmo): Run single threaded if chunkSize is 0
    if (chunkSize <= 0 || ctx->atmos->Nspace <= chunkSize)
    {
        for (int a = 0; a < Natom; ++a)
        {
            time_dependent_update(ctx->activeAtoms[a], oldPops[a], dt);
        }
        return;
    }

    struct UpdateData
    {
        Atom* atom;
        const F64View2D* nOld;
        f64 dt;
        bool exceptionThrown;
    };

    UpdateData* threadData = (UpdateData*)malloc(Natom * sizeof(UpdateData));
    sched_task* atomTasks = (sched_task*)malloc(Natom * sizeof(sched_task));
    for (int a = 0; a < Natom; ++a)
    {
        threadData[a].atom = ctx->activeAtoms[a];
        threadData[a].nOld = &oldPops[a];
        threadData[a].dt = dt;
        threadData[a].exceptionThrown = false;
    }

    auto update_handler = [](void* data, scheduler* s,
                             sched_task_partition p, sched_uint threadId)
    {
        UpdateData* d = (UpdateData*)data;
        try
        {
            time_dependent_update(d->atom, *d->nOld, d->dt, p.start, p.end);
        }
        catch (const std::runtime_error& e)
        {
            d->exceptionThrown = true;
        }
    };

    for (int a = 0; a < Natom; ++a)
        scheduler_add(&ctx->threading.sched, &atomTasks[a], update_handler,
                        (void*)(&threadData[a]), ctx->atmos->Nspace, chunkSize);
    for (int a = 0; a < Natom; ++a)
        scheduler_join(&ctx->threading.sched, &atomTasks[a]);

    free(atomTasks);

    bool throwNeeded = false;
    for (int a = 0; a < Natom; ++a)
        if (threadData[a].exceptionThrown)
            throwNeeded = true;

    free(threadData);
    if (throwNeeded)
        throw std::runtime_error("Singular Matrix");
}