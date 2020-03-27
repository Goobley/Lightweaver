#include "ThreadStorage.hpp"
#include "Lightweaver.hpp"

namespace LwInternal
{

TransitionStorageFactory::TransitionStorageFactory(Transition* t) : trans(t)
{}

void TransitionStorageFactory::reserve(int Nthreads)
{
    tStorage.reserve(Nthreads);
    arrayStorage.reserve(Nthreads);
}

Transition* TransitionStorageFactory::copy_transition()
{
    if (!trans)
        return nullptr;

    tStorage.emplace_back(Transition());
    Transition* t = &tStorage.back();
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

    arrayStorage.emplace_back(TransitionStorage());
    auto& ts = arrayStorage.back();
    ts.Rij = F64Arr(0.0, trans->Rij.shape(0));
    t->Rij = ts.Rij;
    ts.Rji = F64Arr(0.0, trans->Rji.shape(0));
    t->Rji = ts.Rji;

    return t;
}
void TransitionStorageFactory::accumulate_rates()
{
    const int Nspace = trans->Rij.shape(0);
    for (const auto& t : tStorage)
    {
        for (int k = 0; k < Nspace; ++k)
        {
            trans->Rij(k) += t.Rij(k);
            trans->Rji(k) += t.Rji(k);
        }
    }
}

AtomStorageFactory::AtomStorageFactory(Atom* a, bool detail) 
    : atom(a),         
      detailedStatic(detail)
{
    tStorage.reserve(atom->trans.size());
    for (auto t : atom->trans)
        tStorage.emplace_back(TransitionStorageFactory(t));
}

void AtomStorageFactory::reserve(int Nthreads)
{
    aStorage.reserve(Nthreads);
    tStorage.reserve(Nthreads);
    arrayStorage.reserve(Nthreads);
    for (auto& t : tStorage)
        t.reserve(Nthreads);
}

Atom* AtomStorageFactory::copy_atom()
{
    if (!atom)
        return nullptr;

    aStorage.emplace_back(Atom());
    Atom* a = &aStorage.back();
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

    arrayStorage.emplace_back(AtomStorage());
    auto& as = arrayStorage.back();
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
                    atom->Gamma(i, j, k) += a.Gamma(i, j, k);

}

void IntensityCoreFactory::initialise(Context* context)
{
    ctx = context;
    atmos = context->atmos;
    spect = context->spect;
    background = context->background;
    if (ctx->Nthreads <= 1)
        return;

    bool detailedStatic = false;
    activeAtoms.reserve(ctx->activeAtoms.size());
    for (auto a : ctx->activeAtoms)
    {
        printf("Adding atom\n");
        activeAtoms.emplace_back(AtomStorageFactory(a, detailedStatic=false));
        activeAtoms.back().reserve(ctx->Nthreads);
    }

    detailedAtoms.reserve(ctx->detailedAtoms.size());
    for (auto a : ctx->detailedAtoms)
    {
        detailedAtoms.emplace_back(AtomStorageFactory(a, detailedStatic=true));
        detailedAtoms.back().reserve(ctx->Nthreads);
    }

    arrayStorage.reserve(ctx->Nthreads);
    fdStorage.reserve(ctx->Nthreads);
}

IntensityCoreData IntensityCoreFactory::new_intensity_core(bool psiOperator)
{
    arrayStorage.emplace_back(IntensityCoreStorage());
    auto& as = arrayStorage.back();
    const int Nspace = atmos->Nspace;
    as.set_Nspace(Nspace);
    fdStorage.emplace_back(FormalData());
    auto& fd = fdStorage.back();

    IntensityCoreData iCore;
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
        printf("Adding atom inner\n");
    }
    iCore.activeAtoms = &as.activeAtoms;
    
    as.detailedAtoms.reserve(detailedAtoms.size());
    for (auto& atom : detailedAtoms)
        as.detailedAtoms.emplace_back(atom.copy_atom());
    iCore.detailedAtoms = &as.detailedAtoms;

    return iCore;
}

void IntensityCoreFactory::accumulate_Gamma_rates()
{
    for (auto& a : activeAtoms)
        a.accumulate_Gamma_rates();
    for (auto& a : detailedAtoms)
        a.accumulate_Gamma_rates();
}

}