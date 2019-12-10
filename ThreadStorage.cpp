#include "Lightweaver.hpp"
#include "Constants.hpp"
#include "CmoArray.hpp"
#include <vector>

struct TransitionStorage
{
    F64Arr Rij;
    F64Arr Rji;
};

struct TransitionStorageFactory
{
    Transition* trans;
    std::vector<Transition> tStorage;
    std::vector<TransitionStorage> arrayStorage;
    TransitionStorageFactory(Transition* t) : trans(t)
    {}

    Transition* copy_transition()
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
};

struct AtomStorage
{
    F64Arr3D Gamma;
    F64Arr eta;
    F64Arr2D gij;
    F64Arr2D wla;
    F64Arr2D V;
    F64Arr2D U;
    F64Arr2D chi;
};

struct AtomStorageFactory
{
    Atom* atom;
    bool lte;
    std::vector<Atom> aStorage;
    std::vector<TransitionStorageFactory> tStorage;
    std::vector<AtomStorage> arrayStorage;
    AtomStorageFactory(Atom* a) : atom(a), lte(false)
    {
        if (a->V)
            lte = true;

        for (auto t : atom->trans)
            tStorage.emplace_back(TransitionStorageFactory(t));
    }

    Atom* copy_atom()
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

        if (lte)
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
};