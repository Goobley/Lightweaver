#ifndef CMO_LW_ATOM_HPP
#define CMO_LW_ATOM_HPP

#include "CmoArray.hpp"
#include "Ng.hpp"
#include "LwAtmosphere.hpp"
#include "LwMisc.hpp"
#include "LwTransition.hpp"

#include <algorithm>

namespace LwInternal
{
struct AtomScratchStore
{
    F64Arr2D wla;
    F64Arr2D gij;
    F64Arr1D eta;
    F64Arr2D U;
    F64Arr2D chi;

    AtomScratchStore() = default;
    AtomScratchStore(int Nlevel, int Ntrans, i64 Nspace, bool detailed,
                     bool wlaGijStorage, bool defaultPerAtomStorage)
    {
        if (Ntrans > 0 && wlaGijStorage)
        {
            wla = F64Arr2D(Ntrans, Nspace);
            gij = F64Arr2D(Ntrans, Nspace);
        }

        if ((!detailed) && (defaultPerAtomStorage))
        {
            eta = F64Arr1D(Nspace);
            U = F64Arr2D(Nlevel, Nspace);
            chi = F64Arr2D(Nlevel, Nspace);
        }
    }
};
}

struct Atom
{
    int Nlevel;
    int Ntrans;

    Atmosphere* atmos;
    F64View2D n;
    F64View2D nStar;
    F64View1D nTotal;
    F64View1D vBroad;
    F64View1D stages;

    F64View3D Gamma;
    F64View3D C;

    F64View2D wla;
    F64View2D gij;
    F64View1D eta;
    F64View2D U;
    F64View2D chi;

    std::vector<Transition*> trans;

    Ng ng;
    LwInternal::AtomScratchStore scratch;
    void* methodScratch;

    void init_scratch(i64 Nspace, bool detailed,
                      bool wlaGijStorage, bool defaultPerAtomStorage)
    {
        scratch = LwInternal::AtomScratchStore(Nlevel, Ntrans, Nspace,
                                               detailed, wlaGijStorage,
                                               defaultPerAtomStorage);
        wla = scratch.wla;
        gij = scratch.gij;
        eta = scratch.eta;
        U = scratch.U;
        chi = scratch.chi;
    }

    inline void setup_wavelength(int laIdx, int fsWidth=1)
    {
        namespace C = Constants;
        constexpr f64 pi4_h = 4.0 * C::Pi / C::HPlanck;
        constexpr f64 hc_4pi = 0.25 * C::HC / C::Pi;
        constexpr f64 pi4_hc = 1.0 / hc_4pi;
        constexpr f64 hc_k = C::HC / (C::KBoltzmann * C::NM_TO_M);
        for (int kr = 0; kr < Ntrans; ++kr)
        {
            auto& t = *trans[kr];
            if (!t.active(laIdx))
                continue;

            auto g = gij(kr);
            auto w = wla(kr);
            const int lt = t.lt_idx(laIdx);
            auto wlambda = t.wlambda(lt);
            if (t.type == TransitionType::LINE)
            {
                for (int k = 0; k < g.shape(0); ++k)
                {
                    g(k) = t.Bji / t.Bij;
                    w(k) = wlambda * t.wphi(k) * pi4_hc;
                }
            }
            else
            {
                const f64 hc_kl = hc_k / t.wavelength(t.lt_idx(laIdx));
                const f64 wlambda_lambda = wlambda / t.wavelength(t.lt_idx(laIdx));
                for (int k = 0; k < g.shape(0); ++k)
                {
                    g(k) = nStar(t.i, k) / nStar(t.j, k) * exp(-hc_kl / atmos->temperature(k));
                    w(k) = wlambda_lambda * pi4_h;
                }
            }

            // NOTE(cmo): We have to do a linear interpolation on rhoPrd in the
            // case of hybrid PRD, so we can't pre-multiply here in that
            // instance.
            if (t.rhoPrd && !t.hPrdCoeffs)
                for (int k = 0; k < g.shape(0); ++k)
                    g(k) *= t.rhoPrd(lt, k);

            if (!t.gij)
                t.gij = g;
        }
    }

    inline void zero_angle_dependent_vars()
    {
        eta.fill(0.0);
        U.fill(0.0);
        chi.fill(0.0);
    }

    inline void zero_rates()
    {
        for (auto& t : trans)
            t->zero_rates();
    }

    inline void zero_Gamma()
    {
        if (!Gamma)
            return;

        Gamma.fill(0.0);
    }
};

#else
#endif