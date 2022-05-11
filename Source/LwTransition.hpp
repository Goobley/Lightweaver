#ifndef CMO_LW_TRANSITION_HPP
#define CMO_LW_TRANSITION_HPP

#include "CmoArray.hpp"
#include "LwAtmosphere.hpp"
#include "LwMisc.hpp"

#include <algorithm>
#include <functional>

enum TransitionType
{
    LINE,
    CONTINUUM
};

namespace LwInternal
{
    struct ThreadData;
}

struct Transition
{
    // NOTE(cmo): Transitions are often queried only for whether they are active, so
    // bring Nblue and Nred in on the first cache line
    int Nblue;
    int Nred;

    enum TransitionType type;
    int i;
    int j;

    f64 Aji;
    f64 Bji;
    f64 Bij;
    f64 lambda0;
    f64 dopplerWidth;

    F64View wavelength;
    F64View gij;
    F64View alpha;
    F64View4D phi;
    F64View wphi;
    bool polarised;
    F64View4D phiQ;
    F64View4D phiU;
    F64View4D phiV;
    F64View4D psiQ;
    F64View4D psiU;
    F64View4D psiV;
    F64View Qelast;
    F64View aDamp;
    // NOTE(cmo): This was probably a bad idea for cache locality, and I can't
    // remember why I did it.
    BoolView active;

    F64View Rij;
    F64View Rji;
    F64View2D rhoPrd;

    void* methodScratch;

    Prd::PrdStorage* prdData;

    Prd::RhoCoeffView hPrdCoeffs;
    Prd::PrdStorage prdStorage;

    Transition() : prdData(&prdStorage)
    {}

    inline f64 wlambda(int la) const
    {
        if (la == 0)
            return 0.5 * (wavelength(1) - wavelength(0)) * dopplerWidth;

        int len = wavelength.shape(0);
        if (la == len-1)
            return 0.5 * (wavelength(len-1) - wavelength(len-2)) * dopplerWidth;

        return 0.5 * (wavelength(la+1) - wavelength(la-1)) * dopplerWidth;
    }

    inline int lt_idx(int la) const
    {
        return la - Nblue;
    }

    inline bool is_active(int la) const
    {
        return (la >= Nblue) && (la < Nred);
    }

    inline void uv(int la, int mu, bool toObs, F64View Uji, F64View Vij, F64View Vji) const
    {
        namespace C = Constants;
        int lt = lt_idx(la);

        if (type == TransitionType::LINE)
        {
            constexpr f64 hc_4pi = 0.25 * C::HC / C::Pi;
            // NOTE(cmo): This term (hnu_4pi) added to improve treatment of emissivities
            // opacities if they are calculated across very wide wavelength
            // ranges, rather than just the standard range used to integrate the
            // line rates. This happens, for example in ctx.compute_rays.
            // The result of this is the complete hnu/4pi * phi expression is
            // used, rather than the hc/4pi*phi_num expression from the paper.
            const f64 hnu_4pi = hc_4pi * (lambda0 / wavelength(lt));
            auto p = phi(lt, mu, (int)toObs);
            for (int k = 0; k < Vij.shape(0); ++k)
            {
                // Vij(k) = hc_4pi * Bij * p(k);
                // NOTE(cmo): Updated term
                Vij(k) = hnu_4pi * Bij * p(k);
                Vji(k) = gij(k) * Vij(k);
            }
            // NOTE(cmo): Do the HPRD linear interpolation on rho here
            // As we make Uji, Vij, and Vji explicit, there shouldn't be any need for direct access to gij
            if (hPrdCoeffs)
            {
                for (int k = 0; k < Vij.shape(0); ++k)
                {
                    const auto& coeffs = hPrdCoeffs(lt, mu, toObs, k);
                    f64 rho = (1.0 - coeffs.frac) * rhoPrd(coeffs.i0, k) + coeffs.frac * rhoPrd(coeffs.i1, k);
                    Vji(k) *= rho;
                }
            }
            for (int k = 0; k < Vij.shape(0); ++k)
            {
                Uji(k) = Aji / Bji * Vji(k);
            }
        }
        else
        {
            constexpr f64 twoHc = 2.0 * C::HC / cube(C::NM_TO_M);
            const f64 hcl = twoHc / cube(wavelength(lt));
            const f64 a = alpha(lt);
            for (int k = 0; k < Vij.shape(0); ++k)
            {
                Vij(k) = a;
                Vji(k) = gij(k) * Vij(k);
                Uji(k) = hcl * Vji(k);
            }
        }
    }

    inline void zero_rates()
    {
        Rij.fill(0.0);
        Rji.fill(0.0);
    }

    inline void recompute_gII()
    {
        // NOTE(cmo): This is just a flag to recompute it in a deferred manner
        // next time the prd_scatter function is run.
        prdStorage.upToDate = false;
    }

    void compute_phi(const Atmosphere& atmos, F64View aDamp, F64View vBroad);
    void compute_phi_la(const Atmosphere& atmos, const F64View& aDamp, const F64View& vBroad, int lt);
    void compute_phi_parallel(LwInternal::ThreadData* threading, const Atmosphere& atmos, F64View aDamp, F64View vBroad);
    std::function<void(const Atmosphere&, F64View, F64View)> bound_parallel_compute_phi;
    void compute_wphi(const Atmosphere& atmos);
    void compute_polarised_profiles(const Atmosphere& atmos, F64View aDamp, F64View vBroad, const ZeemanComponents& z);
};

#else
#endif