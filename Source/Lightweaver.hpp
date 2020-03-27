#ifndef CMO_LIGHTWEAVER_HPP
#define CMO_LIGHTWEAVER_HPP

#include "CmoArray.hpp"
#include "Constants.hpp"
#include "Ng.hpp"
#include "Faddeeva.hh"
#include "ThreadStorage.hpp"
#include "LwInternal.hpp"
#include <complex>
#include <vector>

typedef View<bool> BoolView;
typedef Arr<i8> BoolArr; //  Avoid the dreaded vector<bool>
typedef View<i32> I32View;
typedef Arr<i32> I32Arr;

inline f64 voigt_H(f64 a, f64 v)
{
    using Faddeeva::w;
    using namespace std::complex_literals;
    auto z = (v + a * 1i);
    return w(z).real();
}

inline std::complex<f64> voigt_HF(f64 a, f64 v)
{
    using Faddeeva::w;
    using namespace std::complex_literals;
    auto z = (v + a * 1i);
    return w(z);
}

namespace Prd
{
    struct RhoInterpCoeffs
    {
        int i0;
        int i1;
        f64 frac;
        RhoInterpCoeffs() : i0(0), i1(0), frac(0.0) {}
        RhoInterpCoeffs(int idx0, int idx1, f64 f) : i0(idx0), i1(idx1), frac(f) {}
    };
    struct JInterpCoeffs
    {
        f64 frac;
        int idx;
        JInterpCoeffs() : frac(0.0), idx(0) {}
        JInterpCoeffs(int i, f64 f) : frac(f), idx(i) {}
    };
    typedef Jasnah::Array4Own<RhoInterpCoeffs> RhoCoeffVec;
    typedef Jasnah::Array4NonOwn<RhoInterpCoeffs> RhoCoeffView;
    typedef Jasnah::Array4Own<std::vector<JInterpCoeffs>> JCoeffVec;

    struct PrdStorage
    {
        F64Arr3D gII;
        Prd::RhoCoeffVec hPrdCoeffs;
    };
}

namespace PrdCores
{
    constexpr int max_fine_grid_size();
}

enum RadiationBC
{
    ZERO,
    THERMALISED
};

struct Atmosphere
{
    F64View cmass;
    F64View height;
    F64View tau_ref;
    F64View temperature;
    F64View ne;
    F64View vlos;
    F64View2D vlosMu;
    F64View B;
    F64View gammaB;
    F64View chiB;
    F64View2D cosGamma;
    F64View2D cos2chi;
    F64View2D sin2chi;
    F64View vturb;
    F64View nHtot;
    F64View muz;
    F64View muy;
    F64View mux;
    F64View wmu;
    int Nspace;
    int Nrays;

    enum RadiationBC lowerBc;
    enum RadiationBC upperBc;

    void update_projections();
};


struct Background
{
    F64View2D chi;
    F64View2D eta;
    F64View2D sca;
};

struct Spectrum
{
    F64View wavelength;
    F64View2D I;
    F64View3D Quv;
    F64View2D J;
    BoolArr prdActive;
    std::vector<int> prdIdxs;
    BoolArr hPrdActive;
    std::vector<int> hPrdIdxs;
    I32Arr la_to_prdLa;
    I32Arr la_to_hPrdLa;
    Prd::JCoeffVec JCoeffs;
    F64Arr2D JRest;
};

struct ZeemanComponents
{
    I32View alpha;
    F64View shift;
    F64View strength;
};

enum TransitionType
{
    LINE,
    CONTINUUM
};

struct Transition
{
    enum TransitionType type;
    f64 Aji;
    f64 Bji;
    f64 Bij;
    f64 lambda0;
    f64 dopplerWidth;
    int Nblue;
    int i;
    int j;
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
    BoolView active;

    F64View Rij;
    F64View Rji;
    F64View2D rhoPrd;

    F64View3D gII;
    Prd::RhoCoeffView hPrdCoeffs;
    Prd::PrdStorage prdStorage;

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

    inline void uv(int la, int mu, bool toObs, F64View Uji, F64View Vij, F64View Vji) const
    {
        namespace C = Constants;
        int lt = lt_idx(la);

        if (type == TransitionType::LINE)
        {
            constexpr f64 hc_4pi = 0.25 * C::HC / C::Pi;
            auto p = phi(lt, mu, (int)toObs);
            for (int k = 0; k < Vij.shape(0); ++k)
            {
                Vij(k) = hc_4pi * Bij * p(k);
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

    void compute_phi(const Atmosphere& atmos, F64View aDamp, F64View vBroad);
    void compute_polarised_profiles(const Atmosphere& atmos, F64View aDamp, F64View vBroad, const ZeemanComponents& z);
};

struct Atom
{
    Atmosphere* atmos;
    F64View2D n;
    F64View2D nStar;
    F64View vBroad;
    F64View nTotal;

    F64View3D Gamma;
    F64View3D C;

    F64View eta;
    F64View2D gij;
    F64View2D wla;
    F64View2D V;
    F64View2D U;
    F64View2D chi;

    std::vector<Transition*> trans;

    Ng ng;

    int Nlevel;
    int Ntrans;

    inline void setup_wavelength(int laIdx)
    {
        namespace C = Constants;
        constexpr f64 pi4_h = 4.0 * C::Pi / C::HPlanck;
        constexpr f64 hc_4pi = 0.25 * C::HC / C::Pi;
        constexpr f64 pi4_hc = 1.0 / hc_4pi; 
        constexpr f64 hc_k = C::HC / (C::KBoltzmann * C::NM_TO_M);
        gij.fill(0.0);
        wla.fill(0.0);
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
        V.fill(0.0);
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

struct Context
{
    Atmosphere* atmos;
    Spectrum* spect;
    std::vector<Atom*> activeAtoms;
    std::vector<Atom*> detailedAtoms;
    Background* background;
    int Nthreads;
    LwInternal::ThreadData threading;

    void initialise_threads()
    {
        threading.threadDataFactory.initialise(this);
        if (Nthreads <= 1)
            return;

        if (threading.schedMemory)
            assert(false && "Tried to re initialise_threads for a Context");

        sched_size memNeeded;
        scheduler_init(&threading.sched, &memNeeded, Nthreads, nullptr);
        threading.schedMemory = calloc(memNeeded, 1);
        scheduler_start(&threading.sched, threading.schedMemory);

        threading.intensityCores.reserve(Nthreads);
        for (int t = 0; t < Nthreads; ++t)
        {
            threading.intensityCores.emplace_back(threading.threadDataFactory.new_intensity_core(true));
        }

    }

    void update_threads()
    {
        assert(false && "do me");
        // NOTE(cmo): Can we use references on the scalars to get transparent update on the Atom "copies" per thread?
        // It would end up being a derived type (extra pointers) -- unless we changed it everywhere and held the originals in cython...

    }
};

struct PrdIterData
{
    int iter;
    f64 dRho;
};

f64 formal_sol_gamma_matrices(Context& ctx);
f64 formal_sol_update_rates(Context& ctx);
f64 formal_sol_update_rates_fixed_J(Context& ctx);
f64 formal_sol(Context& ctx);
f64 formal_sol_full_stokes(Context& ctx, bool updateJ=true);
PrdIterData redistribute_prd_lines(Context& ctx, int maxIter, f64 tol);
void stat_eq(Atom* atom);
void time_dependent_update(Atom* atomIn, F64View2D nOld, f64 dt);
void planck_nu(long Nspace, double *T, double lambda, double *Bnu);
void configure_hprd_coeffs(Context& ctx);

namespace EscapeProbability
{
void gamma_matrices_escape_prob(Atom* a, Background& background, 
                                const Atmosphere& atmos);
}


#else
#endif