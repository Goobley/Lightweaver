#ifndef CMO_LIGHTWEAVER_HPP
#define CMO_LIGHTWEAVER_HPP

#include "CmoArray.hpp"
#include "Constants.hpp"
#include "Ng.hpp"
#include "Faddeeva.hh"
#include <complex>

typedef Jasnah::Array1NonOwn<bool> BoolView;
typedef Jasnah::Array1Own<i8> BoolArr; //  Avoid the dreaded vector<bool>
typedef Jasnah::Array1NonOwn<i32> I32View;
typedef Jasnah::Array1Own<i32> I32Arr;

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
    typedef Jasnah::Array4Own<std::vector<JInterpCoeffs>> JCoeffVec;
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

typedef std::complex<f64> (*WofZType)(std::complex<f64>);
void print_complex(std::complex<f64> cmp, WofZType wofz);

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
    F64Arr3D gII;
    Prd::RhoCoeffVec hPrdCoeffs;

    // F64Arr wlambda() const
    // {
    //     auto wla = F64Arr(wavelength.shape(0));
    //     int len = wavelength.shape(0);
    //     for (int i = 1; i < len-1; ++i)
    //     {
    //         wla(i) = 0.5 * (wavelength(i+1) - wavelength(i-1)) * dopplerWidth;
    //     }
    //     wla(0) = 0.5 * (wavelength(1) - wavelength(0)) * dopplerWidth;
    //     wla(len-1) = 0.5 * (wavelength(len-1) - wavelength(len-2)) * dopplerWidth;
    //     return wla;
    // }

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
};

struct Context
{
    Atmosphere* atmos;
    Spectrum* spect;
    std::vector<Atom*> activeAtoms;
    std::vector<Atom*> lteAtoms;
    Background* background;
};

f64 gamma_matrices_formal_sol(Context& ctx);
f64 formal_sol_full_stokes(Context& ctx);
f64 formal_sol(Context& ctx, I32View wavelengthIdxs, bool updateRates=false, bool updateJ=false);
f64 redistribute_prd_lines(Context& ctx, int maxIter, f64 tol);
void stat_eq(Atom* atom);
void planck_nu(long Nspace, double *T, double lambda, double *Bnu);
void piecewise_linear_1d(Atmosphere* atmos, int mu, bool toObs, f64 wav, 
                         F64View chi, F64View S, F64View I, F64View Psi);
void configure_hprd_coeffs(Context& ctx);

namespace EscapeProbability
{
void gamma_matrices_escape_prob(Atom* a, Background& background, 
                                const Atmosphere& atmos);
}

namespace LwInternal
{
    struct FormalData
    {
        Atmosphere* atmos;
        F64View chi;
        F64View S;
        F64View I;
        F64View Psi;
    };

    struct FormalDataStokes
    {
        Atmosphere* atmos;
        F64View2D chi;
        F64View2D S;
        F64View2D I;
        FormalData fdIntens;
    };

    struct IntensityCoreData
    {
        Atmosphere* atmos;
        Spectrum* spect;
        FormalData* fd;
        Background* background;
        std::vector<Atom*>* activeAtoms;
        std::vector<Atom*>* lteAtoms;
        F64Arr* JDag;
        F64View chiTot;
        F64View etaTot;
        F64View Uji;
        F64View Vij;
        F64View Vji;
        F64View I;
        F64View S;
        F64View Ieff;
        F64View PsiStar;
    };

    struct StokesCoreData
    {
        Atmosphere* atmos;
        Spectrum* spect;
        FormalDataStokes* fd;
        Background* background;
        std::vector<Atom*>* activeAtoms;
        std::vector<Atom*>* lteAtoms;
        F64Arr* JDag;
        F64View2D chiTot;
        F64View2D etaTot;
        F64View Uji;
        F64View Vij;
        F64View Vji;
        F64View2D I;
        F64View2D S;
    };

    inline void w2(f64 dtau, f64* w)
    {
        f64 expdt;

        if (dtau < 5.0E-4)
        {
            w[0] = dtau * (1.0 - 0.5 * dtau);
            w[1] = square(dtau) * (0.5 - dtau / 3.0);
        }
        else if (dtau > 50.0)
        {
            w[1] = w[0] = 1.0;
        }
        else
        {
            expdt = exp(-dtau);
            w[0] = 1.0 - expdt;
            w[1] = w[0] - dtau * expdt;
        }
    }

    enum FsMode : u32
    {
        UpdateJ = 1 << 0,
        UpdateRates = 1 << 1,
    };
    constexpr inline FsMode
    operator|(FsMode a, FsMode b)
    {
        return static_cast<FsMode>(static_cast<u32>(a) | static_cast<u32>(b));
    }

    void piecewise_bezier3_1d(FormalData* fd, int mu, bool toObs, f64 wav);
    void piecewise_stokes_bezier3_1d(FormalDataStokes* fd, int mu, bool toObs, f64 wav, bool polarisedFrequency);
    f64 intensity_core(IntensityCoreData& data, int la, FsMode mode);

}

#else
#endif