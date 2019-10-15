#ifndef CMO_FORMAL_HPP
#define CMO_FORMAL_HPP

#include "Atmosphere.hpp"
#include "CmoArray.hpp"
#include "Constants.hpp"
#include "Ng.hpp"
#include <complex>

typedef double f64;
typedef Jasnah::Array1NonOwn<bool> BoolView;

struct Background
{
    F64View2D chi;
    F64View2D eta;
};

struct Spectrum
{
    F64View wavelength;
    F64View2D I;
    F64View2D J;
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
    BoolView active;

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
                Uji(k) = Aji / Bji * Vji(k);
            }
        }
        else
        {
            constexpr f64 hc = 2 * C::HC / cube(C::NM_TO_M);
            const f64 hcl = hc / cube(wavelength(lt));
            const f64 a = alpha(lt);
            for (int k = 0; k < Vij.shape(0); ++k)
            {
                Vij(k) = a;
                Vji(k) = gij(k) * Vij(k);
                Uji(k) = hcl * Vji(k);
            }
        }
    }

    void compute_phi(const Atmosphere& atmos, F64View aDamp, F64View vBroad);
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
            auto wlambda = t.wlambda(t.lt_idx(laIdx));
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
};

struct Context
{
    Atmosphere* atmos;
    Spectrum* spect;
    std::vector<Atom*> activeAtoms;
    Background* background;
};

typedef void (*DgesvType)(int* n, int* nrhs, f64* a, int* lda, int* ipiv, f64* b, int* ldb, int* info);
f64 gamma_matrices_formal_sol(Context ctx);
void stat_eq(Atom* atom, DgesvType f);
void planck_nu(long Nspace, double *T, double lambda, double *Bnu);
void piecewise_linear_1d(Atmosphere* atmos, int mu, bool toObs, f64 wav, 
                         F64View chi, F64View S, F64View I, F64View Psi);

#else
#endif