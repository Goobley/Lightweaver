#include "Constants.hpp"
#include "LwAtmosphere.hpp"
#include "LwInternal.hpp"
#include "Lightweaver.hpp"
#include "JasPP.hpp"
#include "Utils.hpp"
#include "LwFormalInterface.hpp"

#include <algorithm>
#include <limits>
#include <x86intrin.h>

using namespace LwInternal;

namespace CmoLinear4x
{
#define WIDTH 4

inline __m256d polynomial_13_lin(__m256d x, f64 c2, f64 c3, f64 c4,
                                 f64 c5, f64 c6, f64 c7, f64 c8,
                                 f64 c9, f64 c10, f64 c11, f64 c12, f64 c13)
{
    // c13*x^13 + ... + c2*x^2 + x + 0

    __m256d x2 = _mm256_mul_pd(x, x);
    __m256d x4 = _mm256_mul_pd(x2, x2);
    __m256d x8 = _mm256_mul_pd(x4, x4);

    // NOTE(cmo): Do all the inner powers first
    __m256d t3 = _mm256_fmadd_pd(_mm256_set1_pd(c3), x, _mm256_set1_pd(c2));
    __m256d t5 = _mm256_fmadd_pd(_mm256_set1_pd(c5), x, _mm256_set1_pd(c4));
    __m256d t7 = _mm256_fmadd_pd(_mm256_set1_pd(c7), x, _mm256_set1_pd(c6));
    __m256d t9 = _mm256_fmadd_pd(_mm256_set1_pd(c9), x, _mm256_set1_pd(c8));
    __m256d t11 = _mm256_fmadd_pd(_mm256_set1_pd(c11), x, _mm256_set1_pd(c10));
    __m256d t13 = _mm256_fmadd_pd(_mm256_set1_pd(c13), x, _mm256_set1_pd(c12));

    // NOTE(cmo): Next layer
    __m256d tt3 = _mm256_fmadd_pd(t3, x2, x);
    __m256d tt7 = _mm256_fmadd_pd(t7, x2, t5);
    __m256d tt11 = _mm256_fmadd_pd(t11, x2, t9);

    __m256d ttt7 = _mm256_fmadd_pd(tt7, x4, tt3);
    __m256d ttt13 = _mm256_fmadd_pd(t13, x4, tt11);

    return _mm256_fmadd_pd(ttt13, x8, ttt7);
}

inline __m256d polynomial_13_full(__m256d x, f64 c0, f64 c1, f64 c2,
                                  f64 c3, f64 c4, f64 c5, f64 c6,
                                  f64 c7,  f64 c8, f64 c9, f64 c10,
                                  f64 c11, f64 c12, f64 c13)
{
    // c13*x^13 + ... + c2*x^2 + c1*x + c0

    __m256d x2 = _mm256_mul_pd(x, x);
    __m256d x4 = _mm256_mul_pd(x2, x2);
    __m256d x8 = _mm256_mul_pd(x4, x4);

    // NOTE(cmo): Do all the inner powers first
    __m256d t1 = _mm256_fmadd_pd(_mm256_set1_pd(c1), x, _mm256_set1_pd(c0));
    __m256d t3 = _mm256_fmadd_pd(_mm256_set1_pd(c3), x, _mm256_set1_pd(c2));
    __m256d t5 = _mm256_fmadd_pd(_mm256_set1_pd(c5), x, _mm256_set1_pd(c4));
    __m256d t7 = _mm256_fmadd_pd(_mm256_set1_pd(c7), x, _mm256_set1_pd(c6));
    __m256d t9 = _mm256_fmadd_pd(_mm256_set1_pd(c9), x, _mm256_set1_pd(c8));
    __m256d t11 = _mm256_fmadd_pd(_mm256_set1_pd(c11), x, _mm256_set1_pd(c10));
    __m256d t13 = _mm256_fmadd_pd(_mm256_set1_pd(c13), x, _mm256_set1_pd(c12));

    // NOTE(cmo): Next layer
    __m256d tt3 = _mm256_fmadd_pd(t3, x2, t1);
    __m256d tt7 = _mm256_fmadd_pd(t7, x2, t5);
    __m256d tt11 = _mm256_fmadd_pd(t11, x2, t9);

    __m256d ttt7 = _mm256_fmadd_pd(tt7, x4, tt3);
    __m256d ttt13 = _mm256_fmadd_pd(t13, x4, tt11);

    return _mm256_fmadd_pd(ttt13, x8, ttt7);
}

inline __m256d polynomial_7_full(__m256d x, f64 c0, f64 c1, f64 c2,
                                 f64 c3, f64 c4, f64 c5, f64 c6, f64 c7)
{

    // c7*x^7 + ... + c2*x^2 + c1*x + c0

    __m256d x2 = _mm256_mul_pd(x, x);
    __m256d x4 = _mm256_mul_pd(x2, x2);

    // NOTE(cmo): Do all the inner powers first
    __m256d t1 = _mm256_fmadd_pd(_mm256_set1_pd(c1), x, _mm256_set1_pd(c0));
    __m256d t3 = _mm256_fmadd_pd(_mm256_set1_pd(c3), x, _mm256_set1_pd(c2));
    __m256d t5 = _mm256_fmadd_pd(_mm256_set1_pd(c5), x, _mm256_set1_pd(c4));
    __m256d t7 = _mm256_fmadd_pd(_mm256_set1_pd(c7), x, _mm256_set1_pd(c6));

    // NOTE(cmo): Next layer
    __m256d tt3 = _mm256_fmadd_pd(t3, x2, t1);
    __m256d tt7 = _mm256_fmadd_pd(t7, x2, t5);
    __m256d ttt7 = _mm256_fmadd_pd(tt7, x4, tt3);

    return ttt7;
}

inline __m256d pow2n (const __m256d n) {
    const __m256d pow2_52 = _mm256_set1_pd(4503599627370496.0);   // 2^52
    const __m256d bias = _mm256_set1_pd(1023.0);                  // bias in exponent
    __m256d a = _mm256_add_pd(n, _mm256_add_pd(bias, pow2_52));   // put n + bias in least significant bits
    __m256i b = _mm256_castpd_si256(a);  // bit-cast to integer
    __m256i c = _mm256_slli_epi64(b,52); // shift left 52 places to get value into exponent field
    __m256d d = _mm256_castsi256_pd(c);   // bit-cast back to double
    return d;
}
// NOTE(cmo): AVX impl of exp_pd, based on Agner Fog's vector class
// https://github.com/vectorclass/version2/blob/master/vectormath_exp.h
// The implementation here, based on a classic Taylor series, rather than a
// minimax function makes sense for our case, as we primarily value precision
// close to 0. i.e. we know the behaviour outwith this.
// Original under Apache v2 license.
inline __m256d exp_pd(__m256d xIn)
{
    constexpr f64 p2 = 1.0/2.0;
    constexpr f64 p3 = 1.0/6.0;
    constexpr f64 p4 = 1.0/24.0;
    constexpr f64 p5 = 1.0/120.0;
    constexpr f64 p6 = 1.0/720.0;
    constexpr f64 p7 = 1.0/5040.0;
    constexpr f64 p8 = 1.0/40320.0;
    constexpr f64 p9 = 1.0/362880.0;
    constexpr f64 p10 = 1.0/3628800.0;
    constexpr f64 p11 = 1.0/39916800.0;
    constexpr f64 p12 = 1.0/479001600.0;
    constexpr f64 p13 = 1.0/6227020800.0;

    constexpr f64 log2e = 1.44269504088896340736;

    constexpr f64 xMax = 708.39;
    constexpr f64 ln2dHi = 0.693145751953125;
    constexpr f64 ln2dLo = 1.42860682030941723212e-6;

    __m256d x = xIn;

    __m256d r = _mm256_round_pd(_mm256_mul_pd(xIn, _mm256_set1_pd(log2e)),
                                _MM_FROUND_TO_NEAREST_INT);
    // nmul_add(a, b, c) -> -(a * b) + c i.e. fnmadd
    // x = x0 - r * ln2Lo - r * ln2Hi
    x = _mm256_fnmadd_pd(r, _mm256_set1_pd(ln2dHi), x);
    x = _mm256_fnmadd_pd(r, _mm256_set1_pd(ln2dLo), x);

    __m256d z = polynomial_13_lin(x, p2, p3, p4, p5, p6, p7, p8,
                                  p9, p10, p11, p12, p13);
    __m256d n2 = pow2n(r);
    z = _mm256_mul_pd(_mm256_add_pd(z, _mm256_set1_pd(1.0)), n2);

    // TODO(cmo): Probably should have some of the nan/inf error handling code.
    return z;
}

struct alignas(32) w4x
{
    __m256d w0;
    __m256d w1;
};

inline w4x w01_expansion(__m256d xIn)
{
    constexpr f64 w0_0  = 0.0;
    constexpr f64 w0_1  = -1.0;
    constexpr f64 w0_2  = -1.0/2.0;
    constexpr f64 w0_3  = -1.0/6.0;
    constexpr f64 w0_4  = -1.0/24.0;
    constexpr f64 w0_5  = -1.0/120.0;
    constexpr f64 w0_6  = -1.0/720.0;
    constexpr f64 w0_7  = -1.0/5040.0;
    constexpr f64 w0_8  = -1.0/40320.0;
    constexpr f64 w0_9  = -1.0/362880.0;
    constexpr f64 w0_10 = -1.0/3628800.0;
    constexpr f64 w0_11 = -1.0/39916800.0;
    constexpr f64 w0_12 = -1.0/479001600.0;
    constexpr f64 w0_13 = -1.0/6227020800.0;

    constexpr f64 w1_0  = 0.0;
    constexpr f64 w1_1  = 0.0;
    constexpr f64 w1_2  = (2.0 - 1.0)/2.0;
    constexpr f64 w1_3  = (3.0 - 1.0)/6.0;
    constexpr f64 w1_4  = (4.0 - 1.0)/24.0;
    constexpr f64 w1_5  = (5.0 - 1.0)/120.0;
    constexpr f64 w1_6  = (6.0 - 1.0)/720.0;
    constexpr f64 w1_7  = (7.0 - 1.0)/5040.0;
    constexpr f64 w1_8  = (8.0 - 1.0)/40320.0;
    constexpr f64 w1_9  = (9.0 - 1.0)/362880.0;
    constexpr f64 w1_10 = (10.0 - 1.0)/3628800.0;
    constexpr f64 w1_11 = (11.0 - 1.0)/39916800.0;
    constexpr f64 w1_12 = (12.0 - 1.0)/479001600.0;
    constexpr f64 w1_13 = (13.0 - 1.0)/6227020800.0;


    constexpr f64 log2e = 1.44269504088896340736;

    constexpr f64 xMax = 708.39;
    constexpr f64 ln2dHi = 0.693145751953125;
    constexpr f64 ln2dLo = 1.42860682030941723212e-6;

    __m256d x = xIn;

    __m256d r = _mm256_round_pd(_mm256_mul_pd(xIn, _mm256_set1_pd(log2e)),
                                _MM_FROUND_TO_NEAREST_INT);
    // nmul_add(a, b, c) -> -(a * b) + c i.e. fnmadd
    // x = x0 - r * ln2Lo - r * ln2Hi
    x = _mm256_fnmadd_pd(r, _mm256_set1_pd(ln2dHi), x);
    x = _mm256_fnmadd_pd(r, _mm256_set1_pd(ln2dLo), x);

    // __m256d z0 = polynomial_13_full(x, w0_0, w0_1, w0_2, w0_3, w0_4, w0_5,
    //                                    w0_6, w0_7, w0_8, w0_9, w0_10, w0_11,
    //                                    w0_12, w0_13);
    __m256d z0 = polynomial_7_full(x, w0_0, w0_1, w0_2, w0_3, w0_4, w0_5,
                                      w0_6, w0_7);

    __m256d n2 = pow2n(r);

    const __m256d One = _mm256_set1_pd(1.0);
    w4x ws;
    // 1-e^x = (z0 * n2) - (n2 - 1)
    // This series is good enough for any x < 0 we care about
    ws.w0 = _mm256_fmsub_pd(z0, n2, _mm256_sub_pd(n2, One));
    // 1 - (1 - x) e^x
    // w0 + x e^x
    // w0 - x * (z0 - 1) * n2
    // fmnadd(mul(sub(z0, One), n2), x, w0)
    ws.w1 = _mm256_fnmadd_pd(_mm256_mul_pd(_mm256_sub_pd(z0, One), n2),
                             xIn, ws.w0);
    __m256d w1Lim = _mm256_set1_pd(-1.0e-3);
    __m256d w1Cond = _mm256_cmp_pd(xIn, w1Lim, _CMP_GT_OS);
    int mask = _mm256_movemask_pd(w1Cond);
    if (mask) // If any need expansion treatment
    {
        // This series is not good for larger x, hence the condition
        // __m256d z1 = polynomial_13_full(x, w1_0, w1_1, w1_2, w1_3, w1_4, w1_5,
        //                                 w1_6, w1_7, w1_8, w1_9, w1_10, w1_11,
        //                                 w1_12, w1_13);
        __m256d z1 = polynomial_7_full(x, w1_0, w1_1, w1_2, w1_3, w1_4, w1_5,
                                          w1_6, w1_7);
                            //  direct,  expansion,             if need expansion
        ws.w1 = _mm256_blendv_pd(ws.w1, _mm256_mul_pd(z1, n2), w1Cond);
    }

    // TODO(cmo): Probably should have some of the nan/inf error handling code.
    return ws;
}

w4x w2_4x(__m256d dtau)
{
    const __m256d bigTauLim = _mm256_set1_pd(50.0);
    const __m256d bigTauCond = _mm256_cmp_pd(dtau, bigTauLim, _CMP_GT_OS);
    const __m256d One = _mm256_set1_pd(1.0);
    const __m256d Zero = _mm256_setzero_pd();
    w4x result;
    result.w0 = One;
    result.w1 = One;
    // If they are not all over the limit
    if (_mm256_movemask_pd(bigTauCond) != 0xf)
    {
        __m256d  mdtau = _mm256_sub_pd(Zero, dtau);
        w4x smallerTerms = w01_expansion(mdtau);
                                     // small,        big,       if big
        result.w0 = _mm256_blendv_pd(smallerTerms.w0, result.w0, bigTauCond);
        result.w1 = _mm256_blendv_pd(smallerTerms.w1, result.w1, bigTauCond);
    }

    return result;
}

// NOTE(cmo): Abs impl based on https://stackoverflow.com/a/32422471/3847013
__m256d abs_mask()
{
    // NOTE(cmo): All 1s except sign bit
    __m256i m1 = _mm256_set1_epi64x(-1);
    return _mm256_castsi256_pd(_mm256_srli_epi64(m1, 1));
}

__m256d mm256_abs_pd(__m256d x)
{
    return _mm256_and_pd(abs_mask(), x);
}

void piecewise_linear_1d_4x_impl(FormalData* fd, f64 zmu, bool toObs,
                                 f64 Istart[WIDTH], int width)
{
    JasUnpack((*(fd->wideData)), chi, S, Psi, I);
    const auto& atmos = fd->atmos;
    const auto& height = atmos->height;
    const int Ndep = atmos->Nspace;
    bool computeOperator = bool(Psi);

    int dk = -1;
    int kStart = Ndep - 1;
    int kEnd = 0;
    if (!toObs)
    {
        dk = 1;
        kStart = 0;
        kEnd = Ndep - 1;
    }

    if (width == WIDTH)
    // if (false)
    {

        __m256d chik = _mm256_load_pd(&chi(kStart, 0));
        __m256d chikp = _mm256_load_pd(&chi(kStart + dk, 0));
        __m256d Sk = _mm256_load_pd(&S(kStart, 0));
        __m256d Skp = _mm256_load_pd(&S(kStart + dk, 0));
        __m256d heightk = _mm256_set1_pd(height(kStart));
        __m256d heightkp = _mm256_set1_pd(height(kStart + dk));
        __m256d zmuw = _mm256_set1_pd(zmu);

        __m256d dtauUw = _mm256_mul_pd(_mm256_mul_pd(zmuw, _mm256_add_pd(chik, chikp)),
                                       mm256_abs_pd(_mm256_sub_pd(heightk, heightkp)));
        __m256d dSUw = _mm256_div_pd(_mm256_sub_pd(Sk, Skp), dtauUw);
        __m256d Iupw = _mm256_load_pd(Istart);

        _mm256_store_pd(&I(kStart, 0), Iupw);
        if (computeOperator)
            _mm256_store_pd(&Psi(kStart, 0), _mm256_setzero_pd());


        __m256d One = _mm256_set1_pd(1.0);
        for (int k = kStart + dk; k != kEnd; k += dk)
        {
            w4x ws = w2_4x(dtauUw);
            __m256d chik = _mm256_load_pd(&chi(k, 0));
            __m256d chikp = _mm256_load_pd(&chi(k + dk, 0));
            __m256d Sk = _mm256_load_pd(&S(k, 0));
            __m256d Skp = _mm256_load_pd(&S(k + dk, 0));
            __m256d heightk = _mm256_set1_pd(height(k));
            __m256d heightkp = _mm256_set1_pd(height(k + dk));

            __m256d dtauDw = _mm256_mul_pd(_mm256_mul_pd(zmuw, _mm256_add_pd(chik, chikp)),
                                           mm256_abs_pd(_mm256_sub_pd(heightk, heightkp)));
            __m256d dSDw = _mm256_div_pd(_mm256_sub_pd(Sk, Skp), dtauDw);
            __m256d Ik = _mm256_add_pd(
                            _mm256_add_pd(
                                _mm256_mul_pd(_mm256_sub_pd(One, ws.w0), Iupw),
                                _mm256_mul_pd(ws.w0, Sk)),
                            _mm256_mul_pd(ws.w1, dSUw));
            _mm256_store_pd(&I(k, 0), Ik);

            if (computeOperator)
            {
                __m256d psik = _mm256_sub_pd(ws.w0, _mm256_div_pd(ws.w1, dtauUw));
                _mm256_store_pd(&Psi(k, 0), _mm256_div_pd(psik, chik));
            }

            Iupw = Ik;
            dSUw = dSDw;
            dtauUw = dtauDw;
        }

        w4x ws = w2_4x(dtauUw);
        __m256d Ik = _mm256_add_pd(
                        _mm256_add_pd(
                            _mm256_mul_pd(_mm256_sub_pd(One, ws.w0), Iupw),
                            _mm256_mul_pd(ws.w0, Sk)),
                        _mm256_mul_pd(ws.w1, dSUw));
        _mm256_store_pd(&I(kEnd, 0), Ik);
        if (computeOperator)
        {
            __m256d psik = _mm256_sub_pd(ws.w0, _mm256_div_pd(ws.w1, dtauUw));
            _mm256_store_pd(&Psi(kEnd, 0), _mm256_div_pd(psik, chikp));
        }

    }
    else
    {
        for (int laS = 0; laS < width; ++laS)
        {
            f64 dtau_uw = zmu * (chi(kStart, laS) + chi(kStart + dk, laS))
                            * abs(height(kStart) - height(kStart + dk));
            f64 dS_uw = (S(kStart, laS) - S(kStart + dk, laS)) / dtau_uw;

            f64 Iupw = Istart[laS];

            I(kStart, laS) = Iupw;
            if (computeOperator)
                Psi(kStart, laS) = 0.0;

            /* --- Solve transfer along ray --                   -------------- */

            f64 w[2];
            for (int k = kStart + dk; k != kEnd; k += dk)
            {
                w2(dtau_uw, w);

                /* --- Piecewise linear here --               -------------- */
                f64 dtau_dw = zmu * (chi(k, laS) + chi(k + dk, laS))
                                * abs(height(k) - height(k + dk));
                f64 dS_dw = (S(k, laS) - S(k + dk, laS)) / dtau_dw;

                I(k, laS) = (1.0 - w[0]) * Iupw + w[0] * S(k, laS) + w[1] * dS_uw;

                if (computeOperator)
                    Psi(k, laS) = w[0] - w[1] / dtau_uw;

                /* --- Re-use downwind quantities for next upwind position -- --- */
                Iupw = I(k, laS);
                dS_uw = dS_dw;
                dtau_uw = dtau_dw;
            }

            /* --- Piecewise linear integration at end of ray -- ---------- */
            w2(dtau_uw, w);
            I(kEnd, laS) = (1.0 - w[0]) * Iupw + w[0] * S(kEnd, laS) + w[1] * dS_uw;
            if (computeOperator)
            {
                Psi(kEnd, laS) = w[0] - w[1] / dtau_uw;
                for (int k = 0; k < Psi.shape(0); ++k)
                    Psi(k, laS) /= chi(k, laS);
            }
        }
    }
}

void piecewise_linear_1d_4x(FormalData* fd, int la, int mu,
                            bool toObs, const F64View1D& wave)
{
    auto& atmos = fd->atmos;
    auto& chi = fd->wideData->chi;
    f64 zmu = 0.5 / atmos->muz(mu);
    auto height = atmos->height;

    int dk = -1;
    int kStart = atmos->Nspace - 1;
    int kEnd = 0;
    if (!toObs)
    {
        dk = 1;
        kStart = 0;
        kEnd = atmos->Nspace - 1;
    }

    alignas(32) f64 Iupw[WIDTH] = {};
    const int fsEffWidth = min((i64)WIDTH, wave.shape(0) - la);
    for (int laS = 0; laS < fsEffWidth; ++laS)
    {
        f64 dtau_uw = zmu * (chi(kStart, laS) + chi(kStart + dk, laS))
                          * abs(height(kStart) - height(kStart + dk));
        if (toObs)
        {
            if (atmos->zLowerBc.type == THERMALISED)
            {
                f64 Bnu[2];
                int Nspace = atmos->Nspace;
                planck_nu(2, &atmos->temperature(Nspace - 2), wave(la+laS), Bnu);
                Iupw[laS] = Bnu[1] - (Bnu[0] - Bnu[1]) / dtau_uw;
            }
            else if (atmos->zLowerBc.type == CALLABLE)
            {
                int muIdx = atmos->zLowerBc.idxs(mu, int(toObs));
                if (muIdx == -1)
                {
                    // NOTE(cmo): This shouldn't be possible, so I won't try to
                    // recover.
                    printf("Error in boundary condition indexing\n");
                    assert(false);
                }
                Iupw[laS] = atmos->zLowerBc.bcData(la+laS, muIdx, 0);
            }
        }
        else
        {
            if (atmos->zUpperBc.type == THERMALISED)
            {
                f64 Bnu[2];
                planck_nu(2, &atmos->temperature(0), wave(la+laS), Bnu);
                Iupw[laS] = Bnu[0] - (Bnu[1] - Bnu[0]) / dtau_uw;
            }
            else if (atmos->zUpperBc.type == CALLABLE)
            {
                int muIdx = atmos->zUpperBc.idxs(mu, int(toObs));
                if (muIdx == -1)
                {
                    printf("Error in boundary condition indexing\n");
                    assert(false);
                    Iupw[laS] = 0.0;
                }
                else
                    Iupw[laS] = atmos->zUpperBc.bcData(la+laS, muIdx, 0);
            }
        }
    }

    piecewise_linear_1d_4x_impl(fd, zmu, toObs, Iupw, fsEffWidth);
}
}

extern "C"
{
FormalSolver fs_provider()
{
    return FormalSolver{1, 4, "piecewise_linear_1d_4x", CmoLinear4x::piecewise_linear_1d_4x};
}
}

#ifdef CMO_TESTING
int main()
{
    using namespace CmoLinear4x;
    f64 x = -20;
    f64 a0 = exp(x);
    f64 a1 = exp(x/2);
    f64 a2 = exp(x*2);
    f64 a3 = exp(x*x);
    __m256d vb = exp_pd(_mm256_set_pd(x*x, x*2, x/2, x));
    f64* b = (f64*)(&vb);
    printf("%e, %e, %e, %e\n", a0, a1, a2, a3);
    printf("%e, %e, %e, %e\n", b[0], b[1], b[2], b[3]);
}
#endif