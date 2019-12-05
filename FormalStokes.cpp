#include "Lightweaver.hpp"
#include "Bezier.hpp"
#include "JasPP.hpp"
#include <x86intrin.h>
#include <string.h>

using namespace LwInternal;

void Transition::compute_polarised_profiles(
    const Atmosphere& atmos, F64View aDamp, F64View vBroad, const ZeemanComponents& z)
{
    namespace C = Constants;
    if (type == TransitionType::CONTINUUM)
        return;

    if (!polarised)
        return;

    constexpr f64 sign[] = { -1.0, 1.0 };

    const f64 larmor = C::QElectron / (4.0 * C::Pi * C::MElectron) * (lambda0 * C::NM_TO_M);
    const f64 sqrtPi = sqrt(C::Pi);

    assert((bool)atmos.B && "Must provide magnetic field when computing polarised profiles");
    assert((bool)atmos.cosGamma && "Must call Atmosphere::update_projections before computing polarised profiles");
    F64Arr vB(atmos.Nspace);
    F64Arr sv(atmos.Nspace);
    for (int k = 0; k < atmos.Nspace; ++k)
    {
        vB(k) = larmor * atmos.B(k) / vBroad(k);
        sv(k) = 1.0 / (sqrt(C::Pi) * vBroad(k));
    }
    phi.fill(0.0);
    wphi.fill(0.0);
    phiQ.fill(0.0);
    phiU.fill(0.0);
    phiV.fill(0.0);
    psiQ.fill(0.0);
    psiU.fill(0.0);
    psiV.fill(0.0);

    for (int la = 0; la < wavelength.shape(0); ++la)
    {
        const f64 vBase = (wavelength(la) - lambda0) * C::CLight / lambda0;
        const f64 wla = wlambda(la);
        for (int mu = 0; mu < phi.shape(1); ++mu)
        {
            const f64 wlamu = wla * 0.5 * atmos.wmu(mu);
            for (int toObs = 0; toObs < 2; ++toObs)
            {
                const f64 s = sign[toObs];
                for (int k = 0; k < atmos.Nspace; ++k)
                {
                    const f64 vk = (vBase + s * atmos.vlosMu(mu, k)) / vBroad(k);
                    f64 phi_sb = 0.0, phi_pi = 0.0, phi_sr = 0.0;
                    f64 psi_sb = 0.0, psi_pi = 0.0, psi_sr = 0.0;

                    for (int nz = 0; nz < z.alpha.shape(0); ++nz)
                    {
                        auto HF = voigt_HF(aDamp(k), vk - z.shift(nz) * vB(k));
                        auto H = HF.real();
                        auto F = HF.imag(); // NOTE(cmo): Not sure if the 0.5 should be here -- don't think so. Think
                            // it's accounted for later.

                        switch (z.alpha(nz))
                        {
                        case -1:
                        {
                            phi_sb += z.strength(nz) * H;
                            psi_sb += z.strength(nz) * F;
                        }
                        break;
                        case 0:
                        {
                            phi_pi += z.strength(nz) * H;
                            psi_pi += z.strength(nz) * F;
                        }
                        break;
                        case 1:
                        {
                            phi_sr += z.strength(nz) * H;
                            psi_sr += z.strength(nz) * F;
                        }
                        break;
                        }
                    }
                    f64 sin2_gamma = 1.0 - square(atmos.cosGamma(mu, k));
                    f64 cos_2chi = atmos.cos2chi(mu, k);
                    f64 sin_2chi = atmos.sin2chi(mu, k);
                    f64 cos_gamma = atmos.cosGamma(mu, k);

                    f64 phi_sigma = phi_sr + phi_sb;
                    f64 phi_delta = 0.5 * phi_pi - 0.25 * phi_sigma;
                    phi(la, mu, toObs, k) += (phi_delta * sin2_gamma + 0.5 * phi_sigma) * sv(k);

                    phiQ(la, mu, toObs, k) += s * phi_delta * sin2_gamma * cos_2chi * sv(k);
                    phiU(la, mu, toObs, k) += phi_delta * sin2_gamma * sin_2chi * sv(k);
                    phiV(la, mu, toObs, k) += s * 0.5 * (phi_sr - phi_sb) * cos_gamma * sv(k);

                    f64 psi_sigma = psi_sr + psi_sb;
                    f64 psi_delta = 0.5 * psi_pi - 0.25 * psi_sigma;

                    psiQ(la, mu, toObs, k) += s * psi_delta * sin2_gamma * cos_2chi * sv(k);
                    psiU(la, mu, toObs, k) += psi_delta * sin2_gamma * sin_2chi * sv(k);
                    psiV(la, mu, toObs, k) += s * 0.5 * (psi_sr - psi_sb) * cos_gamma * sv(k);

                    wphi(k) += wlamu * phi(la, mu, toObs, k);
                }
            }
        }
    }
    for (int k = 0; k < wphi.shape(0); ++k)
    {
        wphi(k) = 1.0 / wphi(k);
    }
}

void SIMD_MatInv(float* src)
{
    /* ---

        Very fast in-place 4x4 Matrix inversion using SIMD instrutions
        Only works with 32-bits floats. It uses Cramer's rule.

        Provided by Intel

        Requires SSE instructions but all x86 machines since
        Pentium III have them.

        --                                            ------------------ */
    // NOTE(cmo): This can also be done equivalently for f64 with avx/avx2 on newer cpus

    __m128 minor0, minor1, minor2, minor3;
    __m128 row0, row1, row2, row3;
    __m128 det, tmp1;

    // -----------------------------------------------
    tmp1 = _mm_loadh_pi(_mm_loadl_pi(tmp1, (__m64*)(src)), (__m64*)(src + 4));
    row1 = _mm_loadh_pi(_mm_loadl_pi(row1, (__m64*)(src + 8)), (__m64*)(src + 12));
    row0 = _mm_shuffle_ps(tmp1, row1, 0x88);
    row1 = _mm_shuffle_ps(row1, tmp1, 0xDD);
    tmp1 = _mm_loadh_pi(_mm_loadl_pi(tmp1, (__m64*)(src + 2)), (__m64*)(src + 6));
    row3 = _mm_loadh_pi(_mm_loadl_pi(row3, (__m64*)(src + 10)), (__m64*)(src + 14));
    row2 = _mm_shuffle_ps(tmp1, row3, 0x88);
    row3 = _mm_shuffle_ps(row3, tmp1, 0xDD);
    // -----------------------------------------------
    tmp1 = _mm_mul_ps(row2, row3);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
    minor0 = _mm_mul_ps(row1, tmp1);
    minor1 = _mm_mul_ps(row0, tmp1);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
    minor0 = _mm_sub_ps(_mm_mul_ps(row1, tmp1), minor0);
    minor1 = _mm_sub_ps(_mm_mul_ps(row0, tmp1), minor1);
    minor1 = _mm_shuffle_ps(minor1, minor1, 0x4E);
    // -----------------------------------------------
    tmp1 = _mm_mul_ps(row1, row2);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
    minor0 = _mm_add_ps(_mm_mul_ps(row3, tmp1), minor0);
    minor3 = _mm_mul_ps(row0, tmp1);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
    minor0 = _mm_sub_ps(minor0, _mm_mul_ps(row3, tmp1));
    minor3 = _mm_sub_ps(_mm_mul_ps(row0, tmp1), minor3);
    minor3 = _mm_shuffle_ps(minor3, minor3, 0x4E);
    // -----------------------------------------------
    tmp1 = _mm_mul_ps(_mm_shuffle_ps(row1, row1, 0x4E), row3);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
    row2 = _mm_shuffle_ps(row2, row2, 0x4E);
    minor0 = _mm_add_ps(_mm_mul_ps(row2, tmp1), minor0);
    minor2 = _mm_mul_ps(row0, tmp1);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
    minor0 = _mm_sub_ps(minor0, _mm_mul_ps(row2, tmp1));
    minor2 = _mm_sub_ps(_mm_mul_ps(row0, tmp1), minor2);
    minor2 = _mm_shuffle_ps(minor2, minor2, 0x4E);
    // -----------------------------------------------
    tmp1 = _mm_mul_ps(row0, row1);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
    minor2 = _mm_add_ps(_mm_mul_ps(row3, tmp1), minor2);
    minor3 = _mm_sub_ps(_mm_mul_ps(row2, tmp1), minor3);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
    minor2 = _mm_sub_ps(_mm_mul_ps(row3, tmp1), minor2);
    minor3 = _mm_sub_ps(minor3, _mm_mul_ps(row2, tmp1));
    // -----------------------------------------------
    tmp1 = _mm_mul_ps(row0, row3);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
    minor1 = _mm_sub_ps(minor1, _mm_mul_ps(row2, tmp1));
    minor2 = _mm_add_ps(_mm_mul_ps(row1, tmp1), minor2);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
    minor1 = _mm_add_ps(_mm_mul_ps(row2, tmp1), minor1);
    minor2 = _mm_sub_ps(minor2, _mm_mul_ps(row1, tmp1));
    // -----------------------------------------------
    tmp1 = _mm_mul_ps(row0, row2);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
    minor1 = _mm_add_ps(_mm_mul_ps(row3, tmp1), minor1);
    minor3 = _mm_sub_ps(minor3, _mm_mul_ps(row1, tmp1));
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
    minor1 = _mm_sub_ps(minor1, _mm_mul_ps(row3, tmp1));
    minor3 = _mm_add_ps(_mm_mul_ps(row1, tmp1), minor3);
    // -----------------------------------------------
    det = _mm_mul_ps(row0, minor0);
    det = _mm_add_ps(_mm_shuffle_ps(det, det, 0x4E), det);
    det = _mm_add_ss(_mm_shuffle_ps(det, det, 0xB1), det);
    tmp1 = _mm_rcp_ss(det);
    det = _mm_sub_ss(_mm_add_ss(tmp1, tmp1), _mm_mul_ss(det, _mm_mul_ss(tmp1, tmp1)));
    det = _mm_shuffle_ps(det, det, 0x00);
    minor0 = _mm_mul_ps(det, minor0);
    _mm_storel_pi((__m64*)(src), minor0);
    _mm_storeh_pi((__m64*)(src + 2), minor0);
    minor1 = _mm_mul_ps(det, minor1);
    _mm_storel_pi((__m64*)(src + 4), minor1);
    _mm_storeh_pi((__m64*)(src + 6), minor1);
    minor2 = _mm_mul_ps(det, minor2);
    _mm_storel_pi((__m64*)(src + 8), minor2);
    _mm_storeh_pi((__m64*)(src + 10), minor2);
    minor3 = _mm_mul_ps(det, minor3);
    _mm_storel_pi((__m64*)(src + 12), minor3);
    _mm_storeh_pi((__m64*)(src + 14), minor3);
}

bool gluInvertMatrix(const double m[16], double invOut[16])
{
    double inv[16], det;
    int i;

    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14]
        + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];

    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14]
        - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];

    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13]
        + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];

    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13]
        - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];

    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14]
        - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];

    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14]
        + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];

    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13]
        - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];

    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13]
        + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];

    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7]
        - m[13] * m[3] * m[6];

    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14]
        - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];

    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13]
        + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];

    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13]
        - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7]
        + m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7]
        - m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7]
        + m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6]
        - m[8] * m[2] * m[5];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    if (det == 0)
        return false;

    det = 1.0 / det;

    for (i = 0; i < 16; i++)
        invOut[i] = inv[i] * det;

    return true;
}

void stokes_K(int k, const F64View2D& chi, f64 chiI, f64 K[4][4])
{
    for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 4; ++i)
            K[j][i] = 0.0;
    K[0][1] = chi(1, k);
    K[0][2] = chi(2, k);
    K[0][3] = chi(3, k);

    K[1][2] = chi(6, k);
    K[1][3] = chi(5, k);
    K[2][3] = chi(4, k);

    for (int j = 0; j < 3; ++j)
    {
        for (int i = j + 1; i < 4; ++i)
        {
            K[j][i] /= chiI;
            K[i][j] = K[j][i];
        }
    }

    K[1][3] *= -1.0;
    K[2][1] *= -1.0;
    K[3][2] *= -1.0;
}

inline void prod(f64 a[4][4], f64 b[4][4], f64 c[4][4])
{
    for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 4; ++i)
            c[j][i] = 0.0;

    for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 4; ++i)
            for (int k = 0; k < 4; ++k)
                c[j][i] += a[k][i] * b[j][k];
}

inline void prod(f64 a[4][4], f64 b[4], f64 c[4])
{
    for (int i = 0; i < 4; ++i)
        c[i] = 0.0;

    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 4; ++k)
            c[i] += a[i][k] * b[k];
}

inline void prod(f32 a[4][4], f64 b[4], f64 c[4])
{
    for (int i = 0; i < 4; ++i)
        c[i] = 0.0;

    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 4; ++k)
            c[i] += f64(a[i][k]) * b[k];
}

#define GLU_MAT 1
void piecewise_stokes_bezier3_1d_impl(FormalDataStokes* fd, f64 zmu, bool toObs, f64 Istart[4], bool polarisedFrequency)
{
    JasUnpack((*fd), atmos, chi, S, I);
    const auto& height = atmos->height;
    const int Ndep = atmos->Nspace;

    // clang-format off
    constexpr f64 id[4][4] = { { 1.0, 0.0, 0.0, 0.0 },
                               { 0.0, 1.0, 0.0, 0.0 },
                               { 0.0, 0.0, 1.0, 0.0 },
                               { 0.0, 0.0, 0.0, 1.0 } };
    // clang-format on

    auto slice_s4 = [&S](int k, f64 slice[4]) {
        for (int i = 0; i < 4; ++i)
        {
            slice[i] = S(i, k);
        }
    };

    /* --- Distinguish between rays going from BOTTOM to TOP
            (to_obs == TRUE), and vice versa --      -------------- */

    int dk = -1;
    int k_start = Ndep - 1;
    int k_end = 0;
    if (!toObs)
    {
        dk = 1;
        k_start = 0;
        k_end = Ndep - 1;
    }

    /* --- Boundary conditions --                        -------------- */

    for (int n = 0; n < 4; ++n)
        I(n, k_start) = Istart[n];

    int k = k_start + dk;
    f64 ds_uw = abs(height(k) - height(k - dk)) * zmu;
    f64 ds_dw = abs(height(k + dk) - height(k)) * zmu;
    f64 dx_uw = (chi(0, k) - chi(0, k - dk)) / ds_uw;
    f64 dx_c = Bezier::cent_deriv(ds_uw, ds_dw, chi(0, k - dk), chi(0, k), chi(0, k + dk));
    f64 c1 = max(chi(0, k) - (ds_uw / 3.0) * dx_c, 0.0);
    f64 c2 = max(chi(0, k - dk) + (ds_uw / 3.0) * dx_uw, 0.0);
    f64 dtau_uw = ds_uw * (chi(0, k) + chi(0, k - dk) + c1 + c2) * 0.25;

    f64 Ku[4][4], K0[4][4], Su[4], S0[4];
    f64 dKu[4][4], dK0[4][4], dSu[4], dS0[4];
    stokes_K(k_start, chi, chi(0, k_start), Ku);
    stokes_K(k, chi, chi(0, k), K0);
    // memset(Ku[0], 0, 16*sizeof(f64));
    // memset(K0[0], 0, 16*sizeof(f64));
    slice_s4(k_start, Su);
    slice_s4(k, S0);

    for (int n = 0; n < 4; ++n)
    {
        dSu[n] = (S0[n] - Su[n]) / dtau_uw;
        for (int m = 0; m < 4; ++m)
            dKu[n][m] = (K0[n][m] - Ku[n][m]) / dtau_uw;
    }

    f64 ds_dw2 = 0.0;
    f64 dtau_dw = 0.0;
    auto dx_downwind = [&ds_dw, &ds_dw2, &chi, &k, dk] {
        return Bezier::cent_deriv(ds_dw, ds_dw2, chi(0, k), chi(0, k + dk), chi(0, k + 2 * dk));
    };

    f64 Kd[4][4], A[4][4], Ma[4][4], Mb[4][4], Mc[4][4], V0[4], V1[4], Sd[4];
#if GLU_MAT
    f64 Md[4][4], Mdi[4][4];
#else
    f32 Md[4][4];
#endif
    for (; k != k_end - dk; k += dk)
    {
        ds_dw2 = abs(height(k + 2 * dk) - height(k + dk)) * zmu;
        f64 dx_dw = dx_downwind();
        c1 = max(chi(0, k) + (ds_dw / 3.0) * dx_c, 0.0);
        c2 = max(chi(0, k + dk) - (ds_dw / 3.0) * dx_dw, 0.0);
        dtau_dw = ds_dw * (chi(0, k) + chi(0, k + dk) + c1 + c2) * 0.25;

        f64 alpha, beta, gamma, edt, eps;
        Bezier::Bezier3_coeffs(dtau_uw, &alpha, &beta, &gamma, &eps, &edt);

        stokes_K(k + dk, chi, chi(0, k + dk), Kd);
        // memset(Kd[0], 0, 16*sizeof(f64));
        slice_s4(k + dk, Sd);

        Bezier::cent_deriv(dK0, dtau_uw, dtau_dw, Ku, K0, Kd);
        Bezier::cent_deriv(dS0, dtau_uw, dtau_dw, Su, S0, Sd);

        prod(Ku, Ku, Ma); // Ma = Ku @ Ku
        prod(K0, K0, A); // A = K0 @ K0

        // c1 = S0[0] - (dtau_uw/3.0) * dS0[0];
        // c2 = Su[0] + (dtau_uw/3.0) * dSu[0];
        // I(0, k) = I(0, k-dk) * edt + alpha * S0[0] + beta * Su[0] + gamma * c1 + eps * c2;

        for (int j = 0; j < 4; ++j)
        {
            for (int i = 0; i < 4; ++i)
            {
                // A in paper (LHS of system)
                Md[j][i] = id[j][i] + alpha * K0[j][i]
                    - gamma * -(dtau_uw / 3.0 * (A[j][i] + dK0[j][i] + K0[j][i]) + K0[j][i]);

                // Terms to be multiplied by I(:,k-dk) in B: (exp(-dtau) + beta*Ku + epsilon*\bar{f}_k)
                Ma[j][i] = edt * id[j][i] - beta * Ku[j][i]
                    + eps * (dtau_uw / 3.0 * (Ma[j][i] + dKu[j][i] + Ku[j][i]) - Ku[j][i]);

                // Terms to be multiplied by S(:,k-dk) in B i.e. f_k
                Mb[j][i] = beta * id[j][i] + eps * (id[j][i] - dtau_uw / 3.0 * Ku[j][i]);

                // Terms to be multiplied by S(:,k) in B i.e. e_k
                Mc[j][i] = alpha * id[j][i] + gamma * (id[j][i] + dtau_uw / 3.0 * K0[j][i]);
            }
        }

        // printf("%e, %e, %e, %e\n", K0[0][0], K0[1][0], K0[0][1], K0[3][2]);

        for (int i = 0; i < 4; ++i)
        {
            V0[i] = 0.0;
            for (int j = 0; j < 4; ++j)
                V0[i] += Ma[i][j] * I(j, k - dk) + Mb[i][j] * Su[j] + Mc[i][j] * S0[j];

            V0[i] += dtau_uw / 3.0 * (eps * dSu[i] - gamma * dS0[i]);
        }

#if GLU_MAT
        gluInvertMatrix(Md[0], Mdi[0]);
        prod(Mdi, V0, V1);
#else
        SIMD_MatInv(Md[0]);
        prod(Md, V0, V1);
#endif

        for (int i = 0; i < 4; ++i)
            I(i, k) = V1[i];

        memcpy(Su, S0, 4 * sizeof(f64));
        memcpy(S0, Sd, 4 * sizeof(f64));
        memcpy(dSu, dS0, 4 * sizeof(f64));

        memcpy(Ku[0], K0[0], 16 * sizeof(f64));
        memcpy(K0[0], Kd[0], 16 * sizeof(f64));
        memcpy(dKu[0], dK0[0], 16 * sizeof(f64));

        dtau_uw = dtau_dw;
        ds_uw = ds_dw;
        ds_dw = ds_dw2;
        dx_uw = dx_c;
        dx_c = dx_dw;
    }
    // NOTE(cmo): Need to handle last 2 points here
    k = k_end - dk;
    f64 dx_dw = (chi(0, k + dk) - chi(0, k)) / ds_dw;
    c1 = max(chi(0, k) + (ds_dw / 3.0) * dx_c, 0.0);
    c2 = max(chi(0, k + dk) - (ds_dw / 3.0) * dx_dw, 0.0);
    dtau_dw = ds_dw * (chi(0, k) + chi(0, k + dk) + c1 + c2) * 0.25;

    f64 alpha, beta, gamma, edt, eps;
    Bezier::Bezier3_coeffs(dtau_uw, &alpha, &beta, &gamma, &eps, &edt);

    stokes_K(k + dk, chi, chi(0, k + dk), Kd);
    // memset(Kd[0], 0, 16*sizeof(f64));
    slice_s4(k + dk, Sd);

    Bezier::cent_deriv(dK0, dtau_uw, dtau_dw, Ku, K0, Kd);
    Bezier::cent_deriv(dS0, dtau_uw, dtau_dw, Su, S0, Sd);

    prod(Ku, Ku, Ma); // Ma = Ku @ Ku
    prod(K0, K0, A); // A = K0 @ K0

    // c1 = max(S(0, k) - (dtau_uw/3.0) * dS0[0], 0.0);
    // c2 = max(S(0, k-dk) + (dtau_uw/3.0) * dSu[0], 0.0);
    // I(0, k) = I(0, k-dk) * edt + alpha * S(0, k) + beta * S(0, k-dk) + gamma * c1 + eps * c2;

    for (int j = 0; j < 4; ++j)
    {
        for (int i = 0; i < 4; ++i)
        {
            // A in paper (LHS of system)
            Md[j][i]
                = id[j][i] + alpha * K0[j][i] - gamma * -(dtau_uw / 3.0 * (A[j][i] + dK0[j][i] + K0[j][i]) + K0[j][i]);

            // Terms to be multiplied by I(:,k-dk) in B: (exp(-dtau) + beta + \bar{f}_k)
            Ma[j][i] = edt * id[j][i] - beta * Ku[j][i]
                + eps * (dtau_uw / 3.0 * (Ma[j][i] + dKu[j][i] + Ku[j][i]) - Ku[j][i]);

            // Terms to be multiplied by S(:,k-dk) in B i.e. f_k
            Mb[j][i] = beta * id[j][i] + eps * (id[j][i] - dtau_uw / 3.0 * Ku[j][i]);

            // Terms to be multiplied by S(:,k) in B i.e. e_k
            Mc[j][i] = alpha * id[j][i] + gamma * (id[j][i] + dtau_uw / 3.0 * K0[j][i]);
        }
    }

    for (int i = 0; i < 4; ++i)
    {
        V0[i] = 0.0;
        for (int j = 0; j < 4; ++j)
            V0[i] += Ma[i][j] * I(j, k - dk) + Mb[i][j] * Su[j] + Mc[i][j] * S0[j];

        V0[i] += dtau_uw / 3.0 * (eps * dSu[i] - gamma * dS0[i]);
    }

#if GLU_MAT
    gluInvertMatrix(Md[0], Mdi[0]);
    prod(Mdi, V0, V1);
#else
    SIMD_MatInv(Md[0]);
    prod(Md, V0, V1);
#endif

    for (int i = 0; i < 4; ++i)
        I(i, k) = V1[i];

    memcpy(Su, S0, 4 * sizeof(f64));
    memcpy(S0, Sd, 4 * sizeof(f64));
    memcpy(dSu, dS0, 4 * sizeof(f64));

    memcpy(Ku[0], K0[0], 16 * sizeof(f64));
    memcpy(K0[0], Kd[0], 16 * sizeof(f64));
    memcpy(dKu[0], dK0[0], 16 * sizeof(f64));

    dtau_uw = dtau_dw;
    ds_uw = ds_dw;
    ds_dw = ds_dw2;
    dx_uw = dx_c;
    dx_c = dx_dw;

    // Piecewise linear on end
    k = k_end;
    dtau_uw = 0.5 * zmu * (chi(0, k) + chi(0, k - dk)) * abs(height(k) - height(k - dk));

    f64 w[2];
    w2(dtau_uw, w);
    for (int n = 0; n < 4; ++n)
        V0[n] = w[0] * S(n, k) - w[1] * dSu[n];

    for (int n = 0; n < 4; ++n)
    {
        for (int m = 0; m < 4; ++m)
        {
            A[n][m] = -w[1] / dtau_uw * Ku[n][m];
            Md[n][m] = (w[0] - w[1] / dtau_uw) * K0[n][m];
        }
        A[n][n] = 1.0 - w[0];
        Md[n][n] = 1.0;
    }

    for (int n = 0; n < 4; ++n)
        for (int m = 0; m < 4; ++m)
            V0[n] += A[n][m] * I(m, k - dk);

#if GLU_MAT
    gluInvertMatrix(Md[0], Mdi[0]);
    prod(Mdi, V0, V1);
#else
    SIMD_MatInv(Md[0]);
    prod(Md, V0, V1);
#endif

    for (int n = 0; n < 4; ++n)
        I(n, k) = V1[n];
}

namespace LwInternal
{
void piecewise_stokes_bezier3_1d(FormalDataStokes* fd, int mu, bool toObs, f64 wav, bool polarisedFrequency)
{
    if (!polarisedFrequency)
    {
        piecewise_bezier3_1d(&fd->fdIntens, mu, toObs, wav);
        return;
    }

    JasUnpack((*fd), atmos, chi);
    // This is 1.0 here, as we are normally effectively rolling in the averaging
    // factor for dtau, whereas it's explicit in this solver
    f64 zmu = 1.0 / atmos->muz(mu);
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
    f64 dtau_uw = 0.5 * zmu * (chi(0, kStart) + chi(0, kStart + dk)) * abs(height(kStart) - height(kStart + dk));

    f64 Iupw[4] = { 0.0, 0.0, 0.0, 0.0 };
    if (toObs && atmos->lowerBc == THERMALISED)
    {
        f64 Bnu[2];
        int Nspace = atmos->Nspace;
        planck_nu(2, &atmos->temperature(Nspace - 2), wav, Bnu);
        Iupw[0] = Bnu[1] - (Bnu[0] - Bnu[1]) / dtau_uw;
    }
    else if (!toObs && atmos->upperBc == THERMALISED)
    {
        f64 Bnu[2];
        planck_nu(2, &atmos->temperature(0), wav, Bnu);

        Iupw[0] = Bnu[0] - (Bnu[1] - Bnu[0]) / dtau_uw;
    }

    piecewise_stokes_bezier3_1d_impl(fd, zmu, toObs, Iupw, polarisedFrequency);
}
}

namespace GammaFsCores
{
f64 stokes_fs_core(StokesCoreData& data, int la, bool updateJ)
{
    JasUnpack(*data, atmos, spect, fd, background);
    JasUnpack(*data, activeAtoms, lteAtoms, JDag);
    JasUnpack(data, chiTot, etaTot, Uji, Vij, Vji, I, S);

    const int Nspace = atmos.Nspace;
    const int Nrays = atmos.Nrays;
    F64View J = spect.J(la);
    if (updateJ)
    {
        JDag = spect.J(la);
        J.fill(0.0);
    }

    for (int a = 0; a < activeAtoms.size(); ++a)
        activeAtoms[a]->setup_wavelength(la);
    for (int a = 0; a < lteAtoms.size(); ++a)
        lteAtoms[a]->setup_wavelength(la);

    // NOTE(cmo): If we only have continua then opacity is angle independent
    bool continuaOnly = true;
    for (int a = 0; a < activeAtoms.size(); ++a)
    {
        auto& atom = *activeAtoms[a];
        for (int kr = 0; kr < atom.Ntrans; ++kr)
        {
            auto& t = *atom.trans[kr];
            if (!t.active(la))
                continue;
            continuaOnly = continuaOnly && (t.type == CONTINUUM);
        }
    }
    for (int a = 0; a < lteAtoms.size(); ++a)
    {
        auto& atom = *lteAtoms[a];
        for (int kr = 0; kr < atom.Ntrans; ++kr)
        {
            auto& t = *atom.trans[kr];
            if (!t.active(la))
                continue;
            continuaOnly = continuaOnly && (t.type == CONTINUUM);
        }
    }

    f64 dJMax = 0.0;
    for (int mu = 0; mu < Nrays; ++mu)
    {
        for (int toObsI = 0; toObsI < 2; toObsI += 1)
        {
            bool toObs = (bool)toObsI;
            bool polarisedFrequency = false;
            // const f64 sign = If toObs Then 1.0 Else -1.0 End;
            if (!continuaOnly || (continuaOnly && (mu == 0 && toObsI == 0)))
            {
                chiTot.fill(0.0);
                etaTot.fill(0.0);

                for (int a = 0; a < activeAtoms.size(); ++a)
                {
                    auto& atom = *activeAtoms[a];
                    for (int kr = 0; kr < atom.Ntrans; ++kr)
                    {
                        auto& t = *atom.trans[kr];
                        if (!t.active(la))
                            continue;

                        t.uv(la, mu, toObs, Uji, Vij, Vji);

                        for (int k = 0; k < Nspace; ++k)
                        {
                            f64 chi = atom.n(t.i, k) * Vij(k) - atom.n(t.j, k) * Vji(k);
                            f64 eta = atom.n(t.j, k) * Uji(k);

                            chiTot(0, k) += chi;
                            etaTot(0, k) += eta;

                            if (t.type == TransitionType::LINE && t.polarised)
                            {
                                polarisedFrequency = true;
                                int lt = t.lt_idx(la);
                                f64 chiNoProfile = chi / t.phi(lt, mu, toObs, k);
                                chiTot(1, k) += chiNoProfile * t.phiQ(lt, mu, toObs, k);
                                chiTot(2, k) += chiNoProfile * t.phiU(lt, mu, toObs, k);
                                chiTot(3, k) += chiNoProfile * t.phiV(lt, mu, toObs, k);
                                chiTot(4, k) += chiNoProfile * t.psiQ(lt, mu, toObs, k);
                                chiTot(5, k) += chiNoProfile * t.psiU(lt, mu, toObs, k);
                                chiTot(6, k) += chiNoProfile * t.psiV(lt, mu, toObs, k);

                                f64 etaNoProfile = eta / t.phi(lt, mu, toObs, k);
                                etaTot(1, k) += etaNoProfile * t.phiQ(lt, mu, toObs, k);
                                etaTot(2, k) += etaNoProfile * t.phiU(lt, mu, toObs, k);
                                etaTot(3, k) += etaNoProfile * t.phiV(lt, mu, toObs, k);
                            }
                        }
                    }
                }
                for (int a = 0; a < lteAtoms.size(); ++a)
                {
                    auto& atom = *lteAtoms[a];
                    for (int kr = 0; kr < atom.Ntrans; ++kr)
                    {
                        auto& t = *atom.trans[kr];
                        if (!t.active(la))
                            continue;

                        t.uv(la, mu, toObs, Uji, Vij, Vji);

                        for (int k = 0; k < Nspace; ++k)
                        {
                            f64 chi = atom.n(t.i, k) * Vij(k) - atom.n(t.j, k) * Vji(k);
                            f64 eta = atom.n(t.j, k) * Uji(k);

                            chiTot(0, k) += chi;
                            etaTot(0, k) += eta;

                            if (t.type == TransitionType::LINE && t.polarised)
                            {
                                polarisedFrequency = true;
                                int lt = t.lt_idx(la);
                                f64 chiNoProfile = chi / t.phi(lt, mu, toObs, k);
                                chiTot(1, k) += chiNoProfile * t.phiQ(lt, mu, toObs, k);
                                chiTot(2, k) += chiNoProfile * t.phiU(lt, mu, toObs, k);
                                chiTot(3, k) += chiNoProfile * t.phiV(lt, mu, toObs, k);
                                chiTot(4, k) += chiNoProfile * t.psiQ(lt, mu, toObs, k);
                                chiTot(5, k) += chiNoProfile * t.psiU(lt, mu, toObs, k);
                                chiTot(6, k) += chiNoProfile * t.psiV(lt, mu, toObs, k);

                                f64 etaNoProfile = eta / t.phi(lt, mu, toObs, k);
                                etaTot(1, k) += etaNoProfile * t.phiQ(lt, mu, toObs, k);
                                etaTot(2, k) += etaNoProfile * t.phiU(lt, mu, toObs, k);
                                etaTot(3, k) += etaNoProfile * t.phiV(lt, mu, toObs, k);
                            }
                        }
                    }
                }

                for (int k = 0; k < Nspace; ++k)
                {
                    chiTot(0, k) += background.chi(la, k);
                    S(0, k) = (etaTot(0, k) + background.eta(la, k) + background.sca(la, k) * JDag(k)) / chiTot(0, k);
                }
                if (polarisedFrequency)
                {
                    for (int n = 1; n < 4; ++n)
                    {
                        for (int k = 0; k < Nspace; ++k)
                        {
                            S(n, k) = etaTot(n, k) / chiTot(0, k);
                        }
                    }
                }
            }

#if 1
            piecewise_stokes_bezier3_1d(&fd, mu, toObs, spect.wavelength(la), polarisedFrequency);
            spect.I(la, mu) = I(0, 0);
            spect.Quv(0, la, mu) = I(1, 0);
            spect.Quv(1, la, mu) = I(2, 0);
            spect.Quv(2, la, mu) = I(3, 0);
#else
            // NOTE(cmo): Checking with the normal FS and just using the first row of ezach of the matrices does indeed
            // produce the correct result
            piecewise_bezier3_1d(&fd.fdIntens, mu, toObs, spect.wavelength(la));
            spect.I(la, mu) = I(0, 0);
            spect.Quv(0, la, mu) = 0.0;
            spect.Quv(1, la, mu) = 0.0;
            spect.Quv(2, la, mu) = 0.0;
#endif

            // TODO(cmo): Rates?
            if (updateJ)
            {
                for (int k = 0; k < Nspace; ++k)
                {
                    J(k) += 0.5 * atmos.wmu(mu) * I(0, k);
                }
            }
        }
    }
    if (updateJ)
    {
        for (int k = 0; k < Nspace; ++k)
        {
            f64 dJ = abs(1.0 - JDag(k) / J(k));
            dJMax = max(dJ, dJMax);
        }
    }
    return dJMax;
}
}

f64 formal_sol_full_stokes(Context& ctx, bool updateJ)
{
    // feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
    JasUnpack(*ctx, atmos, spect, background);
    JasUnpack(ctx, activeAtoms, lteAtoms);

    if (!atmos.B)
        assert(false && "Magnetic field required");

    const int Nspace = atmos.Nspace;
    const int Nrays = atmos.Nrays;
    const int Nspect = spect.wavelength.shape(0);
    // auto Iplus = spect.I;

    F64Arr2D chiTot = F64Arr2D(7, Nspace);
    F64Arr2D etaTot = F64Arr2D(4, Nspace);
    F64Arr2D S = F64Arr2D(4, Nspace);
    F64Arr Uji = F64Arr(Nspace);
    F64Arr Vij = F64Arr(Nspace);
    F64Arr Vji = F64Arr(Nspace);
    F64Arr2D I = F64Arr2D(4, Nspace);
    F64Arr JDag = F64Arr(Nspace);
    FormalDataStokes fd;
    fd.atmos = &atmos;
    fd.chi = chiTot;
    fd.S = S;
    fd.I = I;
    fd.fdIntens.atmos = fd.atmos;
    fd.fdIntens.chi = fd.chi(0);
    fd.fdIntens.S = fd.S(0);
    fd.fdIntens.I = fd.I(0);
    StokesCoreData core;
    JasPackPtr(core, atmos, spect, fd, background);
    JasPackPtr(core, activeAtoms, lteAtoms, JDag);
    JasPack(core, chiTot, etaTot, Uji, Vij, Vji, I, S);

    printf("%d, %d, %d\n", Nspace, Nrays, Nspect);

    f64 dJMax = 0.0;
    for (int la = 0; la < Nspect; ++la)
    {
        f64 dJ = GammaFsCores::stokes_fs_core(core, la, true);
        dJMax = max(dJ, dJMax);
    }
    return dJMax;
}