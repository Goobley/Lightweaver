#ifndef CMO_LW_MISC_HPP
#define CMO_LW_MISC_HPP

#include "CmoArray.hpp"
#include "Constants.hpp"
#include "Faddeeva.hh"

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

struct Background
{
    F64View2D chi;
    F64View2D eta;
    F64View2D sca;
};

struct Spectrum
{
    F64View wavelength;
    F64View3D I;
    F64View4D Quv;
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


#else
#endif