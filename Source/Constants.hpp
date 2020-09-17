#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

namespace Constants
{
// Speed of light [m/s]
constexpr double CLight = 2.99792458E+08;
// Planck's constant [Js]
constexpr double HPlanck = 6.6260755E-34;
constexpr double HC = HPlanck * CLight;
// Boltzman's constant [J/K]
constexpr double KBoltzmann = 1.380658E-23;
// Atomic mass unit [kg]
constexpr double Amu = 1.6605402E-27;
// Electron mass [kg]
constexpr double MElectron = 9.1093897E-31;
// Electron charge [C]
constexpr double QElectron = 1.60217733E-19 ;
// Vacuum permittivity [F/m]
constexpr double Epsilon0 = 8.854187817E-12;
// Magnetic induct. of vac.
constexpr double Mu0 = 1.2566370614E-06;
// Bohr radius [m]
constexpr double RBohr = 5.29177349E-11;
// Ion. pot. Hydrogen [J]
constexpr double ERydberg = 2.1798741E-18;
// One electronVolt [J]
constexpr double EV = 1.60217733E-19;
// log10(e) * eV/k [K^-1]
constexpr double Theta0 = 5.03974756E+03;
// polarizability of Hydrogen [Fm^2]
constexpr double ABarH = 7.42E-41;
// pi -- the circle constant
constexpr double Pi = 3.14159265358979323846264338327950288;
// ln(10)
constexpr double Log10 = 2.30258509299404568401799145468436421;
// Ionization energy Hmin in [J]
constexpr double E_ION_HMIN = 0.754*EV;

constexpr double NM_TO_M = 1.0E-09;
constexpr double CM_TO_M = 1.0E-02;
constexpr double KM_TO_M = 1.0E+03;
constexpr double ERG_TO_JOULE = 1.0E-07;
constexpr double G_TO_KG = 1.0E-03;
constexpr double MICRON_TO_NM = 1.0E+03;
constexpr double MEGABARN_TO_M2 = 1.0E-22;
}

#ifndef CMO_NO_TYPEDEFS
#include <cstdint>
#include <cstddef>
#include <cmath>
typedef int8_t i8;
typedef int16_t i16;
#ifdef _WIN32
    // NOTE(cmo): Fix for silly Windows types and int not being compatible with
    // long (which is correct due to C++ standard, but npy_int32 is defined to
    // long on Win32 and we need compatibility).
    typedef long i32;
#else
    typedef int32_t i32;
#endif
typedef int64_t i64;
typedef long long int longboi;
typedef float f32;
typedef double f64;
typedef long double f128;
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef unsigned uint;
using std::abs;
using std::sin;
using std::cos;
using std::tan;
using std::pow;
using std::sqrt;
#endif

#ifndef CMO_NO_MATH
template <typename T>
constexpr
T square(T val)
{
    return val * val;
}

template <typename T>
constexpr
T cube(T val)
{
    return val * val * val;
}

#include <algorithm>
using std::min;
using std::max;

// template <typename T>
// constexpr
// T max(T a, T b)
// {
//     return a < b ? b : a;
// }

// template <typename T>
// constexpr
// T min(T a, T b)
// {
//     return a < b ? a : b;
// }

template <typename T, typename U>
constexpr
T max_idx(T a, T b, U& aIdx, U bIdx)
{
    if (a < b)
    {
        aIdx = bIdx;
        return b;
    }
    else
        return a;
}

template <typename T, typename U>
constexpr
T min_idx(T a, T b, U& aIdx, U bIdx)
{
    if (a < b)
    {
        return a;
    }
    else
    {
        aIdx = bIdx;
        return b;
    }
}
#endif

#else
#endif