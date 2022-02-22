#ifndef CMO_LW_INTERNAL_HPP
#define CMO_LW_INTERNAL_HPP
#include "Constants.hpp"
#include "CmoArray.hpp"
#include "LwFormalInterface.hpp"
#include <vector>

struct Atmosphere;
struct Spectrum;
struct Background;
struct Atom;
struct DepthData;

namespace LwInternal
{
    struct FormalData
    {
        int width;
        Atmosphere* atmos;
        F64View chi;
        F64View S;
        F64View I;
        F64View Psi;
        Interp2d interp;

        FormalData() : width(1),
                       atmos(nullptr),
                       chi(),
                       S(),
                       I(),
                       Psi(),
                       interp()
        {}
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
        LwFsFn formal_solver;
        Atmosphere* atmos;
        Spectrum* spect;
        FormalData* fd;
        Background* background;
        DepthData* depthData;
        std::vector<Atom*>* activeAtoms;
        std::vector<Atom*>* detailedAtoms;
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
        F64View2D JRest;

        IntensityCoreData() = default;
    };

    struct StokesCoreData
    {
        Atmosphere* atmos;
        Spectrum* spect;
        FormalDataStokes* fd;
        Background* background;
        std::vector<Atom*>* activeAtoms;
        std::vector<Atom*>* detailedAtoms;
        F64Arr* JDag;
        F64View2D chiTot;
        F64View2D etaTot;
        F64View Uji;
        F64View Vij;
        F64View Vji;
        F64View2D I;
        F64View2D S;
        F64View2D J20;
        F64Arr* J20Dag;
    };

    inline void w2(f64 dtau, f64* w)
    {
        constexpr f64 third = 1.0 / 3.0;
        f64 expdt;

        if (dtau < 5.0E-4)
        {
            w[0] = dtau * (1.0 - 0.5 * dtau);
            w[1] = square(dtau) * (0.5 - dtau * third);
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
        FsOnly = 0,
        UpdateJ = 1 << 0,
        UpdateRates = 1 << 1,
        PrdOnly = 1 << 2,
        PureLambdaIteration = 1 << 3,
        UpOnly = 1 << 4,
    };
    constexpr inline FsMode
    operator|(FsMode a, FsMode b)
    {
        return static_cast<FsMode>(static_cast<u32>(a) | static_cast<u32>(b));
    }

    void piecewise_linear_1d(FormalData* fd, int la, int mu, bool toObs,
                             const F64View1D& wave);
    void piecewise_besser_1d(FormalData* fd, int la, int mu, bool toObs,
                              const F64View1D& wave);
    void piecewise_bezier3_1d(FormalData* fd, int la, int mu, bool toObs,
                              const F64View1D& wave);
    void piecewise_linear_2d(FormalData* fd, int la, int mu, bool toObs,
                             const F64View1D& wave);
    void piecewise_besser_2d(FormalData* fd, int la, int mu, bool toObs,
                             const F64View1D& wave);
    void piecewise_parabolic_2d(FormalData* fd, int la, int mu, bool toObs, f64 wav);
    void piecewise_stokes_bezier3_1d(FormalDataStokes* fd, int la, int mu, bool toObs, f64 wav, bool polarisedFrequency);
    f64 interp_linear_2d(const IntersectionData&, const IntersectionResult&, const F64View2D&);
    f64 interp_besser_2d(const IntersectionData&, const IntersectionResult&, const F64View2D&);
}
#else
#endif