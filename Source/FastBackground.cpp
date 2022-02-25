#include "FastBackground.hpp"
#include "Background.hpp"
#include "Utils.hpp"


void bf_opacities(BackgroundData* bd, const std::vector<BackgroundAtom>& atoms,
                  Atmosphere* atmos, int laStart, int laEnd)
{
    if (atoms.size() == 0)
        return;
    JasUnpack((*bd), wavelength, chi, eta);

    if (laStart < 0 && laEnd < 0)
    {
        laStart = 0;
        laEnd = wavelength.shape(0);
    }

    namespace C = Constants;

    constexpr f64 hc_k = C::HC / (C::KBoltzmann * C::NM_TO_M);
    constexpr f64 twohc = (2.0 * C::HC) / cube(C::NM_TO_M);

    for (int a = 0; a < atoms.size(); ++a)
    {
        auto& atom = atoms[a];
        for (int c = 0; c < atom.continua.size(); ++c)
        {
            auto& cont = atom.continua[c];
            int cLaStart = cont.laStart;
            int cLaEnd = cont.laEnd;
            if (cLaStart < laStart)
                cLaStart = laStart;
            if (cLaEnd > laEnd)
                cLaEnd = laEnd;

            if (cLaStart >= cLaEnd)
                continue;

            for (int la = cLaStart; la < cLaEnd; ++la)
            {
                f64 alpha = cont.alpha(la);
                f64 hc_kla = hc_k / wavelength(la);
                f64 twohnu3_c2 = twohc / cube(wavelength(la));

                for (int k = 0; k < atmos->Nspace; ++k)
                {
                    f64 expla = exp(-hc_kla / atmos->temperature(k));
                    f64 gijk = atom.nStar(cont.i, k) / atom.nStar(cont.j, k) * expla;
                    chi(la, k) += alpha * (1.0 - expla) * atom.n(cont.i, k);
                    eta(la, k) += twohnu3_c2 * gijk * alpha * atom.n(cont.j, k);
                }
            }
        }
    }
}


void rayleigh_scattering(BackgroundData* bd, const std::vector<BackgroundAtom>& atoms,
                         Atmosphere* atmos, int laStart, int laEnd)
{
    if (atoms.size() == 0)
        return;
    JasUnpack((*bd), wavelength, scatt);

    if (laStart < 0 && laEnd < 0)
    {
        laStart = 0;
        laEnd = wavelength.shape(0);
    }
    namespace C = Constants;

    constexpr f64 c = 2.0 * C::Pi * (C::QElectron / C::Epsilon0)
                      * C::QElectron / C::MElectron / C::CLight;
    const f64 sigmaE = 8.0 * C::Pi / 3.0 * std::pow(C::QElectron / (sqrt(4.0 * C::Pi * C::Epsilon0) * (sqrt(C::MElectron) * C::CLight)), 4.0);

    for (int a = 0; a < atoms.size(); ++a)
    {
        auto& atom = atoms[a];
        for (int l = 0; l < atom.resonanceScatterers.size(); ++l)
        {
            auto& line = atom.resonanceScatterers[l];
            if (wavelength(laEnd-1) <= line.lambdaMax)
                continue;

            for (int la = laStart; la < laEnd; ++la)
            {
                // NOTE(cmo): This is anticipating small wavelength ranges so a
                // linear search is perfectly fine.
                if (wavelength(la) <= line.lambdaMax)
                    continue;
                f64 lambda2 = 1.0 / (square(wavelength(la) / line.lambda0) - 1.0);
                f64 f = line.Aji * line.gRatio * square(line.lambda0 * C::NM_TO_M) / c;
                f64 sigmaRayleigh = f * square(lambda2) * sigmaE;
                for (int k = 0; k < atmos->Nspace; ++k)
                    scatt(la, k) += sigmaRayleigh * atom.n(0, k);
            }
        }
    }
}

void FastBackgroundContext::basic_background(BackgroundData* bd, Atmosphere* atmos)
{
    if (Nthreads <= 1)
    {
        ::basic_background(bd, atmos);
    }
    else
    {
        struct BasicBackgroundData
        {
            BackgroundData* bd;
            Atmosphere* atmos;
        };

        bd->chi.fill(0.0);
        bd->eta.fill(0.0);
        bd->scatt.fill(0.0);

        auto background_handler = [](void* data, enki::TaskScheduler* s,
                                     enki::TaskSetPartition p, u32 threadId)
        {
            BasicBackgroundData* args = (BasicBackgroundData*)data;
            ::basic_background(args->bd, args->atmos, p.start, p.end);
        };

        BasicBackgroundData args{bd, atmos};

        {
            LwTaskSet bgOpacities((void*)&args, &sched, bd->wavelength.shape(0),
                                20, background_handler);
            sched.AddTaskSetToPipe(&bgOpacities);
            sched.WaitforTask(&bgOpacities);
        }
    }
}

void FastBackgroundContext::bf_opacities(BackgroundData* bd,
                                         std::vector<BackgroundAtom>* atoms,
                                         Atmosphere* atmos)
{
    if (Nthreads <= 1)
    {
        ::bf_opacities(bd, *atoms, atmos, -1, -1);
    }
    else
    {
        struct BfData
        {
            BackgroundData* bd;
            std::vector<BackgroundAtom>* atoms;
            Atmosphere* atmos;
        };

        auto bf_handler = [](void* data, enki::TaskScheduler* s,
                                     enki::TaskSetPartition p, u32 threadId)
        {
            BfData* args = (BfData*)data;
            ::bf_opacities(args->bd, *args->atoms, args->atmos, p.start, p.end);
        };

        BfData args{bd, atoms, atmos};

        {
            LwTaskSet bfOpacities((void*)&args, &sched, bd->wavelength.shape(0),
                                  20, bf_handler);
            sched.AddTaskSetToPipe(&bfOpacities);
            sched.WaitforTask(&bfOpacities);
        }
    }
}
void FastBackgroundContext::rayleigh_scatter(BackgroundData* bd,
                                             std::vector<BackgroundAtom>* atoms,
                                             Atmosphere* atmos)
{
    if (Nthreads <= 1)
    {
        ::rayleigh_scattering(bd, *atoms, atmos, -1, -1);
    }
    else
    {
        struct RayleighData
        {
            BackgroundData* bd;
            std::vector<BackgroundAtom>* atoms;
            Atmosphere* atmos;
        };

        auto rayleigh_handler = [](void* data, enki::TaskScheduler* s,
                                     enki::TaskSetPartition p, u32 threadId)
        {
            RayleighData* args = (RayleighData*)data;
            ::rayleigh_scattering(args->bd, *args->atoms, args->atmos, p.start, p.end);
        };

        RayleighData args{bd, atoms, atmos};

        {
            LwTaskSet rayleighScatter((void*)&args, &sched, bd->wavelength.shape(0),
                                    40, rayleigh_handler);
            sched.AddTaskSetToPipe(&rayleighScatter);
            sched.WaitforTask(&rayleighScatter);
        }
    }
}