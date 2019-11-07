#include "Atmosphere.hpp"
#include "Constants.hpp"
#include <cmath>
#include <cstdio>

void Atmosphere::print_tau() const
{
    for (int i = 0; i < tau_ref.shape(0); ++i)
    {
        printf("%.3e, ", tau_ref(i));
    }
    printf("\n");
}

void Atmosphere::test_copy_constructor()
{
    F64Arr a = tau_ref;
    for (int i = 0; i < a.shape(0); ++i)
    {
        a[i] *= 4;
    }
    printf("-------------a---------------\n");
    for (int i = 0; i < a.shape(0); ++i)
    {
        printf("%.3e, ", a(i));
    }
    printf("\n");
    printf("-------------tau---------------\n");
    print_tau();
}

void Atmosphere::update_projections()
{
    for (int mu = 0; mu < Nrays; ++mu)
    {
        for (int k = 0; k < Nspace; ++k)
        {
            vlosMu(mu, k) = muz(mu) * vlos(k);
        }
    }

    if (!B)
        return;

    for (int mu = 0; mu < Nrays; ++mu)
    {
        if (muz(mu) == 1.0)
        {
            for (int k = 0; k < Nspace; ++k)
            {
                cosGamma(mu, k) = cos(gammaB(k));
                cos2chi(mu, k)  = cos(2.0 * chiB(k));
                sin2chi(mu, k)  = sin(2.0 * chiB(k));
            }
        }
        else
        {
            f64 cscTheta = 1.0 / sqrt(1.0 - square(muz(mu)));
            for (int k = 0; k < Nspace; ++k)
            {
                f64 sinGamma = sin(gammaB(k));
                f64 bx = sinGamma * cos(chiB(k));
                f64 by = sinGamma * sin(chiB(k));
                f64 bz = cos(gammaB(k));

                f64 b3 = mux(mu)*bx + muy(mu)*by + muz(mu)*bz;
                f64 b1 = cscTheta * (bz - muz(mu)*b3);
                f64 b2 = cscTheta * (muy(mu)*bx - mux(mu)*by);

                cosGamma(mu, k) = b3;
                cos2chi(mu, k)  = (square(b1) - square(b2)) / (1.0 - square(b3));
                sin2chi(mu, k)  = 2.0 * b1*b2 / (1.0 - square(b3));
            }
        }
    }
    
}