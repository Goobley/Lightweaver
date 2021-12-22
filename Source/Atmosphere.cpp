#include "Lightweaver.hpp"
#include "Constants.hpp"
#include <cmath>

void Atmosphere::update_projections()
{
    switch (Ndim)
    {
        case 1:
        {
            for (int mu = 0; mu < Nrays; ++mu)
            {
                for (int k = 0; k < Nspace; ++k)
                {
                    vlosMu(mu, k) = muz(mu) * vz(k);
                }
            }
        } break;

        case 2:
        {
            for (int mu = 0; mu < Nrays; ++mu)
            {
                for (int k = 0; k < Nspace; ++k)
                {
                    vlosMu(mu, k) = mux(mu) * vx(k) + muz(mu) * vz(k);
                }
            }
        } break;

        case 3:
        {
            for (int mu = 0; mu < Nrays; ++mu)
            {
                for (int k = 0; k < Nspace; ++k)
                {
                    vlosMu(mu, k) = mux(mu) * vx(k) + muy(mu) * vy(k) + muz(mu) * vz(k);
                }
            }
        } break;

        default:
        {
        } break;
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
                // NOTE(cmo): Basic projection using spherical polar
                // coordinates.
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