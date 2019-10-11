#include "Atmosphere.hpp"
#include <cstdio>

void Atmosphere::print_tau() const
{
    for (int i = 0; i < tau_ref.shape(0); ++i)
    {
        printf("%.3e, ", tau_ref(i));
    }
    printf("\n");
}