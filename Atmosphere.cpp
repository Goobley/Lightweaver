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