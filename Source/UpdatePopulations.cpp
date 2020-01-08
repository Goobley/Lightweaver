#include "Lightweaver.hpp"

void stat_eq(Atom* atomIn)
{
    auto& atom = *atomIn;
    const int Nlevel = atom.Nlevel;
    const int Nspace = atom.n.shape(1);

    auto nk = F64Arr(Nlevel);
    auto Gamma = F64Arr2D(Nlevel, Nlevel);

    for (int k = 0; k < Nspace; ++k)
    {
        for (int i = 0; i < Nlevel; ++i)
        {
            nk(i) = atom.n(i, k);
            for (int j = 0; j < Nlevel; ++j)
                Gamma(i, j) = atom.Gamma(i, j, k);
        }

        int iEliminate = 0;
        f64 nMax = 0.0;
        for (int i = 0; i < Nlevel; ++i)
            nMax = max_idx(nMax, nk(i), iEliminate, i);

        for (int i = 0; i < Nlevel; ++i)
        {
            Gamma(iEliminate, i) = 1.0;
            nk(i) = 0.0;
        }
        nk(iEliminate) = atom.nTotal(k);

        solve_lin_eq(Gamma, nk);
        for (int i = 0; i < Nlevel; ++i)
            atom.n(i, k) = nk(i);
    }
}

void time_dependent_update(Atom* atomIn, F64View2D nOld, f64 dt)
{
    auto& atom  = *atomIn;
    const int Nlevel = atom.Nlevel;
    const int Nspace = atom.n.shape(1);

    auto nk = F64Arr(Nlevel);
    auto Gamma = F64Arr2D(Nlevel, Nlevel);

    // throw std::runtime_error("Singular Matrix");

    for (int k = 0; k < Nspace; ++k)
    {
        for (int i = 0; i < Nlevel; ++i)
        {
            nk(i) = nOld(i, k);
            for (int j = 0; j < Nlevel; ++j)
                Gamma(i, j) = -atom.Gamma(i, j, k) * dt;
            Gamma(i, i) = 1.0 - atom.Gamma(i, i, k) * dt;
        }

        solve_lin_eq(Gamma, nk);

        for (int i = 0; i < Nlevel; ++i)
        {
            atom.n(i, k) = nk(i);
        }
    }
}