#ifndef CMO_LW_ITERATION_RESULT_HPP
#define CMO_LW_ITERATION_RESULT_HPP
#include "Constants.hpp"
#include <vector>

struct IterationResult
{
    bool updatedJ;
    f64 dJMax;
    int dJMaxIdx;

    bool updatedPops;
    std::vector<f64> dPops;
    std::vector<int> dPopsMaxIdx;
    bool ngAccelerated;

    bool updatedNe;
    f64 dNe;
    int dNeMaxIdx;

    bool updatedRho;
    std::vector<f64> dRho;
    std::vector<int> dRhoMaxIdx;
    int NprdSubIter;
    bool updatedJPrd;
    std::vector<f64> dJPrdMax;
    std::vector<int> dJPrdMaxIdx;
};

#endif