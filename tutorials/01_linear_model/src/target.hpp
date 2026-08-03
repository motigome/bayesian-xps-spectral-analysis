#pragma once

#include "bayes_emc/bayes_emc.hpp"

#include <vector>

namespace bayes_emc_user {

// Parameter order follows config.json:
//   parameter 0: a (slope)
//   parameter 1: b (intercept)
// The Gaussian likelihood in the Core adds the observation-noise model, so
// this function returns only the conditional mean a * x + b.
inline void TargetFunction(
    const std::vector<double> & x,
    const bayes_emc::ParameterView & params,
    std::vector<double> & out
) {
    const double a = params.Value(0, 0, 0);
    const double b = params.Value(0, 0, 1);
    out[0] = a * x[0] + b;
}

} // namespace bayes_emc_user
