#pragma once

#include "bayes_emc/bayes_emc.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace bayes_emc_user {

inline void TargetFunction(
    const std::vector<double> & input,
    const bayes_emc::ParameterView & params,
    std::vector<double> & output
) {
    const std::size_t peak_count = params.Layout().Spec().models[0].basis_count;
    const double energy = input[0];
    double intensity = 0.0;

    for (std::size_t peak = 0; peak < peak_count; ++peak) {
        const double amplitude = params.Value(0, peak, 0); // A_k
        const double position = params.Value(0, peak, 1);  // mu_k
        const double width = params.Value(0, peak, 2);     // w_k
        const double scaled = (energy - position) / width;
        intensity += amplitude * std::exp(-0.5 * scaled * scaled);
    }

    output[0] = intensity;
}

} // namespace bayes_emc_user
