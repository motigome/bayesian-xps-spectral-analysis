#pragma once

#include "bayes_emc/prior.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bayes_emc {

enum class LikelihoodType {
    Gaussian,
    Poisson,
};

struct ParameterSpec {
    std::string name;
    PriorDistribution prior;
    double proposal_scale = 1.0;
    double proposal_decay = 0.5;
    std::vector<double> replica_step_scales;
};

struct LayerSpec {
    std::string name = "default";
    std::vector<ParameterSpec> parameters;
};

struct ModelSpec {
    std::string name;
    std::size_t basis_count = 1;
    std::vector<LayerSpec> layers;

    static ModelSpec WithParameters(
        std::string name,
        const std::size_t basis_count,
        std::vector<ParameterSpec> parameters
    ) {
        ModelSpec spec;
        spec.name = std::move(name);
        spec.basis_count = basis_count;
        spec.layers.push_back(LayerSpec{"default", std::move(parameters)});
        return spec;
    }
};

struct AnalysisSpec {
    std::size_t input_dim = 1;
    std::size_t output_dim = 1;
    LikelihoodType likelihood_type = LikelihoodType::Gaussian;
    double gaussian_sigma2 = 1.0;
    std::vector<ModelSpec> models;

    void Validate() const {
        if (input_dim == 0 || output_dim == 0) {
            throw std::invalid_argument("input_dim and output_dim must be positive.");
        }
        if (!(gaussian_sigma2 > 0.0)) {
            throw std::invalid_argument("gaussian_sigma2 must be positive.");
        }
        if (models.empty()) {
            throw std::invalid_argument("At least one model is required.");
        }
        for (const ModelSpec & model : models) {
            if (model.name.empty()) {
                throw std::invalid_argument("Model name must not be empty.");
            }
            if (model.basis_count == 0) {
                throw std::invalid_argument("basis_count must be positive.");
            }
            if (model.layers.empty()) {
                throw std::invalid_argument("Each model must have at least one layer.");
            }
            for (const LayerSpec & layer : model.layers) {
                if (layer.parameters.empty()) {
                    throw std::invalid_argument("Each layer must have at least one parameter.");
                }
                for (const ParameterSpec & parameter : layer.parameters) {
                    if (parameter.name.empty()) {
                        throw std::invalid_argument("Parameter name must not be empty.");
                    }
                    if (!(parameter.proposal_scale > 0.0)) {
                        throw std::invalid_argument("proposal_scale must be positive.");
                    }
                    if (!(parameter.proposal_decay >= 0.0)) {
                        throw std::invalid_argument("proposal_decay must be non-negative.");
                    }
                    for (const double scale : parameter.replica_step_scales) {
                        if (!(scale > 0.0)) {
                            throw std::invalid_argument("replica_step_scales values must be positive.");
                        }
                    }
                }
            }
        }
    }
};

} // namespace bayes_emc
