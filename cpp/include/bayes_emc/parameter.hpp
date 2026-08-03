#pragma once

#include "bayes_emc/model_spec.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bayes_emc {

struct ParameterIndex {
    std::size_t model = 0;
    std::size_t basis = 0;
    std::size_t layer = 0;
    std::size_t parameter = 0;
};

class ParameterLayout {
    public:
        explicit ParameterLayout(AnalysisSpec spec) : spec_(std::move(spec)) {
            spec_.Validate();
            BuildOffsets();
        }

        const AnalysisSpec & Spec() const { return spec_; }
        std::size_t Size() const { return indices_.size(); }
        const std::vector<ParameterIndex> & Indices() const { return indices_; }

        const ParameterIndex & IndexAt(const std::size_t offset) const {
            if (offset >= indices_.size()) {
                throw std::out_of_range("Parameter offset is out of range.");
            }
            return indices_[offset];
        }

        std::size_t Offset(const ParameterIndex & index) const {
            if (index.model >= offsets_.size()
                || index.basis >= offsets_[index.model].size()
                || index.layer >= offsets_[index.model][index.basis].size()
                || index.parameter >= offsets_[index.model][index.basis][index.layer].size()) {
                throw std::out_of_range("ParameterIndex is out of range.");
            }
            return offsets_[index.model][index.basis][index.layer][index.parameter];
        }

        const ParameterSpec & Parameter(const ParameterIndex & index) const {
            return spec_.models[index.model].layers[index.layer].parameters[index.parameter];
        }

        const ParameterSpec & ParameterAt(const std::size_t offset) const {
            if (offset >= flat_parameters_.size()) {
                throw std::out_of_range("Parameter offset is out of range.");
            }
            return flat_parameters_[offset];
        }

        std::string Label(const ParameterIndex & index) const {
            const ModelSpec & model = spec_.models[index.model];
            const LayerSpec & layer = model.layers[index.layer];
            const ParameterSpec & parameter = layer.parameters[index.parameter];
            return model.name
                + "[" + std::to_string(index.basis) + "]"
                + "." + layer.name
                + "." + parameter.name;
        }

    private:
        using Offset4 = std::vector<std::vector<std::vector<std::vector<std::size_t>>>>;

        void BuildOffsets() {
            offsets_.clear();
            indices_.clear();
            flat_parameters_.clear();
            offsets_.resize(spec_.models.size());
            std::size_t offset = 0;
            for (std::size_t model_id = 0; model_id < spec_.models.size(); ++model_id) {
                const ModelSpec & model = spec_.models[model_id];
                offsets_[model_id].resize(model.basis_count);
                for (std::size_t basis_id = 0; basis_id < model.basis_count; ++basis_id) {
                    offsets_[model_id][basis_id].resize(model.layers.size());
                    for (std::size_t layer_id = 0; layer_id < model.layers.size(); ++layer_id) {
                        const LayerSpec & layer = model.layers[layer_id];
                        offsets_[model_id][basis_id][layer_id].resize(layer.parameters.size());
                        for (std::size_t parameter_id = 0; parameter_id < layer.parameters.size(); ++parameter_id) {
                            offsets_[model_id][basis_id][layer_id][parameter_id] = offset++;
                            indices_.push_back(ParameterIndex{model_id, basis_id, layer_id, parameter_id});
                            flat_parameters_.push_back(layer.parameters[parameter_id]);
                        }
                    }
                }
            }
        }

        AnalysisSpec spec_;
        Offset4 offsets_;
        std::vector<ParameterIndex> indices_;
        std::vector<ParameterSpec> flat_parameters_;
};

class ParameterView {
    public:
        ParameterView(const ParameterLayout & layout, const std::vector<double> & values)
            : layout_(&layout), values_(&values) {}

        double Value(const ParameterIndex & index) const {
            return (*values_)[layout_->Offset(index)];
        }

        double Value(
            const std::size_t model,
            const std::size_t basis,
            const std::size_t parameter
        ) const {
            return Value(ParameterIndex{model, basis, 0, parameter});
        }

        double Value(
            const std::size_t model,
            const std::size_t basis,
            const std::size_t layer,
            const std::size_t parameter
        ) const {
            return Value(ParameterIndex{model, basis, layer, parameter});
        }

        const ParameterLayout & Layout() const { return *layout_; }

    private:
        const ParameterLayout * layout_;
        const std::vector<double> * values_;
};

struct ParameterState {
    std::vector<double> values;
    std::vector<double> log_prior_values;
    double log_prior_sum = 0.0;

    explicit ParameterState(const std::size_t size = 0)
        : values(size, 0.0), log_prior_values(size, NegativeInfinity()) {}

    template <class RandomEngine>
    static ParameterState FromPrior(const ParameterLayout & layout, RandomEngine & engine) {
        ParameterState state(layout.Size());
        for (const ParameterIndex & index : layout.Indices()) {
            const std::size_t offset = layout.Offset(index);
            const ParameterSpec & parameter = layout.Parameter(index);
            bool accepted = false;
            for (int attempt = 0; attempt < 1000; ++attempt) {
                const double value = parameter.prior.Sample(engine);
                const double log_prior = parameter.prior.LogPdf(value);
                if (std::isfinite(log_prior)) {
                    state.values[offset] = value;
                    state.log_prior_values[offset] = log_prior;
                    state.log_prior_sum += log_prior;
                    accepted = true;
                    break;
                }
            }
            if (!accepted) {
                throw std::runtime_error("Could not draw a valid value from prior: " + layout.Label(index));
            }
        }
        return state;
    }

    ParameterView View(const ParameterLayout & layout) const {
        return ParameterView(layout, values);
    }

    double LogPrior() const {
        return log_prior_sum;
    }

    private:
        static double NegativeInfinity() {
            return -std::numeric_limits<double>::infinity();
        }
};

} // namespace bayes_emc
