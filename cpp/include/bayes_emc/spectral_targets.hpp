#pragma once

#include "bayes_emc/data.hpp"
#include "bayes_emc/parameter.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace bayes_emc {

class GaussianPeakSumTarget {
    public:
        struct IncrementalGaussianCache {
            std::vector<double> residuals;
            std::vector<double> spectral_basis_values;
            std::size_t observation_count = 0;
            std::size_t spectral_basis_count = 0;
            double squared_error = 0.0;
        };

        static GaussianPeakSumTarget SpectralOnly(const std::size_t spectral_model_id = 0) {
            return GaussianPeakSumTarget(false, 0, spectral_model_id);
        }

        static GaussianPeakSumTarget WithLinearBackground(
            const std::size_t linear_model_id = 0,
            const std::size_t spectral_model_id = 1
        ) {
            return GaussianPeakSumTarget(true, linear_model_id, spectral_model_id);
        }

        void operator()(
            const std::vector<double> & x,
            const ParameterView & params,
            std::vector<double> & out
        ) const {
            if (out.empty()) {
                out.resize(1);
            }
            out[0] = Predict(x[0], params);
        }

        void InitializeIncrementalGaussianCache(
            const DataSet & data,
            const ParameterLayout & layout,
            const ParameterState & state,
            IncrementalGaussianCache & cache
        ) const {
            ValidateShape(data, layout);
            const std::size_t observation_count = data.Size();
            const std::size_t basis_count = layout.Spec().models[spectral_model_id_].basis_count;
            cache.observation_count = observation_count;
            cache.spectral_basis_count = basis_count;
            cache.residuals.resize(observation_count);
            cache.spectral_basis_values.assign(observation_count * basis_count, 0.0);
            cache.squared_error = 0.0;

            const ParameterView view = state.View(layout);
            const auto & observations = data.Observations();
            for (std::size_t observation_id = 0; observation_id < observations.size(); ++observation_id) {
                const Observation & observation = observations[observation_id];
                double residual = observation.y[0];
                if (has_linear_background_) {
                    residual -= view.Value(linear_model_id_, 0, 0)
                        + view.Value(linear_model_id_, 0, 1) * observation.x[0];
                }
                for (std::size_t basis = 0; basis < basis_count; ++basis) {
                    const double contribution = Peak(
                        observation.x[0],
                        view.Value(spectral_model_id_, basis, 0),
                        view.Value(spectral_model_id_, basis, 1),
                        view.Value(spectral_model_id_, basis, 2)
                    );
                    cache.spectral_basis_values[basis * observation_count + observation_id] = contribution;
                    residual -= contribution;
                }
                cache.residuals[observation_id] = residual;
                cache.squared_error += residual * residual;
            }
        }

        double IncrementalGaussianEnergy(
            const DataSet & data,
            const ParameterLayout &,
            const ParameterState &,
            const IncrementalGaussianCache & cache,
            const double sigma2
        ) const {
            if (!(sigma2 > 0.0)) {
                throw std::invalid_argument("sigma2 must be positive.");
            }
            return cache.squared_error / (2.0 * static_cast<double>(data.Size()) * sigma2);
        }

        double ProposeIncrementalGaussianEnergy(
            const DataSet & data,
            const ParameterLayout & layout,
            const ParameterState & state,
            IncrementalGaussianCache & cache,
            const std::size_t offset,
            const double proposed_value,
            const double sigma2
        ) const {
            if (!(sigma2 > 0.0)) {
                throw std::invalid_argument("sigma2 must be positive.");
            }
            const double squared_error = ProposedSquaredError(
                data,
                layout,
                state,
                cache,
                layout.IndexAt(offset),
                proposed_value,
                false
            );
            if (!std::isfinite(squared_error)) {
                return std::numeric_limits<double>::infinity();
            }
            return squared_error / (2.0 * static_cast<double>(data.Size()) * sigma2);
        }

        void CommitIncrementalGaussianProposal(
            const DataSet & data,
            const ParameterLayout & layout,
            const ParameterState & state,
            IncrementalGaussianCache & cache,
            const std::size_t offset,
            const double proposed_value
        ) const {
            cache.squared_error = ProposedSquaredError(
                data,
                layout,
                state,
                cache,
                layout.IndexAt(offset),
                proposed_value,
                true
            );
        }

    private:
        GaussianPeakSumTarget(
            const bool has_linear_background,
            const std::size_t linear_model_id,
            const std::size_t spectral_model_id
        ) : has_linear_background_(has_linear_background),
            linear_model_id_(linear_model_id),
            spectral_model_id_(spectral_model_id) {}

        void ValidateShape(const DataSet & data, const ParameterLayout & layout) const {
            if (data.InputDim() != 1 || data.OutputDim() != 1) {
                throw std::invalid_argument("GaussianPeakSumTarget requires one input and one output column.");
            }
            const auto & models = layout.Spec().models;
            if (spectral_model_id_ >= models.size()) {
                throw std::invalid_argument("spectral_model_id is out of range.");
            }
            if (has_linear_background_ && linear_model_id_ >= models.size()) {
                throw std::invalid_argument("linear_model_id is out of range.");
            }
            if (models[spectral_model_id_].layers.empty()
                || models[spectral_model_id_].layers[0].parameters.size() < 3) {
                throw std::invalid_argument("spectral model requires parameters a, mu, b.");
            }
            if (has_linear_background_
                && (models[linear_model_id_].layers.empty()
                    || models[linear_model_id_].layers[0].parameters.size() < 2)) {
                throw std::invalid_argument("linear background model requires intercept and slope.");
            }
        }

        double Predict(const double x, const ParameterView & params) const {
            double y = 0.0;
            if (has_linear_background_) {
                y += params.Value(linear_model_id_, 0, 0)
                    + params.Value(linear_model_id_, 0, 1) * x;
            }

            const std::size_t basis_count = params.Layout().Spec().models[spectral_model_id_].basis_count;
            for (std::size_t basis = 0; basis < basis_count; ++basis) {
                y += Peak(
                    x,
                    params.Value(spectral_model_id_, basis, 0),
                    params.Value(spectral_model_id_, basis, 1),
                    params.Value(spectral_model_id_, basis, 2)
                );
            }
            return y;
        }

        double ProposedSquaredError(
            const DataSet & data,
            const ParameterLayout & layout,
            const ParameterState & state,
            IncrementalGaussianCache & cache,
            const ParameterIndex & index,
            const double proposed_value,
            const bool commit
        ) const {
            if (index.model == spectral_model_id_) {
                return ProposedSpectralSquaredError(data, layout, state, cache, index, proposed_value, commit);
            }
            if (has_linear_background_ && index.model == linear_model_id_) {
                return ProposedLinearSquaredError(data, layout, state, cache, index, proposed_value, commit);
            }
            throw std::invalid_argument("GaussianPeakSumTarget received a parameter outside its models.");
        }

        double ProposedLinearSquaredError(
            const DataSet & data,
            const ParameterLayout & layout,
            const ParameterState & state,
            IncrementalGaussianCache & cache,
            const ParameterIndex & index,
            const double proposed_value,
            const bool commit
        ) const {
            if (index.basis != 0 || index.parameter > 1) {
                throw std::invalid_argument("linear background supports only intercept and slope.");
            }
            const std::size_t current_offset = layout.Offset(index);
            const double current_value = state.values[current_offset];
            const double delta_value = proposed_value - current_value;
            return ApplyDelta(data, cache, commit, [&](const Observation & observation) {
                return index.parameter == 0 ? delta_value : delta_value * observation.x[0];
            });
        }

        double ProposedSpectralSquaredError(
            const DataSet & data,
            const ParameterLayout & layout,
            const ParameterState & state,
            IncrementalGaussianCache & cache,
            const ParameterIndex & index,
            const double proposed_value,
            const bool commit
        ) const {
            if (index.parameter > 2) {
                throw std::invalid_argument("spectral model supports only a, mu, b parameters.");
            }
            if (cache.observation_count != data.Size()
                || index.basis >= cache.spectral_basis_count
                || cache.spectral_basis_values.size() != cache.spectral_basis_count * cache.observation_count) {
                throw std::runtime_error("Incremental spectral cache does not match data size or basis count.");
            }
            double current_a = state.values[layout.Offset(ParameterIndex{index.model, index.basis, index.layer, 0})];
            double current_mu = state.values[layout.Offset(ParameterIndex{index.model, index.basis, index.layer, 1})];
            double current_b = state.values[layout.Offset(ParameterIndex{index.model, index.basis, index.layer, 2})];
            double proposed_a = current_a;
            double proposed_mu = current_mu;
            double proposed_b = current_b;
            if (index.parameter == 0) proposed_a = proposed_value;
            if (index.parameter == 1) proposed_mu = proposed_value;
            if (index.parameter == 2) proposed_b = proposed_value;

            double squared_error = 0.0;
            const std::size_t basis_offset = index.basis * cache.observation_count;
            const bool can_scale_amplitude = index.parameter == 0 && current_a != 0.0;
            const double amplitude_ratio = can_scale_amplitude ? proposed_a / current_a : 0.0;
            const auto & observations = data.Observations();
            for (std::size_t observation_id = 0; observation_id < observations.size(); ++observation_id) {
                const double current_peak = cache.spectral_basis_values[basis_offset + observation_id];
                const double proposed_peak = can_scale_amplitude
                    ? current_peak * amplitude_ratio
                    : Peak(observations[observation_id].x[0], proposed_a, proposed_mu, proposed_b);
                const double updated_residual = cache.residuals[observation_id] - (proposed_peak - current_peak);
                squared_error += updated_residual * updated_residual;
                if (commit) {
                    cache.residuals[observation_id] = updated_residual;
                    cache.spectral_basis_values[basis_offset + observation_id] = proposed_peak;
                }
            }
            return squared_error;
        }

        template <class DeltaFunction>
        double ApplyDelta(
            const DataSet & data,
            IncrementalGaussianCache & cache,
            const bool commit,
            const DeltaFunction & delta
        ) const {
            if (cache.residuals.size() != data.Size()) {
                throw std::runtime_error("Incremental Gaussian cache does not match data size.");
            }

            double squared_error = 0.0;
            const auto & observations = data.Observations();
            for (std::size_t observation_id = 0; observation_id < observations.size(); ++observation_id) {
                const double updated_residual = cache.residuals[observation_id] - delta(observations[observation_id]);
                squared_error += updated_residual * updated_residual;
                if (commit) {
                    cache.residuals[observation_id] = updated_residual;
                }
            }
            return squared_error;
        }

        static double Peak(const double x, const double a, const double mu, const double b) {
            const double diff = x - mu;
            return a * std::exp(-0.5 * b * diff * diff);
        }

        bool has_linear_background_ = false;
        std::size_t linear_model_id_ = 0;
        std::size_t spectral_model_id_ = 0;
};

} // namespace bayes_emc
