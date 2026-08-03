#pragma once

#include "bayes_emc/data.hpp"
#include "bayes_emc/likelihood.hpp"
#include "bayes_emc/parameter.hpp"
#include "bayes_emc/rng.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace bayes_emc {

namespace detail {

struct NoIncrementalGaussianCache {};

template <class TargetFunction, class = void>
struct IncrementalGaussianCacheFor {
    using type = NoIncrementalGaussianCache;
    static constexpr bool available = false;
};

template <class TargetFunction>
struct IncrementalGaussianCacheFor<TargetFunction, std::void_t<typename TargetFunction::IncrementalGaussianCache>> {
    using type = typename TargetFunction::IncrementalGaussianCache;
    static constexpr bool available = true;
};

template <class TargetFunction>
inline constexpr bool HasIncrementalGaussianTarget =
    IncrementalGaussianCacheFor<TargetFunction>::available;

} // namespace detail

struct EngineOptions {
    std::size_t replica_count = 8;
    double gamma = 1.4;
    std::size_t burnin_count = 1000;
    std::size_t sample_count = 1000;
    std::size_t sample_stride = 1;
    std::size_t exchange_stride = 1;
    std::size_t parallel_worker_count = 1;
    std::size_t likelihood_worker_count = 1;
    std::size_t likelihood_parallel_min_rows = 2048;
    bool progress_enabled = false;
    std::size_t progress_interval_steps = 0;
    std::size_t progress_bar_width = 32;
    unsigned int seed = 5489u;
};

struct SampleRecord {
    std::size_t sample_id = 0;
    std::size_t replica_id = 0;
    double inverse_temperature = 1.0;
    double energy = 0.0;
    double log_posterior = 0.0;
    std::vector<double> values;
};

struct EngineResult {
    ParameterLayout layout;
    std::vector<SampleRecord> samples;
    std::vector<std::vector<SampleRecord>> replica_samples;
    std::vector<double> inverse_temperatures;
    std::vector<std::vector<double>> replica_energy_samples;
    std::size_t parallel_worker_count = 1;
    std::size_t likelihood_worker_count = 1;
    std::size_t data_count = 0;
    std::vector<std::size_t> mh_attempt_counts;
    std::vector<std::size_t> mh_accept_counts;
    std::vector<std::vector<std::size_t>> mh_parameter_attempt_counts;
    std::vector<std::vector<std::size_t>> mh_parameter_accept_counts;
    std::vector<std::size_t> exchange_attempt_counts;
    std::vector<std::size_t> exchange_accept_counts;
};

template <class TargetFunction>
class ExchangeMonteCarlo {
    using IncrementalGaussianCache = typename detail::IncrementalGaussianCacheFor<TargetFunction>::type;

    public:
        ExchangeMonteCarlo(
            AnalysisSpec spec,
            DataSet data,
            TargetFunction target,
            EngineOptions options
        ) : layout_(std::move(spec)),
            data_(std::move(data)),
            target_(std::move(target)),
            options_(options),
            engine_(MakeRandomEngine(static_cast<std::uint64_t>(options.seed))) {
            if (options_.replica_count == 0) {
                throw std::invalid_argument("replica_count must be positive.");
            }
            if (!(options_.gamma >= 1.0)) {
                throw std::invalid_argument("gamma must be >= 1.0.");
            }
            if (options_.sample_count == 0) {
                throw std::invalid_argument("sample_count must be positive.");
            }
            if (options_.sample_stride == 0) {
                throw std::invalid_argument("sample_stride must be positive.");
            }
            if (options_.exchange_stride == 0) {
                throw std::invalid_argument("exchange_stride must be positive.");
            }
            if (data_.InputDim() != layout_.Spec().input_dim || data_.OutputDim() != layout_.Spec().output_dim) {
                throw std::invalid_argument("Data dimensions do not match AnalysisSpec.");
            }
            inverse_temperatures_ = BuildInverseTemperatures(options_.replica_count, options_.gamma);
            weighted_inverse_temperatures_ = BuildWeightedInverseTemperatures();
            exchange_log_factors_ = BuildExchangeLogFactors();
            step_sizes_ = BuildStepSizes();
            prediction_buffers_.assign(
                options_.replica_count,
                std::vector<double>(data_.OutputDim(), 0.0)
            );
        }

        EngineResult Run() {
            effective_replica_worker_count_ = EffectiveReplicaWorkerCountFor(options_.replica_count);
            effective_likelihood_worker_count_ = EffectiveLikelihoodWorkerCountFor(data_.Size());
            InitializeLikelihoodPools();
            InitializeReplicas();
            InitializeDiagnostics();
            std::vector<SampleRecord> samples;
            samples.reserve(options_.sample_count);
            std::vector<std::vector<SampleRecord>> replica_samples(options_.replica_count);
            for (std::vector<SampleRecord> & records : replica_samples) {
                records.reserve(options_.sample_count);
            }
            std::vector<std::vector<double>> replica_energy_samples(options_.replica_count);
            for (std::vector<double> & energies : replica_energy_samples) {
                energies.reserve(options_.sample_count);
            }
            const std::size_t worker_count = effective_replica_worker_count_;

            const std::size_t total_steps = options_.burnin_count
                + options_.sample_count * options_.sample_stride;
            BeginProgress(total_steps);
            try {
                if (worker_count <= 1) {
                    for (std::size_t step = 0; step < total_steps; ++step) {
                        UpdateAllReplicasSerial(step >= options_.burnin_count);
                        FinishStep(step, samples, replica_samples, replica_energy_samples);
                        MaybeReportProgress(step + 1, total_steps, samples.size());
                    }
                } else {
                    RunParallelLoop(total_steps, worker_count, samples, replica_samples, replica_energy_samples);
                }
            } catch (...) {
                AbortProgress();
                throw;
            }
            return EngineResult{
                layout_,
                std::move(samples),
                std::move(replica_samples),
                inverse_temperatures_,
                std::move(replica_energy_samples),
                worker_count,
                effective_likelihood_worker_count_,
                data_.Size(),
                mh_attempt_counts_,
                mh_accept_counts_,
                mh_parameter_attempt_counts_,
                mh_parameter_accept_counts_,
                exchange_attempt_counts_,
                exchange_accept_counts_,
            };
        }

    private:
        static std::vector<double> BuildInverseTemperatures(
            const std::size_t replica_count,
            const double gamma
        ) {
            std::vector<double> temperatures(replica_count, 1.0);
            if (replica_count == 1) return temperatures;
            temperatures[0] = 0.0;
            for (std::size_t replica_id = 1; replica_id < replica_count; ++replica_id) {
                const int exponent = static_cast<int>(replica_id + 1) - static_cast<int>(replica_count);
                temperatures[replica_id] = std::pow(gamma, exponent);
            }
            return temperatures;
        }

        std::vector<double> BuildWeightedInverseTemperatures() const {
            std::vector<double> temperatures(options_.replica_count, 0.0);
            const double data_count = static_cast<double>(data_.Size());
            for (std::size_t replica_id = 0; replica_id < options_.replica_count; ++replica_id) {
                temperatures[replica_id] = data_count * inverse_temperatures_[replica_id];
            }
            return temperatures;
        }

        std::vector<long double> BuildExchangeLogFactors() const {
            const std::size_t pair_count = options_.replica_count > 0 ? options_.replica_count - 1 : 0;
            std::vector<long double> factors(pair_count, 0.0L);
            const long double data_count = static_cast<long double>(data_.Size());
            for (std::size_t replica_id = 0; replica_id < pair_count; ++replica_id) {
                factors[replica_id] = data_count
                    * (static_cast<long double>(inverse_temperatures_[replica_id])
                        - static_cast<long double>(inverse_temperatures_[replica_id + 1]));
            }
            return factors;
        }

        bool IsPriorReplica(const std::size_t replica_id) const {
            return inverse_temperatures_[replica_id] == 0.0;
        }

        std::vector<std::vector<double>> BuildStepSizes() const {
            std::vector<std::vector<double>> step_sizes(
                options_.replica_count,
                std::vector<double>(layout_.Size(), 0.0)
            );
            for (std::size_t replica_id = 0; replica_id < options_.replica_count; ++replica_id) {
                const double n_beta = weighted_inverse_temperatures_[replica_id];
                for (std::size_t offset = 0; offset < layout_.Size(); ++offset) {
                    const ParameterSpec & parameter = layout_.ParameterAt(offset);
                    if (!parameter.replica_step_scales.empty()
                        && parameter.replica_step_scales.size() != options_.replica_count) {
                        throw std::invalid_argument("replica_step_scales size must match replica_count.");
                    }
                    const double base_step = n_beta < 1.0
                        ? parameter.proposal_scale
                        : parameter.proposal_scale / std::pow(n_beta, parameter.proposal_decay);
                    const double replica_scale = parameter.replica_step_scales.empty()
                        ? 1.0
                        : parameter.replica_step_scales[replica_id];
                    step_sizes[replica_id][offset] = base_step * replica_scale;
                }
            }
            return step_sizes;
        }

        std::size_t EffectiveReplicaWorkerCountFor(const std::size_t task_count) const {
            if (task_count < 2) return 1;
            std::size_t requested = options_.parallel_worker_count;
            if (requested == 0) {
                requested = std::thread::hardware_concurrency();
                if (requested == 0) requested = 1;
            }
            requested = std::max<std::size_t>(1, requested);
            return std::min<std::size_t>(requested, task_count);
        }

        std::size_t EffectiveLikelihoodWorkerCountFor(const std::size_t observation_count) const {
            if (observation_count < options_.likelihood_parallel_min_rows) return 1;
            std::size_t requested = options_.likelihood_worker_count;
            if (requested == 0) {
                requested = std::thread::hardware_concurrency();
                if (requested == 0) requested = 1;
            }
            requested = std::max<std::size_t>(1, requested);
            return std::min<std::size_t>(requested, observation_count);
        }

        void InitializeLikelihoodPools() {
            likelihood_pools_.clear();
            if (effective_likelihood_worker_count_ <= 1) return;
            const std::size_t pool_count = effective_replica_worker_count_ > 1 ? options_.replica_count : 1;
            likelihood_pools_.reserve(pool_count);
            for (std::size_t pool_id = 0; pool_id < pool_count; ++pool_id) {
                likelihood_pools_.push_back(std::make_unique<LikelihoodThreadPool>(effective_likelihood_worker_count_));
            }
        }

        LikelihoodThreadPool * LikelihoodPoolFor(const std::size_t replica_id) const {
            if (likelihood_pools_.empty()) return nullptr;
            if (likelihood_pools_.size() == 1) return likelihood_pools_[0].get();
            return likelihood_pools_[replica_id].get();
        }

        void InitializeReplicaEngines() {
            replica_engines_.clear();
            replica_engines_.reserve(options_.replica_count);
            for (std::size_t replica_id = 0; replica_id < options_.replica_count; ++replica_id) {
                RandomEngine replica_engine = MakeRandomEngine(
                    static_cast<std::uint64_t>(options_.seed),
                    static_cast<std::uint64_t>(replica_id + 1)
                        * 0x9e3779b97f4a7c15ULL
                        + static_cast<std::uint64_t>(options_.replica_count)
                );
                replica_engines_.push_back(std::move(replica_engine));
            }
        }

        void InitializeReplicas() {
            states_.clear();
            energies_.clear();
            states_.reserve(options_.replica_count);
            energies_.reserve(options_.replica_count);
            const std::size_t worker_count = effective_replica_worker_count_;
            if (worker_count > 1) {
                InitializeReplicaEngines();
            }
            InitializeIncrementalCaches();
            for (std::size_t replica_id = 0; replica_id < options_.replica_count; ++replica_id) {
                RandomEngine & engine = worker_count > 1 ? replica_engines_[replica_id] : engine_;
                auto [state, energy] = DrawPriorStateWithFiniteEnergy(
                    replica_id,
                    engine,
                    IncrementalCacheForReplica(replica_id)
                );
                states_.push_back(std::move(state));
                energies_.push_back(energy);
            }
        }

        std::pair<ParameterState, double> DrawPriorStateWithFiniteEnergy(
            const std::size_t replica_id,
            RandomEngine & engine,
            IncrementalGaussianCache * cache
        ) const {
            for (int attempt = 0; attempt < 1000; ++attempt) {
                ParameterState state = ParameterState::FromPrior(layout_, engine);
                const double energy = PriorStateEnergy(state, replica_id, cache);
                if (std::isfinite(energy)) {
                    return {std::move(state), energy};
                }
            }
            throw std::runtime_error("Could not draw a finite-energy state from the prior.");
        }

        void InitializeDiagnostics() {
            mh_attempt_counts_.assign(options_.replica_count, 0);
            mh_accept_counts_.assign(options_.replica_count, 0);
            mh_parameter_attempt_counts_.assign(
                options_.replica_count,
                std::vector<std::size_t>(layout_.Size(), 0)
            );
            mh_parameter_accept_counts_.assign(
                options_.replica_count,
                std::vector<std::size_t>(layout_.Size(), 0)
            );
            const std::size_t exchange_pair_count = options_.replica_count > 0 ? options_.replica_count - 1 : 0;
            exchange_attempt_counts_.assign(exchange_pair_count, 0);
            exchange_accept_counts_.assign(exchange_pair_count, 0);
        }

        void InitializeIncrementalCaches() {
            if constexpr (detail::HasIncrementalGaussianTarget<TargetFunction>) {
                incremental_caches_.assign(options_.replica_count, IncrementalGaussianCache{});
            }
        }

        double PriorStateEnergy(
            const ParameterState & state,
            const std::size_t replica_id,
            IncrementalGaussianCache * cache
        ) const {
            if (layout_.Spec().likelihood_type == LikelihoodType::Gaussian) {
                if constexpr (detail::HasIncrementalGaussianTarget<TargetFunction>) {
                    if (cache == nullptr) {
                        throw std::runtime_error("Incremental cache is not initialized.");
                    }
                    target_.InitializeIncrementalGaussianCache(
                        data_,
                        layout_,
                        state,
                        *cache
                    );
                    return target_.IncrementalGaussianEnergy(
                        data_,
                        layout_,
                        state,
                        *cache,
                        layout_.Spec().gaussian_sigma2
                    );
                }
            }
            return FullEnergy(state, replica_id);
        }

        IncrementalGaussianCache * IncrementalCacheForReplica(const std::size_t replica_id) {
            if constexpr (detail::HasIncrementalGaussianTarget<TargetFunction>) {
                return &incremental_caches_[replica_id];
            }
            return nullptr;
        }

        void SwapIncrementalCaches(const std::size_t left, const std::size_t right) {
            if constexpr (detail::HasIncrementalGaussianTarget<TargetFunction>) {
                std::swap(incremental_caches_[left], incremental_caches_[right]);
            }
        }

        double FullEnergy(const ParameterState & state, const std::size_t replica_id) const {
            if (effective_likelihood_worker_count_ <= 1
                && replica_id < prediction_buffers_.size()) {
                if (layout_.Spec().likelihood_type == LikelihoodType::Poisson) {
                    return PoissonEnergy(
                        data_,
                        layout_,
                        state,
                        target_,
                        prediction_buffers_[replica_id]
                    );
                }
                return GaussianEnergy(
                    data_,
                    layout_,
                    state,
                    target_,
                    layout_.Spec().gaussian_sigma2,
                    prediction_buffers_[replica_id]
                );
            }
            if (layout_.Spec().likelihood_type == LikelihoodType::Poisson) {
                return PoissonEnergy(
                    data_,
                    layout_,
                    state,
                    target_,
                    LikelihoodPoolFor(replica_id),
                    options_.likelihood_parallel_min_rows
                );
            }
            return GaussianEnergy(
                data_,
                layout_,
                state,
                target_,
                layout_.Spec().gaussian_sigma2,
                LikelihoodPoolFor(replica_id),
                options_.likelihood_parallel_min_rows
            );
        }

        void UpdateAllReplicasSerial(const bool collect_diagnostics) {
            for (std::size_t replica_id = 0; replica_id < states_.size(); ++replica_id) {
                UpdateReplica(replica_id, engine_, collect_diagnostics);
            }
        }

        void RunParallelLoop(
            const std::size_t total_steps,
            const std::size_t worker_count,
            std::vector<SampleRecord> & samples,
            std::vector<std::vector<SampleRecord>> & replica_samples,
            std::vector<std::vector<double>> & replica_energy_samples
        ) {
            if (replica_engines_.size() != states_.size()) {
                InitializeReplicaEngines();
            }

            std::mutex mutex;
            std::condition_variable start_cv;
            std::condition_variable done_cv;
            std::size_t generation = 0;
            std::size_t finished_count = 0;
            bool stop = false;
            bool collect_diagnostics = false;
            std::exception_ptr worker_exception = nullptr;

            std::vector<std::thread> workers;
            workers.reserve(worker_count);
            for (std::size_t worker_id = 0; worker_id < worker_count; ++worker_id) {
                const std::size_t begin = worker_id * states_.size() / worker_count;
                const std::size_t end = (worker_id + 1) * states_.size() / worker_count;
                workers.emplace_back([&, begin, end]() {
                    std::size_t observed_generation = 0;
                    while (true) {
                        bool worker_collect_diagnostics = false;
                        {
                            std::unique_lock<std::mutex> lock(mutex);
                            start_cv.wait(lock, [&]() {
                                return stop || generation != observed_generation;
                            });
                            if (stop) return;
                            observed_generation = generation;
                            worker_collect_diagnostics = collect_diagnostics;
                        }
                        try {
                            for (std::size_t replica_id = begin; replica_id < end; ++replica_id) {
                                UpdateReplica(replica_id, replica_engines_[replica_id], worker_collect_diagnostics);
                            }
                        } catch (...) {
                            std::lock_guard<std::mutex> lock(mutex);
                            if (worker_exception == nullptr) {
                                worker_exception = std::current_exception();
                            }
                            stop = true;
                            ++finished_count;
                            done_cv.notify_one();
                            start_cv.notify_all();
                            return;
                        }
                        {
                            std::lock_guard<std::mutex> lock(mutex);
                            ++finished_count;
                        }
                        done_cv.notify_one();
                    }
                });
            }

            auto stop_workers = [&]() {
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    stop = true;
                }
                start_cv.notify_all();
                for (std::thread & worker : workers) {
                    if (worker.joinable()) {
                        worker.join();
                    }
                }
            };

            try {
                for (std::size_t step = 0; step < total_steps; ++step) {
                    {
	                        std::lock_guard<std::mutex> lock(mutex);
	                        finished_count = 0;
	                        collect_diagnostics = step >= options_.burnin_count;
	                        ++generation;
	                    }
                    start_cv.notify_all();

                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        done_cv.wait(lock, [&]() {
                            return finished_count == worker_count || worker_exception != nullptr;
                        });
                        if (worker_exception != nullptr) {
                            stop = true;
                        }
                    }
                    if (worker_exception != nullptr) {
                        stop_workers();
                        std::rethrow_exception(worker_exception);
                    }

                    FinishStep(step, samples, replica_samples, replica_energy_samples);
                    MaybeReportProgress(step + 1, total_steps, samples.size());
                }
            } catch (...) {
                stop_workers();
                throw;
            }
            stop_workers();
        }

        void FinishStep(
            const std::size_t step,
            std::vector<SampleRecord> & samples,
            std::vector<std::vector<SampleRecord>> & replica_samples,
            std::vector<std::vector<double>> & replica_energy_samples
        ) {
            if ((step + 1) % options_.exchange_stride == 0) {
                ExchangeReplicas(step >= options_.burnin_count);
            }
            if (step >= options_.burnin_count
                && (step - options_.burnin_count) % options_.sample_stride == 0) {
                const std::size_t sample_id = samples.size();
                for (std::size_t replica_id = 0; replica_id < energies_.size(); ++replica_id) {
                    replica_samples[replica_id].push_back(MakeSample(sample_id, replica_id));
                    replica_energy_samples[replica_id].push_back(energies_[replica_id]);
                }
                const std::size_t cold_replica = states_.size() - 1;
                samples.push_back(replica_samples[cold_replica].back());
            }
        }

        void UpdateReplica(
            const std::size_t replica_id,
            RandomEngine & engine,
            const bool collect_diagnostics
        ) {
            if (IsPriorReplica(replica_id)) {
                UpdatePriorReplica(replica_id, engine, collect_diagnostics);
                return;
            }
            const std::size_t parameter_count = layout_.Size();
            for (std::size_t offset = 0; offset < parameter_count; ++offset) {
                UpdateParameter(replica_id, offset, engine, collect_diagnostics);
            }
        }

        void UpdatePriorReplica(
            const std::size_t replica_id,
            RandomEngine & engine,
            const bool collect_diagnostics
        ) {
            auto [state, energy] = DrawPriorStateWithFiniteEnergy(
                replica_id,
                engine,
                IncrementalCacheForReplica(replica_id)
            );
            states_[replica_id] = std::move(state);
            energies_[replica_id] = energy;
            if (!collect_diagnostics) return;

            const std::size_t parameter_count = layout_.Size();
            mh_attempt_counts_[replica_id] += parameter_count;
            mh_accept_counts_[replica_id] += parameter_count;
            for (std::size_t offset = 0; offset < parameter_count; ++offset) {
                ++mh_parameter_attempt_counts_[replica_id][offset];
                ++mh_parameter_accept_counts_[replica_id][offset];
            }
        }

        void UpdateParameter(
            const std::size_t replica_id,
            const std::size_t offset,
            RandomEngine & engine,
            const bool collect_diagnostics
        ) {
            const ParameterSpec & parameter = layout_.ParameterAt(offset);
            const double step_size = step_sizes_[replica_id][offset];

            if (collect_diagnostics) {
                ++mh_attempt_counts_[replica_id];
                ++mh_parameter_attempt_counts_[replica_id][offset];
            }
            ParameterState & state = states_[replica_id];
            const double current_value = state.values[offset];
            const double current_log_prior = state.log_prior_values[offset];
            const double proposed_value = current_value + (2.0 * Uniform01(engine) - 1.0) * step_size;
            const double proposed_log_prior = parameter.prior.LogPdf(proposed_value);
            if (!std::isfinite(proposed_log_prior)) return;

            const double log_prior_delta = proposed_log_prior - current_log_prior;
            double proposed_energy = std::numeric_limits<double>::infinity();

            auto apply_proposal = [&]() {
                state.values[offset] = proposed_value;
                state.log_prior_values[offset] = proposed_log_prior;
                state.log_prior_sum += log_prior_delta;
            };
            auto restore_current = [&]() {
                state.values[offset] = current_value;
                state.log_prior_values[offset] = current_log_prior;
                state.log_prior_sum -= log_prior_delta;
            };

            bool state_was_mutated = false;
            try {
                if (layout_.Spec().likelihood_type == LikelihoodType::Gaussian) {
                    if constexpr (detail::HasIncrementalGaussianTarget<TargetFunction>) {
                        proposed_energy = target_.ProposeIncrementalGaussianEnergy(
                            data_,
                            layout_,
                            state,
                            incremental_caches_[replica_id],
                            offset,
                            proposed_value,
                            layout_.Spec().gaussian_sigma2
                        );
                    } else {
                        apply_proposal();
                        state_was_mutated = true;
                        proposed_energy = FullEnergy(state, replica_id);
                    }
                } else {
                    apply_proposal();
                    state_was_mutated = true;
                    proposed_energy = FullEnergy(state, replica_id);
                }
            } catch (...) {
                if (state_was_mutated) {
                    restore_current();
                }
                throw;
            }
            if (!std::isfinite(proposed_energy)) {
                if (state_was_mutated) {
                    restore_current();
                }
                return;
            }

            const long double log_acceptance =
                -static_cast<long double>(weighted_inverse_temperatures_[replica_id])
                    * (static_cast<long double>(proposed_energy) - static_cast<long double>(energies_[replica_id]))
                + static_cast<long double>(log_prior_delta);
            if (AcceptLogProbability(log_acceptance, engine)) {
                if constexpr (detail::HasIncrementalGaussianTarget<TargetFunction>) {
                    target_.CommitIncrementalGaussianProposal(
                        data_,
                        layout_,
                        state,
                        incremental_caches_[replica_id],
                        offset,
                        proposed_value
                    );
                    apply_proposal();
                }
                energies_[replica_id] = proposed_energy;
                if (collect_diagnostics) {
                    ++mh_accept_counts_[replica_id];
                    ++mh_parameter_accept_counts_[replica_id][offset];
                }
            } else {
                if (state_was_mutated) {
                    restore_current();
                }
            }
        }

        void ExchangeReplicas(const bool collect_diagnostics) {
            if (states_.size() < 2) return;
            for (std::size_t replica_id = 0; replica_id + 1 < states_.size(); ++replica_id) {
                const std::size_t next_id = replica_id + 1;
                if (collect_diagnostics) {
                    ++exchange_attempt_counts_[replica_id];
                }
                const long double log_acceptance = exchange_log_factors_[replica_id]
                    * (static_cast<long double>(energies_[replica_id])
                        - static_cast<long double>(energies_[next_id]));
                if (AcceptLogProbability(log_acceptance, engine_)) {
                    std::swap(states_[replica_id], states_[next_id]);
                    std::swap(energies_[replica_id], energies_[next_id]);
                    SwapIncrementalCaches(replica_id, next_id);
                    if (collect_diagnostics) {
                        ++exchange_accept_counts_[replica_id];
                    }
                }
            }
        }

        bool AcceptLogProbability(const long double log_acceptance, RandomEngine & engine) {
            if (std::isnan(log_acceptance)) return false;
            if (log_acceptance >= 0.0) return true;
            if (!std::isfinite(log_acceptance)) return false;
            return std::log(static_cast<long double>(Uniform01(engine))) < log_acceptance;
        }

        static double Uniform01(RandomEngine & engine) {
            return UniformUnitDouble(engine);
        }

        SampleRecord MakeSample(const std::size_t sample_id, const std::size_t replica_id) const {
            const double beta = inverse_temperatures_[replica_id];
            return SampleRecord{
                sample_id,
                replica_id,
                beta,
                energies_[replica_id],
                -weighted_inverse_temperatures_[replica_id] * energies_[replica_id] + states_[replica_id].LogPrior(),
                states_[replica_id].values,
            };
        }

        void BeginProgress(const std::size_t total_steps) {
            if (!options_.progress_enabled) return;
            progress_started_at_ = std::chrono::steady_clock::now();
            if (options_.progress_interval_steps > 0) {
                progress_interval_steps_ = options_.progress_interval_steps;
            } else {
                progress_interval_steps_ = std::max<std::size_t>(1, total_steps / 20);
            }
            WriteProgress(0, total_steps, 0, false);
        }

        void MaybeReportProgress(
            const std::size_t completed_steps,
            const std::size_t total_steps,
            const std::size_t collected_samples
        ) {
            if (!options_.progress_enabled) return;
            const bool finished = completed_steps >= total_steps;
            if (!finished && completed_steps % progress_interval_steps_ != 0) return;
            WriteProgress(completed_steps, total_steps, collected_samples, finished);
        }

        void AbortProgress() const {
            if (!options_.progress_enabled) return;
            std::cerr << "\n";
        }

        void WriteProgress(
            const std::size_t completed_steps,
            const std::size_t total_steps,
            const std::size_t collected_samples,
            const bool finished
        ) const {
            const double fraction = total_steps == 0
                ? 1.0
                : std::min(1.0, static_cast<double>(completed_steps) / static_cast<double>(total_steps));
            const std::size_t width = std::max<std::size_t>(10, options_.progress_bar_width);
            const std::size_t filled = std::min<std::size_t>(
                width,
                static_cast<std::size_t>(std::floor(fraction * static_cast<double>(width)))
            );
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - progress_started_at_
            ).count();
            const char * phase = completed_steps < options_.burnin_count ? "burnin" : "sample";

            std::ostringstream message;
            message << "\r[bayes-emc] [";
            message << std::string(filled, '#') << std::string(width - filled, '-');
            message << "] " << std::fixed << std::setprecision(1) << std::setw(5) << (100.0 * fraction)
                    << "% step " << completed_steps << "/" << total_steps
                    << " phase " << phase
                    << " samples " << collected_samples << "/" << options_.sample_count
                    << " elapsed " << std::setprecision(1) << elapsed << "s";
            if (finished) {
                message << "\n";
            }
            std::cerr << message.str() << std::flush;
        }

        ParameterLayout layout_;
        DataSet data_;
        TargetFunction target_;
        EngineOptions options_;
        RandomEngine engine_;
        std::vector<RandomEngine> replica_engines_;
        std::vector<std::unique_ptr<LikelihoodThreadPool>> likelihood_pools_;
        std::vector<double> inverse_temperatures_;
        std::vector<double> weighted_inverse_temperatures_;
        std::vector<long double> exchange_log_factors_;
        std::vector<std::vector<double>> step_sizes_;
        mutable std::vector<std::vector<double>> prediction_buffers_;
        std::vector<IncrementalGaussianCache> incremental_caches_;
        std::vector<ParameterState> states_;
        std::vector<double> energies_;
        std::vector<std::size_t> mh_attempt_counts_;
        std::vector<std::size_t> mh_accept_counts_;
        std::vector<std::vector<std::size_t>> mh_parameter_attempt_counts_;
        std::vector<std::vector<std::size_t>> mh_parameter_accept_counts_;
        std::vector<std::size_t> exchange_attempt_counts_;
        std::vector<std::size_t> exchange_accept_counts_;
        std::size_t effective_replica_worker_count_ = 1;
        std::size_t effective_likelihood_worker_count_ = 1;
        std::chrono::steady_clock::time_point progress_started_at_;
        std::size_t progress_interval_steps_ = 1;
};

} // namespace bayes_emc
