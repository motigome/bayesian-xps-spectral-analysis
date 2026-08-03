#pragma once

#include "bayes_emc/data.hpp"
#include "bayes_emc/parameter.hpp"

#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace bayes_emc {

class LikelihoodThreadPool {
    public:
        explicit LikelihoodThreadPool(const std::size_t worker_count) {
            if (worker_count <= 1) return;
            workers_.reserve(worker_count);
            for (std::size_t worker_id = 0; worker_id < worker_count; ++worker_id) {
                workers_.emplace_back([this, worker_id]() { WorkerLoop(worker_id); });
            }
        }

        LikelihoodThreadPool(const LikelihoodThreadPool &) = delete;
        LikelihoodThreadPool & operator=(const LikelihoodThreadPool &) = delete;

        ~LikelihoodThreadPool() {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stop_ = true;
            }
            start_cv_.notify_all();
            for (std::thread & worker : workers_) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
        }

        std::size_t WorkerCount() const {
            return workers_.empty() ? 1 : workers_.size();
        }

        template <class RangeFunction>
        void Run(const std::size_t task_count, RangeFunction && function) {
            if (workers_.empty() || task_count == 0) {
                function(0, 0, task_count);
                return;
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (active_) {
                    throw std::runtime_error("LikelihoodThreadPool cannot run nested tasks.");
                }
                task_count_ = task_count;
                task_ = std::function<void(std::size_t, std::size_t, std::size_t)>(
                    std::forward<RangeFunction>(function)
                );
                exception_ = nullptr;
                finished_count_ = 0;
                active_ = true;
                ++generation_;
            }
            start_cv_.notify_all();
            std::unique_lock<std::mutex> lock(mutex_);
            done_cv_.wait(lock, [this]() {
                return !active_;
            });
            if (exception_ != nullptr) {
                std::rethrow_exception(exception_);
            }
        }

    private:
        void WorkerLoop(const std::size_t worker_id) {
            std::size_t observed_generation = 0;
            while (true) {
                std::function<void(std::size_t, std::size_t, std::size_t)> task;
                std::size_t begin = 0;
                std::size_t end = 0;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    start_cv_.wait(lock, [this, observed_generation]() {
                        return stop_ || generation_ != observed_generation;
                    });
                    if (stop_) return;
                    observed_generation = generation_;
                    task = task_;
                    begin = worker_id * task_count_ / workers_.size();
                    end = (worker_id + 1) * task_count_ / workers_.size();
                }
                try {
                    if (begin < end) {
                        task(worker_id, begin, end);
                    }
                } catch (...) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (exception_ == nullptr) {
                        exception_ = std::current_exception();
                    }
                }
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    ++finished_count_;
                    if (finished_count_ == workers_.size()) {
                        active_ = false;
                    }
                }
                done_cv_.notify_one();
            }
        }

        mutable std::mutex mutex_;
        std::condition_variable start_cv_;
        std::condition_variable done_cv_;
        std::vector<std::thread> workers_;
        std::function<void(std::size_t, std::size_t, std::size_t)> task_;
        std::size_t task_count_ = 0;
        std::size_t generation_ = 0;
        std::size_t finished_count_ = 0;
        bool active_ = false;
        bool stop_ = false;
        std::exception_ptr exception_ = nullptr;
};

namespace detail {

template <class TargetFunction>
auto EvaluateTargetIntoImpl(
    const TargetFunction & target,
    const std::vector<double> & x,
    const ParameterView & view,
    std::vector<double> & output,
    int
) -> decltype(target(x, view, output), void()) {
    target(x, view, output);
}

template <class TargetFunction>
void EvaluateTargetIntoImpl(
    const TargetFunction & target,
    const std::vector<double> & x,
    const ParameterView & view,
    std::vector<double> & output,
    long
) {
    output = target(x, view);
}

template <class TargetFunction>
void EvaluateTargetInto(
    const TargetFunction & target,
    const std::vector<double> & x,
    const ParameterView & view,
    std::vector<double> & output
) {
    EvaluateTargetIntoImpl(target, x, view, output, 0);
}

} // namespace detail

template <class TargetFunction>
double GaussianEnergy(
    const DataSet & data,
    const ParameterLayout & layout,
    const ParameterState & state,
    const TargetFunction & target,
    const double sigma2,
    LikelihoodThreadPool * thread_pool = nullptr,
    const std::size_t min_parallel_observations = 2048
) {
    if (!(sigma2 > 0.0)) {
        throw std::invalid_argument("sigma2 must be positive.");
    }
    const ParameterView view = state.View(layout);
    const std::size_t output_dim = data.OutputDim();
    const auto & observations = data.Observations();

    const auto evaluate_range = [&](const std::size_t begin, const std::size_t end, std::vector<double> & prediction) {
        double partial = 0.0;
        for (std::size_t observation_id = begin; observation_id < end; ++observation_id) {
            const Observation & observation = observations[observation_id];
            detail::EvaluateTargetInto(target, observation.x, view, prediction);
            if (prediction.size() != output_dim) {
                throw std::runtime_error("TargetFunction returned an unexpected output dimension.");
            }
            for (std::size_t output_id = 0; output_id < prediction.size(); ++output_id) {
                const double diff = observation.y[output_id] - prediction[output_id];
                partial += diff * diff;
            }
        }
        return partial;
    };

    double squared_error = 0.0;
    if (thread_pool != nullptr
        && thread_pool->WorkerCount() > 1
        && observations.size() >= min_parallel_observations) {
        std::vector<double> partials(thread_pool->WorkerCount(), 0.0);
        thread_pool->Run(observations.size(), [&](const std::size_t worker_id, const std::size_t begin, const std::size_t end) {
            std::vector<double> prediction(output_dim);
            partials[worker_id] = evaluate_range(begin, end, prediction);
        });
        for (const double partial : partials) {
            squared_error += partial;
        }
    } else {
        std::vector<double> prediction(output_dim);
        squared_error = evaluate_range(0, observations.size(), prediction);
    }

    if (!std::isfinite(squared_error)) {
        return std::numeric_limits<double>::infinity();
    }
    return squared_error / (2.0 * static_cast<double>(data.Size()) * sigma2);
}

template <class TargetFunction>
double GaussianEnergy(
    const DataSet & data,
    const ParameterLayout & layout,
    const ParameterState & state,
    const TargetFunction & target,
    const double sigma2,
    std::vector<double> & prediction_buffer
) {
    if (!(sigma2 > 0.0)) {
        throw std::invalid_argument("sigma2 must be positive.");
    }
    const ParameterView view = state.View(layout);
    const std::size_t output_dim = data.OutputDim();
    if (prediction_buffer.size() != output_dim) {
        prediction_buffer.assign(output_dim, 0.0);
    }

    double squared_error = 0.0;
    for (const Observation & observation : data.Observations()) {
        detail::EvaluateTargetInto(target, observation.x, view, prediction_buffer);
        if (prediction_buffer.size() != output_dim) {
            throw std::runtime_error("TargetFunction returned an unexpected output dimension.");
        }
        for (std::size_t output_id = 0; output_id < output_dim; ++output_id) {
            const double diff = observation.y[output_id] - prediction_buffer[output_id];
            squared_error += diff * diff;
        }
    }
    if (!std::isfinite(squared_error)) {
        return std::numeric_limits<double>::infinity();
    }
    return squared_error / (2.0 * static_cast<double>(data.Size()) * sigma2);
}

template <class TargetFunction>
double PoissonEnergy(
    const DataSet & data,
    const ParameterLayout & layout,
    const ParameterState & state,
    const TargetFunction & target,
    LikelihoodThreadPool * thread_pool = nullptr,
    const std::size_t min_parallel_observations = 2048
) {
    const ParameterView view = state.View(layout);
    const std::size_t output_dim = data.OutputDim();
    const auto & observations = data.Observations();

    const auto evaluate_range = [&](const std::size_t begin, const std::size_t end, std::vector<double> & prediction) {
        double partial = 0.0;
        for (std::size_t observation_id = begin; observation_id < end; ++observation_id) {
            const Observation & observation = observations[observation_id];
            detail::EvaluateTargetInto(target, observation.x, view, prediction);
            if (prediction.size() != output_dim) {
                throw std::runtime_error("TargetFunction returned an unexpected output dimension.");
            }
            for (std::size_t output_id = 0; output_id < prediction.size(); ++output_id) {
                const double count = observation.y[output_id];
                const double rate = prediction[output_id];
                if (!(count >= 0.0) || !(rate > 0.0)) {
                    return std::numeric_limits<double>::infinity();
                }
                partial += rate - count * std::log(rate) + std::lgamma(count + 1.0);
            }
        }
        return partial;
    };

    double negative_log_likelihood = 0.0;
    if (thread_pool != nullptr
        && thread_pool->WorkerCount() > 1
        && observations.size() >= min_parallel_observations) {
        std::vector<double> partials(thread_pool->WorkerCount(), 0.0);
        thread_pool->Run(observations.size(), [&](const std::size_t worker_id, const std::size_t begin, const std::size_t end) {
            std::vector<double> prediction(output_dim);
            partials[worker_id] = evaluate_range(begin, end, prediction);
        });
        for (const double partial : partials) {
            negative_log_likelihood += partial;
        }
    } else {
        std::vector<double> prediction(output_dim);
        negative_log_likelihood = evaluate_range(0, observations.size(), prediction);
    }

    if (!std::isfinite(negative_log_likelihood)) {
        return std::numeric_limits<double>::infinity();
    }
    return negative_log_likelihood / static_cast<double>(data.Size());
}

template <class TargetFunction>
double PoissonEnergy(
    const DataSet & data,
    const ParameterLayout & layout,
    const ParameterState & state,
    const TargetFunction & target,
    std::vector<double> & prediction_buffer
) {
    const ParameterView view = state.View(layout);
    const std::size_t output_dim = data.OutputDim();
    if (prediction_buffer.size() != output_dim) {
        prediction_buffer.assign(output_dim, 0.0);
    }

    double negative_log_likelihood = 0.0;
    for (const Observation & observation : data.Observations()) {
        detail::EvaluateTargetInto(target, observation.x, view, prediction_buffer);
        if (prediction_buffer.size() != output_dim) {
            throw std::runtime_error("TargetFunction returned an unexpected output dimension.");
        }
        for (std::size_t output_id = 0; output_id < output_dim; ++output_id) {
            const double count = observation.y[output_id];
            const double rate = prediction_buffer[output_id];
            if (!(count >= 0.0) || !(rate > 0.0)) {
                return std::numeric_limits<double>::infinity();
            }
            negative_log_likelihood += rate - count * std::log(rate) + std::lgamma(count + 1.0);
        }
    }
    if (!std::isfinite(negative_log_likelihood)) {
        return std::numeric_limits<double>::infinity();
    }
    return negative_log_likelihood / static_cast<double>(data.Size());
}

} // namespace bayes_emc
