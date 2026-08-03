#pragma once

#include "bayes_emc/emc_engine.hpp"

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <vector>

namespace bayes_emc {

struct NoiseEstimationRecord {
    double sigma2 = 0.0;
    double inverse_temperature = 0.0;
    long double free_energy = 0.0L;
};

struct NoiseEstimationResult {
    bool estimate_sigma2 = true;
    double estimated_sigma2 = 0.0;
    std::size_t replica_id = 0;
    long double min_free_energy = 0.0L;
    std::vector<NoiseEstimationRecord> records;
};

inline long double LogMeanExp(const std::vector<long double> & values) {
    if (values.empty()) {
        throw std::invalid_argument("LogMeanExp requires at least one value.");
    }
    long double max_value = -std::numeric_limits<long double>::infinity();
    for (const long double value : values) {
        if (value > max_value) max_value = value;
    }
    long double sum = 0.0L;
    for (const long double value : values) {
        sum += std::exp(value - max_value);
    }
    return max_value + std::log(sum / static_cast<long double>(values.size()));
}

inline NoiseEstimationResult EstimateGaussianNoiseByFreeEnergy(
    const EngineResult & result,
    const double base_sigma2,
    const std::size_t data_count,
    const std::size_t output_dim
) {
    if (!(base_sigma2 > 0.0)) {
        throw std::invalid_argument("base_sigma2 must be positive.");
    }
    if (data_count == 0 || output_dim == 0) {
        throw std::invalid_argument("data_count and output_dim must be positive.");
    }
    if (result.inverse_temperatures.size() < 2) {
        throw std::invalid_argument("Noise estimation requires at least two replicas.");
    }
    if (result.inverse_temperatures.size() != result.replica_energy_samples.size()) {
        throw std::invalid_argument("Replica energy samples do not match inverse temperatures.");
    }

    constexpr long double pi = 3.141592653589793238462643383279502884L;
    const long double energy_scale = static_cast<long double>(data_count);
    const long double observation_count = static_cast<long double>(data_count)
        * static_cast<long double>(output_dim);
    long double log_integral = 0.0L;

    NoiseEstimationResult estimation;
    estimation.min_free_energy = std::numeric_limits<long double>::infinity();

    for (std::size_t interval_id = 0; interval_id + 1 < result.inverse_temperatures.size(); ++interval_id) {
        const long double beta_left = result.inverse_temperatures[interval_id];
        const long double beta_right = result.inverse_temperatures[interval_id + 1];
        const long double delta_beta = beta_right - beta_left;
        if (!(delta_beta > 0.0L)) {
            throw std::invalid_argument("inverse_temperatures must be strictly increasing.");
        }
        const std::vector<double> & left_energies = result.replica_energy_samples[interval_id];
        const std::vector<double> & right_energies = result.replica_energy_samples[interval_id + 1];
        if (left_energies.empty() || right_energies.empty()) {
            throw std::invalid_argument("Noise estimation requires energy samples for every replica.");
        }

        std::vector<long double> forward_terms;
        std::vector<long double> reverse_terms;
        forward_terms.reserve(left_energies.size());
        reverse_terms.reserve(right_energies.size());
        for (const double energy : left_energies) {
            forward_terms.push_back(-delta_beta * energy_scale * static_cast<long double>(energy));
        }
        for (const double energy : right_energies) {
            reverse_terms.push_back(delta_beta * energy_scale * static_cast<long double>(energy));
        }

        const long double forward_log_ratio = LogMeanExp(forward_terms);
        const long double reverse_log_ratio = -LogMeanExp(reverse_terms);
        log_integral += 0.5L * (forward_log_ratio + reverse_log_ratio);

        const long double normalization = 0.5L * observation_count
            * std::log(beta_right / (2.0L * pi * static_cast<long double>(base_sigma2)));
        const long double free_energy = -normalization - log_integral;
        const double sigma2 = base_sigma2 / static_cast<double>(beta_right);
        estimation.records.push_back(NoiseEstimationRecord{
            sigma2,
            static_cast<double>(beta_right),
            free_energy,
        });
        if (free_energy < estimation.min_free_energy) {
            estimation.min_free_energy = free_energy;
            estimation.estimated_sigma2 = sigma2;
            estimation.replica_id = interval_id + 1;
        }
    }
    return estimation;
}

inline NoiseEstimationResult SelectFixedGaussianNoise(
    NoiseEstimationResult estimation,
    const double fixed_sigma2
) {
    if (!(fixed_sigma2 > 0.0)) {
        throw std::invalid_argument("fixed_sigma2 must be positive.");
    }
    if (estimation.records.empty()) {
        throw std::invalid_argument("SelectFixedGaussianNoise requires at least one record.");
    }
    std::size_t selected_id = 0;
    double best_distance = std::numeric_limits<double>::infinity();
    for (std::size_t record_id = 0; record_id < estimation.records.size(); ++record_id) {
        const double distance = std::abs(estimation.records[record_id].sigma2 - fixed_sigma2);
        if (distance < best_distance) {
            best_distance = distance;
            selected_id = record_id;
        }
    }
    estimation.estimate_sigma2 = false;
    estimation.estimated_sigma2 = fixed_sigma2;
    estimation.replica_id = selected_id + 1;
    estimation.min_free_energy = estimation.records[selected_id].free_energy;
    return estimation;
}

inline void WriteNoiseEstimation(
    const std::filesystem::path & path,
    const NoiseEstimationResult & estimation
) {
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Could not open noise estimation output: " + path.string());
    }
    out << std::setprecision(17);
    out << "# sigma2_mode\t" << (estimation.estimate_sigma2 ? "estimated" : "fixed") << "\n";
    out << "# estimated_sigma2\t" << estimation.estimated_sigma2 << "\n";
    out << "# replica_id\t" << estimation.replica_id << "\n";
    out << "# replica_index_base\t0\n";
    out << "# min_free_energy\t" << static_cast<double>(estimation.min_free_energy) << "\n";
    out << "# selected_free_energy\t" << static_cast<double>(estimation.min_free_energy) << "\n";
    if (!estimation.records.empty()) {
        out << "# candidate_min_sigma2\t" << estimation.records.back().sigma2 << "\n";
        out << "# candidate_max_sigma2\t" << estimation.records.front().sigma2 << "\n";
        out << "# candidate_count\t" << estimation.records.size() << "\n";
    }
    out << "sigma2\tinverse_temperature\tfree_energy\n";
    for (const NoiseEstimationRecord & record : estimation.records) {
        out << record.sigma2 << "\t"
            << record.inverse_temperature << "\t"
            << static_cast<double>(record.free_energy) << "\n";
    }
}

} // namespace bayes_emc
