#pragma once

#include "bayes_emc/emc_engine.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bayes_emc {

inline std::string JsonEscape(const std::string & value) {
    std::string escaped;
    for (const char ch : value) {
        if (ch == '"') {
            escaped += "\\\"";
        } else if (ch == '\\') {
            escaped += "\\\\";
        } else if (ch == '\n') {
            escaped += "\\n";
        } else {
            escaped += ch;
        }
    }
    return escaped;
}

inline void WriteValuesJson(std::ostream & out, const std::vector<double> & values) {
    out << "[";
    for (std::size_t value_id = 0; value_id < values.size(); ++value_id) {
        if (value_id > 0) out << ", ";
        out << values[value_id];
    }
    out << "]";
}

inline const std::vector<SampleRecord> & SamplesForReplica(
    const EngineResult & result,
    const std::size_t replica_id
) {
    if (!result.replica_samples.empty()) {
        if (replica_id >= result.replica_samples.size()) {
            throw std::out_of_range("posterior replica id is outside replica_samples.");
        }
        return result.replica_samples[replica_id];
    }
    if (replica_id + 1 != result.inverse_temperatures.size()) {
        throw std::out_of_range("Only cold-replica samples are available in this EngineResult.");
    }
    return result.samples;
}

inline void WriteSampleJson(
    const std::filesystem::path & path,
    const EngineResult & result,
    const std::size_t posterior_replica_id,
    const double posterior_sigma2
) {
    if (posterior_replica_id >= result.inverse_temperatures.size()) {
        throw std::out_of_range("posterior replica id is outside inverse_temperatures.");
    }
    const std::vector<SampleRecord> & samples = SamplesForReplica(result, posterior_replica_id);
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Could not open sample output: " + path.string());
    }
    out << std::setprecision(17);
    out << "{\n";
    out << "  \"schema_version\": 4,\n";
    out << "  \"engine\": \"bayes_emc_v2\",\n";
    out << "  \"parallel_worker_count\": " << result.parallel_worker_count << ",\n";
    out << "  \"likelihood_worker_count\": " << result.likelihood_worker_count << ",\n";
    out << "  \"posterior_replica_id\": " << posterior_replica_id << ",\n";
    out << "  \"posterior_inverse_temperature\": " << result.inverse_temperatures[posterior_replica_id] << ",\n";
    out << "  \"posterior_sigma2\": " << posterior_sigma2 << ",\n";
    out << "  \"sample_count\": " << samples.size() << ",\n";
    out << "  \"parameters\": [\n";
    const auto & indices = result.layout.Indices();
    for (std::size_t i = 0; i < indices.size(); ++i) {
        const ParameterIndex & index = indices[i];
        out << "    {"
            << "\"offset\": " << i
            << ", \"model_id\": " << index.model
            << ", \"basis_id\": " << index.basis
            << ", \"layer_id\": " << index.layer
            << ", \"parameter_id\": " << index.parameter
            << ", \"label\": \"" << JsonEscape(result.layout.Label(index)) << "\""
            << "}";
        if (i + 1 < indices.size()) out << ",";
        out << "\n";
    }
    out << "  ],\n";
    out << "  \"samples\": {\n";
    out << "    \"format\": \"columnar_v1\",\n";
    out << "    \"energy\": [";
    for (std::size_t sample_id = 0; sample_id < samples.size(); ++sample_id) {
        const SampleRecord & sample = samples[sample_id];
        if (sample_id > 0) out << ", ";
        out << sample.energy;
    }
    out << "],\n";
    out << "    \"log_posterior\": [";
    for (std::size_t sample_id = 0; sample_id < samples.size(); ++sample_id) {
        const SampleRecord & sample = samples[sample_id];
        if (sample_id > 0) out << ", ";
        out << sample.log_posterior;
    }
    out << "],\n";
    out << "    \"values\": [\n";
    for (std::size_t sample_id = 0; sample_id < samples.size(); ++sample_id) {
        const SampleRecord & sample = samples[sample_id];
        out << "      ";
        WriteValuesJson(out, sample.values);
        if (sample_id + 1 < samples.size()) out << ",";
        out << "\n";
    }
    out << "    ]\n";
    out << "  }\n";
    out << "}\n";
}

inline void WriteSampleJson(const std::filesystem::path & path, const EngineResult & result) {
    if (result.inverse_temperatures.empty()) {
        throw std::invalid_argument("WriteSampleJson requires at least one inverse temperature.");
    }
    const std::size_t cold_replica_id = result.inverse_temperatures.size() - 1;
    WriteSampleJson(path, result, cold_replica_id, result.layout.Spec().gaussian_sigma2);
}

} // namespace bayes_emc
