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

inline double DiagnosticRate(const std::size_t accepted, const std::size_t attempted) {
    if (attempted == 0) return 0.0;
    return static_cast<double>(accepted) / static_cast<double>(attempted);
}

inline bool IsDiagnosticWarningRate(
    const double rate,
    const double low_threshold,
    const double high_threshold
) {
    return rate < low_threshold || rate > high_threshold;
}

inline const char * DiagnosticWarningReason(
    const double rate,
    const double low_threshold,
    const double high_threshold
) {
    if (rate < low_threshold) return "below_low_threshold";
    if (rate > high_threshold) return "above_high_threshold";
    return "";
}

inline bool UsesUnscaledProposalWidth(const EngineResult & result, const std::size_t replica_id) {
    if (result.data_count == 0 || replica_id >= result.inverse_temperatures.size()) return false;
    return static_cast<double>(result.data_count) * result.inverse_temperatures[replica_id] < 1.0;
}

inline double MeanValue(const std::vector<double> & values) {
    if (values.empty()) return 0.0;
    double total = 0.0;
    for (const double value : values) {
        total += value;
    }
    return total / static_cast<double>(values.size());
}

inline void WriteDiagnosticsTsv(std::ostream & out, const EngineResult & result) {
    out << "Inv Temp";
    const auto & indices = result.layout.Indices();
    for (const ParameterIndex & index : indices) {
        out << "\t" << result.layout.Label(index);
    }
    out << "\tExchange %\t<Energy>\n";

    const std::size_t replica_count = result.inverse_temperatures.size();
    for (std::size_t replica_id = 0; replica_id < replica_count; ++replica_id) {
        if (UsesUnscaledProposalWidth(result, replica_id)) {
            out << "*";
        }
        out << std::scientific << std::setprecision(2) << result.inverse_temperatures[replica_id];
        for (std::size_t offset = 0; offset < indices.size(); ++offset) {
            std::size_t attempted = 0;
            std::size_t accepted = 0;
            if (replica_id < result.mh_parameter_attempt_counts.size()
                && offset < result.mh_parameter_attempt_counts[replica_id].size()) {
                attempted = result.mh_parameter_attempt_counts[replica_id][offset];
            }
            if (replica_id < result.mh_parameter_accept_counts.size()
                && offset < result.mh_parameter_accept_counts[replica_id].size()) {
                accepted = result.mh_parameter_accept_counts[replica_id][offset];
            }
            out << "\t" << std::fixed << std::setprecision(2)
                << 100.0 * DiagnosticRate(accepted, attempted);
        }

        if (replica_id + 1 < replica_count && replica_id < result.exchange_attempt_counts.size()) {
            out << "\t" << std::fixed << std::setprecision(2)
                << 100.0 * DiagnosticRate(
                    result.exchange_accept_counts[replica_id],
                    result.exchange_attempt_counts[replica_id]
                );
        } else {
            out << "\t*****";
        }

        double mean_energy = 0.0;
        if (replica_id < result.replica_energy_samples.size()) {
            mean_energy = MeanValue(result.replica_energy_samples[replica_id]);
        }
        out << "\t" << std::scientific << std::setprecision(2) << mean_energy << "\n";
    }
}

inline void WriteDiagnosticsTsv(const std::filesystem::path & path, const EngineResult & result) {
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Could not open diagnostics TSV output: " + path.string());
    }
    WriteDiagnosticsTsv(out, result);
}

inline std::size_t WriteDiagnosticsWarningsTsv(
    std::ostream & out,
    const EngineResult & result,
    const double low_threshold = 0.10,
    const double high_threshold = 0.99
) {
    out << "type\treplica_id\tnext_replica_id\tinv_temp\tnext_inv_temp\ttarget\trate_percent\twarning\n";

    std::size_t warning_count = 0;
    const std::size_t replica_count = result.inverse_temperatures.size();
    const auto & indices = result.layout.Indices();
    for (std::size_t replica_id = 0; replica_id < replica_count; ++replica_id) {
        const bool prior_replica = result.inverse_temperatures[replica_id] == 0.0;
        for (std::size_t offset = 0; offset < indices.size(); ++offset) {
            if (prior_replica) continue;
            std::size_t attempted = 0;
            std::size_t accepted = 0;
            if (replica_id < result.mh_parameter_attempt_counts.size()
                && offset < result.mh_parameter_attempt_counts[replica_id].size()) {
                attempted = result.mh_parameter_attempt_counts[replica_id][offset];
            }
            if (replica_id < result.mh_parameter_accept_counts.size()
                && offset < result.mh_parameter_accept_counts[replica_id].size()) {
                accepted = result.mh_parameter_accept_counts[replica_id][offset];
            }
            if (attempted == 0) continue;

            const double rate = DiagnosticRate(accepted, attempted);
            if (!IsDiagnosticWarningRate(rate, low_threshold, high_threshold)) continue;
            out << "mh_acceptance"
                << "\t" << replica_id
                << "\t"
                << "\t" << std::scientific << std::setprecision(2) << result.inverse_temperatures[replica_id]
                << "\t"
                << "\t" << result.layout.Label(indices[offset])
                << "\t" << std::fixed << std::setprecision(2) << 100.0 * rate
                << "\t" << DiagnosticWarningReason(rate, low_threshold, high_threshold)
                << "\n";
            ++warning_count;
        }

        if (replica_id + 1 < replica_count && replica_id < result.exchange_attempt_counts.size()) {
            const std::size_t attempted = result.exchange_attempt_counts[replica_id];
            if (attempted == 0) continue;
            const double rate = DiagnosticRate(result.exchange_accept_counts[replica_id], attempted);
            if (!IsDiagnosticWarningRate(rate, low_threshold, high_threshold)) continue;
            out << "replica_exchange"
                << "\t" << replica_id
                << "\t" << (replica_id + 1)
                << "\t" << std::scientific << std::setprecision(2) << result.inverse_temperatures[replica_id]
                << "\t" << std::scientific << std::setprecision(2) << result.inverse_temperatures[replica_id + 1]
                << "\texchange_to_next"
                << "\t" << std::fixed << std::setprecision(2) << 100.0 * rate
                << "\t" << DiagnosticWarningReason(rate, low_threshold, high_threshold)
                << "\n";
            ++warning_count;
        }
    }
    return warning_count;
}

inline std::size_t WriteDiagnosticsWarningsTsv(
    const std::filesystem::path & path,
    const EngineResult & result,
    const double low_threshold = 0.10,
    const double high_threshold = 0.99
) {
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Could not open diagnostics warnings TSV output: " + path.string());
    }
    return WriteDiagnosticsWarningsTsv(out, result, low_threshold, high_threshold);
}

} // namespace bayes_emc
