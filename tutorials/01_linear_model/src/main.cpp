#include "bayes_emc/bayes_emc.hpp"
#include "generated_v2_config.hpp"
#include "target.hpp"

#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

std::filesystem::path PathFromEnv(const char * name, const char * fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    return value;
}

const bayes_emc::SampleRecord & MapSample(const std::vector<bayes_emc::SampleRecord> & samples) {
    if (samples.empty()) {
        throw std::runtime_error("No sample was generated.");
    }
    const bayes_emc::SampleRecord * best = &samples.front();
    for (const bayes_emc::SampleRecord & sample : samples) {
        if (sample.log_posterior > best->log_posterior) {
            best = &sample;
        }
    }
    return *best;
}

void WriteLog(
    const std::filesystem::path & path,
    const bayes_emc::EngineResult & result,
    const bayes_emc::SampleRecord & map,
    const std::size_t posterior_replica_id,
    const double posterior_sigma2,
    const std::size_t diagnostic_warning_count
) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Could not open V2 log output.");
    }
    out << "engine: bayes_emc_v2\n";
    out << "parallel_worker_count: " << result.parallel_worker_count << "\n";
    out << "likelihood_worker_count: " << result.likelihood_worker_count << "\n";
    out << "sample_count: " << result.samples.size() << "\n";
    out << "posterior_replica_id: " << posterior_replica_id << "\n";
    out << "posterior_inverse_temperature: " << result.inverse_temperatures[posterior_replica_id] << "\n";
    out << "posterior_sigma2: " << posterior_sigma2 << "\n";
    out << "map_sample_id: " << map.sample_id << "\n";
    out << "map_log_posterior: " << map.log_posterior << "\n";
    out << "map_energy: " << map.energy << "\n";
    out << "diagnostics_tsv: diagnostics.tsv\n";
    out << "diagnostics_warnings_tsv: diagnostics_warnings.tsv\n";
    out << "diagnostic_warning_count: " << diagnostic_warning_count << "\n";
    out << "map_parameters:\n";
    for (const bayes_emc::ParameterIndex & index : result.layout.Indices()) {
        const std::size_t offset = result.layout.Offset(index);
        out << "  " << result.layout.Label(index) << ": " << map.values[offset] << "\n";
    }
}

} // namespace

int main() {
    namespace fs = std::filesystem;

    auto spec = bayes_emc_generated::MakeAnalysisSpec();
    auto options = bayes_emc_generated::MakeEngineOptions();
    const fs::path data_path = PathFromEnv("BAYES_EMC_DATA_PATH", bayes_emc_generated::DefaultDataPath());
    const fs::path result_dir = PathFromEnv("BAYES_EMC_RESULT_DIR", bayes_emc_generated::DefaultResultDir());

    fs::create_directories(result_dir / "figures");
    bayes_emc::DataSet data = bayes_emc::DataSet::LoadTable(
        data_path.string(),
        spec.input_dim,
        spec.output_dim,
        bayes_emc_generated::MakeDataOptions()
    );

    bayes_emc::ExchangeMonteCarlo engine(spec, data, bayes_emc_user::TargetFunction, options);
    bayes_emc::EngineResult result = engine.Run();
    bayes_emc::NoiseEstimationResult noise = bayes_emc::EstimateGaussianNoiseByFreeEnergy(
        result,
        spec.gaussian_sigma2,
        data.Size(),
        spec.output_dim
    );
    if (!bayes_emc_generated::EstimateSigma2()) {
        noise = bayes_emc::SelectFixedGaussianNoise(noise, spec.gaussian_sigma2);
    }
    bayes_emc::WriteSampleJson(result_dir / "sample.json", result, noise.replica_id, noise.estimated_sigma2);
    bayes_emc::WriteNoiseEstimation(result_dir / "noise_estimation.txt", noise);
    bayes_emc::WriteDiagnosticsTsv(result_dir / "diagnostics.tsv", result);
    const std::size_t diagnostic_warning_count =
        bayes_emc::WriteDiagnosticsWarningsTsv(result_dir / "diagnostics_warnings.tsv", result);
    const std::vector<bayes_emc::SampleRecord> & posterior_samples =
        bayes_emc::SamplesForReplica(result, noise.replica_id);
    const bayes_emc::SampleRecord & map = MapSample(posterior_samples);
    WriteLog(result_dir / "log.txt", result, map, noise.replica_id, noise.estimated_sigma2, diagnostic_warning_count);

    std::cout << "samples: " << result.samples.size() << "\n";
    std::cout << "map_log_posterior: " << map.log_posterior << "\n";
    if (bayes_emc_generated::EstimateSigma2()) {
        std::cout << "estimated_sigma2: " << noise.estimated_sigma2 << "\n";
    } else {
        std::cout << "fixed_sigma2: " << noise.estimated_sigma2 << "\n";
    }
    std::cout << "noise_free_energy: " << static_cast<double>(noise.min_free_energy) << "\n";
    std::cout << "diagnostic_warnings: " << diagnostic_warning_count << "\n";
    for (const bayes_emc::ParameterIndex & index : result.layout.Indices()) {
        const std::size_t offset = result.layout.Offset(index);
        std::cout << result.layout.Label(index) << ": " << map.values[offset] << "\n";
    }
}
