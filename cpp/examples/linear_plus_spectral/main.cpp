#include "bayes_emc/bayes_emc.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

constexpr double kTrueIntercept = 0.25;
constexpr double kTrueSlope = 0.12;
constexpr double kNoiseSigma = 0.03;
constexpr int kDataCount = 100;

double GaussianPeak(const double x, const double a, const double mu, const double b) {
    const double diff = x - mu;
    return a * std::exp(-0.5 * b * diff * diff);
}

double TrueSignal(const double x) {
    return kTrueIntercept
        + kTrueSlope * x
        + GaussianPeak(x, 0.75, -0.45, 18.0)
        + GaussianPeak(x, 1.10, 1.15, 14.0);
}

bayes_emc::DataSet GenerateSyntheticData(const std::filesystem::path & output_path) {
    std::vector<std::vector<double>> rows;
    rows.reserve(kDataCount);

    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("Could not open synthetic data output.");
    }
    std::mt19937 rng(20260416);
    std::normal_distribution<double> noise_dist(0.0, kNoiseSigma);
    for (int i = 0; i < kDataCount; ++i) {
        const double x = -1.5 + 4.0 * static_cast<double>(i) / static_cast<double>(kDataCount - 1);
        const double y = TrueSignal(x) + noise_dist(rng);
        rows.push_back({x, y});
        out << x << " " << y << "\n";
    }
    return bayes_emc::DataSet::FromRows(1, 1, rows);
}

const bayes_emc::SampleRecord & MapSample(const bayes_emc::EngineResult & result) {
    if (result.samples.empty()) {
        throw std::runtime_error("No sample was generated.");
    }
    const bayes_emc::SampleRecord * best = &result.samples.front();
    for (const bayes_emc::SampleRecord & sample : result.samples) {
        if (sample.log_posterior > best->log_posterior) {
            best = &sample;
        }
    }
    return *best;
}

} // namespace

int main() {
    using namespace bayes_emc;

    std::filesystem::create_directories("figures");
    DataSet data = GenerateSyntheticData("linear_plus_spectral_data.txt");

    AnalysisSpec spec;
    spec.input_dim = 1;
    spec.output_dim = 1;
    spec.gaussian_sigma2 = kNoiseSigma * kNoiseSigma;
    spec.models.push_back(ModelSpec::WithParameters(
        "linear_background",
        1,
        {
            ParameterSpec{"intercept", PriorDistribution::Normal(0.0, 1.0), 0.25, 0.5},
            ParameterSpec{"slope", PriorDistribution::Normal(0.0, 0.5), 0.15, 0.5},
        }
    ));
    spec.models.push_back(ModelSpec::WithParameters(
        "spectral_peaks",
        2,
        {
            ParameterSpec{"a", PriorDistribution::Gamma(3.0, 0.3), 0.25, 0.5},
            ParameterSpec{"mu", PriorDistribution::Normal(0.4, 1.0), 0.25, 0.5},
            ParameterSpec{"b", PriorDistribution::Gamma(8.0, 2.0), 2.0, 0.5},
        }
    ));

    const auto target = GaussianPeakSumTarget::WithLinearBackground(0, 1);

    EngineOptions options;
    options.replica_count = 36;
    options.gamma = 1.6;
    options.burnin_count = 3000;
    options.sample_count = 600;
    options.sample_stride = 1;
    options.exchange_stride = 1;
    options.parallel_worker_count = 0;
    options.seed = 424242;

    ExchangeMonteCarlo engine(spec, data, target, options);
    EngineResult result = engine.Run();
    WriteSampleJson("linear_plus_spectral_sample.json", result);

    const SampleRecord & map = MapSample(result);
    std::cout << "samples: " << result.samples.size() << "\n";
    std::cout << "parameters: " << result.layout.Size() << "\n";
    std::cout << "true_intercept: " << kTrueIntercept << "\n";
    std::cout << "true_slope: " << kTrueSlope << "\n";
    std::cout << "map_log_posterior: " << map.log_posterior << "\n";
    for (const ParameterIndex & index : result.layout.Indices()) {
        const std::size_t offset = result.layout.Offset(index);
        std::cout << result.layout.Label(index) << ": " << map.values[offset] << "\n";
    }
}
