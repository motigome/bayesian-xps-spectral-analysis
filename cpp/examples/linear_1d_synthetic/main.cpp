#include "bayes_emc/bayes_emc.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

constexpr double kTrueIntercept = 1.25;
constexpr double kTrueSlope = -0.80;
constexpr double kNoiseSigma = 0.05;
constexpr int kDataCount = 100;

bayes_emc::DataSet GenerateSyntheticData(const std::filesystem::path & output_path) {
    std::vector<std::vector<double>> rows;
    rows.reserve(kDataCount);

    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("Could not open synthetic data output.");
    }
    std::mt19937 rng(20260415);
    std::normal_distribution<double> noise_dist(0.0, kNoiseSigma);
    for (int i = 0; i < kDataCount; ++i) {
        const double x = -2.0 + 4.0 * static_cast<double>(i) / static_cast<double>(kDataCount - 1);
        const double y = kTrueIntercept + kTrueSlope * x + noise_dist(rng);
        rows.push_back({x, y});
        out << x << " " << y << "\n";
    }
    return bayes_emc::DataSet::FromRows(1, 1, rows);
}

} // namespace

int main() {
    using namespace bayes_emc;

    std::filesystem::create_directories("figures");
    DataSet data = GenerateSyntheticData("linear_1d_synthetic_data.txt");

    AnalysisSpec spec;
    spec.input_dim = 1;
    spec.output_dim = 1;
    spec.gaussian_sigma2 = kNoiseSigma * kNoiseSigma;
    spec.models.push_back(ModelSpec::WithParameters(
        "linear",
        1,
        {
            ParameterSpec{"intercept", PriorDistribution::Normal(0.0, 5.0), 0.9, 0.5},
            ParameterSpec{"slope", PriorDistribution::Normal(0.0, 5.0), 0.9, 0.5},
        }
    ));

    auto target = [](const std::vector<double> & x, const ParameterView & params, std::vector<double> & out) {
        const double intercept = params.Value(0, 0, 0);
        const double slope = params.Value(0, 0, 1);
        out[0] = intercept + slope * x[0];
    };

    EngineOptions options;
    options.replica_count = 36;
    options.gamma = 1.6;
    options.burnin_count = 2500;
    options.sample_count = 800;
    options.sample_stride = 1;
    options.exchange_stride = 1;
    options.parallel_worker_count = 0;
    options.seed = 12345;

    ExchangeMonteCarlo engine(spec, data, target, options);
    EngineResult result = engine.Run();
    WriteSampleJson("linear_1d_synthetic_sample.json", result);

    const SampleRecord * map = nullptr;
    for (const SampleRecord & sample : result.samples) {
        if (map == nullptr || sample.log_posterior > map->log_posterior) {
            map = &sample;
        }
    }
    if (map == nullptr) {
        throw std::runtime_error("No sample was generated.");
    }

    std::cout << "true_intercept: " << kTrueIntercept << "\n";
    std::cout << "true_slope: " << kTrueSlope << "\n";
    std::cout << "map_intercept: " << map->values[0] << "\n";
    std::cout << "map_slope: " << map->values[1] << "\n";
    std::cout << "samples: " << result.samples.size() << "\n";
}
