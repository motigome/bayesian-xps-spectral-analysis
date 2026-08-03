#include "bayes_emc/bayes_emc.hpp"

#include <iostream>
#include <vector>

int main() {
    using namespace bayes_emc;

    DataSet data = DataSet::FromRows(
        1,
        1,
        {
            {0.0, 0.10},
            {0.5, 0.55},
            {1.0, 1.20},
            {1.5, 0.95},
            {2.0, 0.30},
        }
    );

    AnalysisSpec spec;
    spec.input_dim = 1;
    spec.output_dim = 1;
    spec.gaussian_sigma2 = 0.05;
    spec.models.push_back(ModelSpec::WithParameters(
        "spectral_peaks",
        3,
        {
            ParameterSpec{"a", PriorDistribution::Gamma(2.0, 0.5), 0.3, 0.5},
            ParameterSpec{"mu", PriorDistribution::Normal(1.0, 0.8), 0.2, 0.5},
            ParameterSpec{"b", PriorDistribution::Gamma(4.0, 4.0), 2.0, 0.5},
        }
    ));

    const auto target = GaussianPeakSumTarget::SpectralOnly(0);

    EngineOptions options;
    options.replica_count = 36;
    options.gamma = 1.6;
    options.burnin_count = 10;
    options.sample_count = 5;
    options.parallel_worker_count = 0;
    options.seed = 23;

    ExchangeMonteCarlo engine(spec, data, target, options);
    EngineResult result = engine.Run();
    WriteSampleJson("spectral_basis_sample.json", result);

    std::cout << "samples: " << result.samples.size() << "\n";
    std::cout << "parameters: " << result.layout.Size() << "\n";
}
