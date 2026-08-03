#include "bayes_emc/bayes_emc.hpp"

#include <iostream>
#include <vector>

int main() {
    using namespace bayes_emc;

    DataSet data = DataSet::FromRows(
        1,
        1,
        {
            {-1.0, -1.0},
            {0.0, 1.0},
            {1.0, 3.0},
            {2.0, 5.0},
            {3.0, 7.0},
        }
    );

    AnalysisSpec spec;
    spec.input_dim = 1;
    spec.output_dim = 1;
    spec.gaussian_sigma2 = 0.05;
    spec.models.push_back(ModelSpec::WithParameters(
        "linear",
        1,
        {
            ParameterSpec{"intercept", PriorDistribution::Normal(0.0, 5.0), 0.4, 0.5},
            ParameterSpec{"slope", PriorDistribution::Normal(0.0, 5.0), 0.4, 0.5},
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
    options.burnin_count = 20;
    options.sample_count = 10;
    options.parallel_worker_count = 0;
    options.seed = 17;

    ExchangeMonteCarlo engine(spec, data, target, options);
    EngineResult result = engine.Run();
    WriteSampleJson("linear_1d_sample.json", result);

    const SampleRecord & last = result.samples.back();
    std::cout << "samples: " << result.samples.size() << "\n";
    std::cout << "last intercept: " << last.values[0] << "\n";
    std::cout << "last slope: " << last.values[1] << "\n";
}
