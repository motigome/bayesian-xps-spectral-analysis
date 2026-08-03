#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace bayes_emc {

enum class PriorKind {
    Uniform,
    Normal,
    Gamma,
    Beta,
};

class PriorDistribution {
    public:
        PriorDistribution() = default;

        static PriorDistribution Uniform(const double lower, const double upper) {
            if (!(lower < upper)) {
                throw std::invalid_argument("Uniform prior requires lower < upper.");
            }
            return PriorDistribution(PriorKind::Uniform, lower, upper);
        }

        static PriorDistribution Normal(const double mean, const double sigma) {
            if (!(sigma > 0)) {
                throw std::invalid_argument("Normal prior requires sigma > 0.");
            }
            return PriorDistribution(PriorKind::Normal, mean, sigma);
        }

        static PriorDistribution Gamma(const double shape, const double scale) {
            if (!(shape > 0 && scale > 0)) {
                throw std::invalid_argument("Gamma prior requires shape > 0 and scale > 0.");
            }
            return PriorDistribution(PriorKind::Gamma, shape, scale);
        }

        static PriorDistribution Beta(
            const double alpha,
            const double beta,
            const double lower,
            const double upper
        ) {
            if (!(alpha > 0 && beta > 0)) {
                throw std::invalid_argument("Beta prior requires alpha > 0 and beta > 0.");
            }
            if (!(lower < upper)) {
                throw std::invalid_argument("Beta prior requires lower < upper.");
            }
            return PriorDistribution(PriorKind::Beta, alpha, beta, lower, upper);
        }

        template <class RandomEngine>
        double Sample(RandomEngine & engine) const {
            if (kind_ == PriorKind::Uniform) {
                std::uniform_real_distribution<double> dist(first_, second_);
                return dist(engine);
            }
            if (kind_ == PriorKind::Normal) {
                std::normal_distribution<double> dist(first_, second_);
                return dist(engine);
            }
            if (kind_ == PriorKind::Gamma) {
                std::gamma_distribution<double> dist(first_, second_);
                return dist(engine);
            }
            std::gamma_distribution<double> alpha_dist(first_, 1.0);
            std::gamma_distribution<double> beta_dist(second_, 1.0);
            const double x = alpha_dist(engine);
            const double y = beta_dist(engine);
            const double u = x / (x + y);
            return third_ + (fourth_ - third_) * u;
        }

        double Pdf(const double value) const {
            const double log_pdf = LogPdf(value);
            if (!std::isfinite(log_pdf)) return 0.0;
            return std::exp(log_pdf);
        }

        double LogPdf(const double value) const {
            if (kind_ == PriorKind::Uniform) {
                if (value < first_ || value > second_) return NegativeInfinity();
                return -std::log(second_ - first_);
            }
            if (kind_ == PriorKind::Normal) {
                const double z = (value - first_) / second_;
                return -0.5 * z * z - std::log(second_) - 0.5 * std::log(2.0 * kPi);
            }
            if (kind_ == PriorKind::Gamma) {
                if (value <= 0.0) return NegativeInfinity();
                return (first_ - 1.0) * std::log(value)
                    - value / second_
                    - std::lgamma(first_)
                    - first_ * std::log(second_);
            }
            if (!(third_ < value && value < fourth_)) return NegativeInfinity();
            const double width = fourth_ - third_;
            const double u = (value - third_) / width;
            return (first_ - 1.0) * std::log(u)
                + (second_ - 1.0) * std::log1p(-u)
                - std::log(width)
                - std::lgamma(first_)
                - std::lgamma(second_)
                + std::lgamma(first_ + second_);
        }

    private:
        static constexpr double kPi = 3.141592653589793238462643383279502884;

        static double NegativeInfinity() {
            return -std::numeric_limits<double>::infinity();
        }

        PriorDistribution(
            const PriorKind kind,
            const double first,
            const double second,
            const double third = 0.0,
            const double fourth = 0.0
        )
            : kind_(kind), first_(first), second_(second), third_(third), fourth_(fourth) {}

        PriorKind kind_ = PriorKind::Normal;
        double first_ = 0.0;
        double second_ = 1.0;
        double third_ = 0.0;
        double fourth_ = 1.0;
};

} // namespace bayes_emc
