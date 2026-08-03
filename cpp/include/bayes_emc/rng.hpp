#pragma once

#include <cstdint>
#include <limits>
#include <random>

namespace bayes_emc {

class SplitMix64 {
    public:
        using result_type = std::uint64_t;

        explicit SplitMix64(const std::uint64_t seed) : state_(seed) {}

        std::uint64_t operator()() {
            std::uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            return z ^ (z >> 31);
        }

    private:
        std::uint64_t state_;
};

class Xoshiro256StarStar {
    public:
        using result_type = std::uint64_t;

        explicit Xoshiro256StarStar(const std::uint64_t seed = 5489ULL) {
            Seed(seed, 0);
        }

        Xoshiro256StarStar(const std::uint64_t seed, const std::uint64_t stream) {
            Seed(seed, stream);
        }

        static constexpr result_type min() {
            return std::numeric_limits<result_type>::min();
        }

        static constexpr result_type max() {
            return std::numeric_limits<result_type>::max();
        }

        void Seed(const std::uint64_t seed, const std::uint64_t stream) {
            SplitMix64 split(seed ^ (stream + 0x9e3779b97f4a7c15ULL));
            for (std::uint64_t & value : state_) {
                value = split();
            }
            if ((state_[0] | state_[1] | state_[2] | state_[3]) == 0) {
                state_[0] = 0x9e3779b97f4a7c15ULL;
            }
        }

        result_type operator()() {
            const std::uint64_t result = RotateLeft(state_[1] * 5ULL, 7) * 9ULL;
            const std::uint64_t t = state_[1] << 17;

            state_[2] ^= state_[0];
            state_[3] ^= state_[1];
            state_[1] ^= state_[2];
            state_[0] ^= state_[3];

            state_[2] ^= t;
            state_[3] = RotateLeft(state_[3], 45);

            return result;
        }

    private:
        static std::uint64_t RotateLeft(const std::uint64_t x, const int k) {
            return (x << k) | (x >> (64 - k));
        }

        std::uint64_t state_[4] = {};
};

#ifdef BAYES_EMC_USE_MT19937
using RandomEngine = std::mt19937;

inline RandomEngine MakeRandomEngine(const std::uint64_t seed, const std::uint64_t stream = 0) {
    std::seed_seq seed_sequence{
        static_cast<std::uint32_t>(seed),
        static_cast<std::uint32_t>(seed >> 32),
        static_cast<std::uint32_t>(stream),
        static_cast<std::uint32_t>(stream >> 32),
        0x9e3779b9U,
    };
    return RandomEngine(seed_sequence);
}

inline double UniformUnitDouble(RandomEngine & engine) {
    return std::generate_canonical<double, std::numeric_limits<double>::digits>(engine);
}
#else
using RandomEngine = Xoshiro256StarStar;

inline RandomEngine MakeRandomEngine(const std::uint64_t seed, const std::uint64_t stream = 0) {
    return RandomEngine(seed, stream);
}

inline double UniformUnitDouble(RandomEngine & engine) {
    constexpr double scale = 1.0 / 9007199254740992.0;
    return static_cast<double>(engine() >> 11) * scale;
}
#endif

} // namespace bayes_emc
