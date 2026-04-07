/// examples/basic_barrier.cpp

#include <simd_mc/simd_mc.hpp>
#include <cstdio>
#include <cmath>

/// BS call price for validation
static double bs_call(double S, double K, double r, double sigma, double T) {
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    auto N = [](double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); };
    return S * N(d1) - K * std::exp(-r * T) * N(d2);
}

int main() {
    using namespace simd_mc;

    std::printf("simd_mc - Stream-Compacting Barrier Option Monte Carlo\n");
    std::printf("SIMD width: %d lanes (float32)\n\n", simd_mc::simd_width);

    float S0    = 100.0f;
    float K     = 100.0f;
    float B     = 80.0f;
    float r     = 0.05f;
    float sigma = 0.20f;
    float T     = 1.0f;
    int   n_paths = 500'000;
    int   n_steps = 252;

    // 1. Vanilla European call (no barrier) — validate against BS
    {
        auto result = price(
            models::GBM{.risk_free_rate = r, .volatility = sigma},
            barriers::None{},
            payoffs::Call{.strike = K},
            SimConfig{.initial_spot = S0, .n_paths = n_paths, .n_steps = n_steps,
                      .T = T, .risk_free_rate = r}
        );

        double bs = bs_call(S0, K, r, sigma, T);

        std::printf("Vanilla European Call\n");
        std::printf("  Black-Scholes:  %.4f\n", bs);
        std::printf("  Monte Carlo:    %.4f (±%.4f)\n", result.price, result.std_error);
        std::printf("  Paths:          %d\n", result.paths_simulated);
        std::printf("  Lane util:      %.1f%%\n\n", result.lane_utilisation * 100);
    }

    // 2. Down-and-out call (hard barrier)
    {
        auto result = price(
            models::GBM{.risk_free_rate = r, .volatility = sigma},
            barriers::DownAndOut{.level = B},
            payoffs::Call{.strike = K},
            SimConfig{.initial_spot = S0, .n_paths = n_paths, .n_steps = n_steps,
                      .T = T, .risk_free_rate = r}
        );

        std::printf("Down-and-Out Call (barrier=%.0f, hard check)\n", B);
        std::printf("  Monte Carlo:    %.4f (±%.4f)\n", result.price, result.std_error);
        std::printf("  Paths:          %d\n", result.paths_simulated);
        std::printf("  Lane util:      %.1f%%\n\n", result.lane_utilisation * 100);
    }

    // 3. Down-and-out call (Brownian bridge correction)
    {
        auto result = price(
            models::GBM{.risk_free_rate = r, .volatility = sigma},
            barriers::DownAndOutBridge{.level = B, .vol = sigma},
            payoffs::Call{.strike = K},
            SimConfig{.initial_spot = S0, .n_paths = n_paths, .n_steps = n_steps,
                      .T = T, .risk_free_rate = r}
        );

        std::printf("Down-and-Out Call (barrier=%.0f, bridge correction)\n", B);
        std::printf("  Monte Carlo:    %.4f (±%.4f)\n", result.price, result.std_error);
        std::printf("  Paths:          %d\n", result.paths_simulated);
        std::printf("  Lane util:      %.1f%%\n\n", result.lane_utilisation * 100);
    }

    // 4. Close barrier — demonstrates higher compaction rate
    {
        float B_close = 95.0f;
        auto result = price(
            models::GBM{.risk_free_rate = r, .volatility = sigma},
            barriers::DownAndOut{.level = B_close},
            payoffs::Call{.strike = K},
            SimConfig{.initial_spot = S0, .n_paths = n_paths, .n_steps = n_steps,
                      .T = T, .risk_free_rate = r}
        );

        std::printf("Down-and-Out Call (barrier=%.0f, close barrier)\n", B_close);
        std::printf("  Monte Carlo:    %.4f (±%.4f)\n", result.price, result.std_error);
        std::printf("  Paths:          %d\n", result.paths_simulated);
        std::printf("  Lane util:      %.1f%%\n", result.lane_utilisation * 100);
        std::printf("  (Lower util expected as more knock-outs > more compaction)\n\n");
    }

    return 0;
}
