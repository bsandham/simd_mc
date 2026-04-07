/// examples/benchmark_crossover.cpp
/// sweep barrier proximity to find the compaction crossover point

#include <simd_mc/simd_mc.hpp>
#include <cstdio>
#include <chrono>

int main() {
    using namespace simd_mc;
    using Clock = std::chrono::high_resolution_clock;

    std::printf("simd_mc — Barrier Proximity Crossover Benchmark\n");
    std::printf("SIMD width: %d lanes\n\n", simd_mc::simd_width);

    float S0    = 100.0f;
    float K     = 100.0f;
    float r     = 0.05f;
    float sigma = 0.20f;
    float T     = 1.0f;
    int   n_paths = 200'000;
    int   n_steps = 252;

    float barriers[] = {70.0f, 75.0f, 80.0f, 85.0f, 90.0f, 92.0f, 95.0f, 97.0f, 98.0f};

    std::printf("%-10s %-12s %-12s %-10s %-10s\n",
                "Barrier", "Price", "Std Error", "Util %", "Time (ms)");

    for (float B : barriers) {
        auto config = SimConfig{
            .initial_spot   = S0,
            .n_paths        = n_paths,
            .n_steps        = n_steps,
            .T              = T,
            .risk_free_rate = r
        };

        auto t0 = Clock::now();
        auto result = price(
            models::GBM{.risk_free_rate = r, .volatility = sigma},
            barriers::DownAndOut{.level = B},
            payoffs::Call{.strike = K},
            config
        );
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::printf("%-10.0f %-12.4f %-12.4f %-10.1f %-10.1f\n",
                    B, result.price, result.std_error,
                    result.lane_utilisation * 100.0, ms);
    }

    std::printf("\nNote: Lane utilisation decreases with closer barriers because\n"
                "more paths are knocked out. The stream-compacting engine\n"
                "refills dead lanes, but the compaction itself takes cycles.\n");
    return 0;
}
