/// main.cpp
/// demonstration of simd_mc — price any barrier option configuration

#include <simd_mc/simd_mc.hpp>
#include <cstdio>
#include <cmath>
#include <chrono>

// BS for validation

static double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

static double bs_call(double S, double K, double r, double sigma, double T) {
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

static double bs_put(double S, double K, double r, double sigma, double T) {
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    return K * std::exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
}

template <typename Model, typename Barrier, typename Payoff>
static void run_and_print(
    const char* label,
    Model model, Barrier barrier, Payoff payoff,
    simd_mc::SimConfig config
) {
    using Clock = std::chrono::high_resolution_clock;

    auto t0 = Clock::now();
    auto result = simd_mc::price(model, barrier, payoff, config);
    auto t1 = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::printf("  Price:       %.4f (±%.4f)\n", result.price, result.std_error);
    std::printf("  Paths:       %d\n", result.paths_simulated);
    std::printf("  Lane util:   %.1f%%\n", result.lane_utilisation * 100.0);
    std::printf("  Time:        %.1f ms\n\n", ms);
}

int main() {
    using namespace simd_mc;

    std::printf("simd_mc — Barrier Option Pricing Demonstration\n");
    std::printf("SIMD width: %d lanes (float32)\n", simd_mc::simd_width);

    std::printf("Vanilla Options (no barrier)\n\n");
    {
        float S = 100.0f, K = 100.0f, r = 0.05f, vol = 0.20f, T = 1.0f;

        run_and_print("ATM Call (S=100, K=100)",
            models::GBM{.risk_free_rate = r, .volatility = vol},
            barriers::None{},
            payoffs::Call{.strike = K},
            SimConfig{.initial_spot = S, .n_paths = 500'000,
                      .n_steps = 252, .T = T, .risk_free_rate = r}
        );
        std::printf("  BS ref:      %.4f\n\n", bs_call(S, K, r, vol, T));

        run_and_print("ATM Put (S=100, K=100)",
            models::GBM{.risk_free_rate = r, .volatility = vol},
            barriers::None{},
            payoffs::Put{.strike = K},
            SimConfig{.initial_spot = S, .n_paths = 500'000,
                      .n_steps = 252, .T = T, .risk_free_rate = r}
        );
        std::printf("  BS ref:      %.4f\n\n", bs_put(S, K, r, vol, T));
    }

    std::printf("Down-and-Out Calls (varying barrier)\n\n");
    {
        float S = 100.0f, K = 100.0f, r = 0.05f, vol = 0.20f, T = 1.0f;
        float barriers[] = {70.0f, 80.0f, 90.0f, 95.0f, 98.0f};

        for (float B : barriers) {
            char label[64];
            std::snprintf(label, sizeof(label),
                          "Down-and-Out Call (S=100, K=100, B=%.0f)", B);

            run_and_print(label,
                models::GBM{.risk_free_rate = r, .volatility = vol},
                barriers::DownAndOut{.level = B},
                payoffs::Call{.strike = K},
                SimConfig{.initial_spot = S, .n_paths = 500'000,
                          .n_steps = 252, .T = T, .risk_free_rate = r}
            );
        }
    }

    std::printf("Up-and-Out Put\n\n");
    {
        float S = 100.0f, K = 105.0f, B = 120.0f;
        float r = 0.05f, vol = 0.25f, T = 0.5f;

        run_and_print("Up-and-Out Put (S=100, K=105, B=120, T=0.5y)",
            models::GBM{.risk_free_rate = r, .volatility = vol},
            barriers::UpAndOut{.level = B},
            payoffs::Put{.strike = K},
            SimConfig{.initial_spot = S, .n_paths = 500'000,
                      .n_steps = 126, .T = T, .risk_free_rate = r}
        );
    }

    std::printf("Brownian Bridge Correction\n\n");
    {
        float S = 100.0f, K = 100.0f, B = 85.0f;
        float r = 0.05f, vol = 0.20f, T = 1.0f;

        run_and_print("Hard barrier check (B=85)",
            models::GBM{.risk_free_rate = r, .volatility = vol},
            barriers::DownAndOut{.level = B},
            payoffs::Call{.strike = K},
            SimConfig{.initial_spot = S, .n_paths = 500'000,
                      .n_steps = 252, .T = T, .risk_free_rate = r}
        );

        run_and_print("Bridge correction (B=85)",
            models::GBM{.risk_free_rate = r, .volatility = vol},
            barriers::DownAndOutBridge{.level = B, .vol = vol},
            payoffs::Call{.strike = K},
            SimConfig{.initial_spot = S, .n_paths = 500'000,
                      .n_steps = 252, .T = T, .risk_free_rate = r}
        );

        std::printf("  Bridge should be lower (catches between-step crossings)\n\n");
    }

    std::printf("Knock-In via Parity\n\n");
    {
        float S = 100.0f, K = 100.0f, B = 85.0f;
        float r = 0.05f, vol = 0.20f, T = 1.0f;

        double vanilla_bs = bs_call(S, K, r, vol, T);

        auto ko_result = price(
            models::GBM{.risk_free_rate = r, .volatility = vol},
            barriers::DownAndOut{.level = B},
            payoffs::Call{.strike = K},
            SimConfig{.initial_spot = S, .n_paths = 500'000,
                      .n_steps = 252, .T = T, .risk_free_rate = r}
        );

        double ki_price = vanilla_bs - ko_result.price;

        std::printf("  Vanilla BS:       %.4f\n", vanilla_bs);
        std::printf("  Knock-Out MC:     %.4f (±%.4f)\n",
                    ko_result.price, ko_result.std_error);
        std::printf("  Knock-In (parity): %.4f\n\n", ki_price);
    }

    std::printf("Example Mkt Scenario\n\n");

    run_and_print("FTSE Down-and-Out Call (S=7500, K=7500, B=6750, vol=18%)",
        models::GBM{.risk_free_rate = 0.045f, .volatility = 0.18f},
        barriers::DownAndOut{.level = 6750.0f},
        payoffs::Call{.strike = 7500.0f},
        SimConfig{.initial_spot = 7500.0f, .n_paths = 500'000,
                  .n_steps = 252, .T = 1.0f, .risk_free_rate = 0.045f}
    );

    run_and_print("AAPL Down-and-Out Call (S=200, K=200, B=160, vol=40%)",
        models::GBM{.risk_free_rate = 0.05f, .volatility = 0.40f},
        barriers::DownAndOut{.level = 160.0f},
        payoffs::Call{.strike = 200.0f},
        SimConfig{.initial_spot = 200.0f, .n_paths = 500'000,
                  .n_steps = 252, .T = 1.0f, .risk_free_rate = 0.05f}
    );

    run_and_print("EURUSD Down-and-Out Call (S=1.25, K=1.25, B=1.20, vol=8%)",
        models::GBM{.risk_free_rate = 0.02f, .volatility = 0.08f},
        barriers::DownAndOutBridge{.level = 1.20f, .vol = 0.08f},
        payoffs::Call{.strike = 1.25f},
        SimConfig{.initial_spot = 1.25f, .n_paths = 500'000,
                  .n_steps = 63, .T = 0.25f, .risk_free_rate = 0.02f}
    );

    run_and_print("GLD Up-and-Out Put (S=80, K=85, B=110, vol=35%)",
        models::GBM{.risk_free_rate = 0.03f, .volatility = 0.35f},
        barriers::UpAndOut{.level = 110.0f},
        payoffs::Put{.strike = 85.0f},
        SimConfig{.initial_spot = 80.0f, .n_paths = 500'000,
                  .n_steps = 504, .T = 2.0f, .risk_free_rate = 0.03f}
    );

    std::printf("Convergence Study\n\n");
    {
        float S = 100.0f, K = 100.0f, B = 90.0f;
        float r = 0.05f, vol = 0.20f, T = 1.0f;

        std::printf("  Down-and-Out Call (S=100, K=100, B=90)\n");
        std::printf("  %-12s %-12s %-12s %-10s\n",
                    "Paths", "Price", "Std Error", "Time (ms)");

        int path_counts[] = {10'000, 50'000, 100'000, 500'000, 1'000'000};
        for (int np : path_counts) {
            using Clock = std::chrono::high_resolution_clock;
            auto t0 = Clock::now();
            auto result = price(
                models::GBM{.risk_free_rate = r, .volatility = vol},
                barriers::DownAndOut{.level = B},
                payoffs::Call{.strike = K},
                SimConfig{.initial_spot = S, .n_paths = np,
                          .n_steps = 252, .T = T, .risk_free_rate = r}
            );
            auto t1 = Clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            std::printf("  %-12d %-12.4f %-12.4f %-10.1f\n",
                        np, result.price, result.std_error, ms);
        }
        std::printf("\n  Std error should halve when paths quadruple (MC convergence rate)\n\n");
    }

    std::printf("All done.\n");
    return 0;
}
