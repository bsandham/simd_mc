/// tests/test_vanilla.cpp

#include <simd_mc/simd_mc.hpp>
#include <cstdio>
#include <cmath>
#include <cassert>

static double bs_call(double S, double K, double r, double sigma, double T) {
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    auto N = [](double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); };
    return S * N(d1) - K * std::exp(-r * T) * N(d2);
}

static double bs_put(double S, double K, double r, double sigma, double T) {
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    auto N = [](double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); };
    return K * std::exp(-r * T) * N(-d2) - S * N(-d1);
}

int main() {
    using namespace simd_mc;

    int pass = 0, fail = 0;

    auto check = [&](const char* name, double mc, double analytical, double tol) {
        double err = std::abs(mc - analytical);
        bool ok = err < tol;
        std::printf("  %s: MC=%.4f  BS=%.4f  err=%.4f  %s\n",
                    name, mc, analytical, err, ok ? "PASS" : "FAIL");
        if (ok) ++pass; else ++fail;
    };

    std::printf("Test: Vanilla European Call convergence\n");
    {
        auto result = price(
            models::GBM{.risk_free_rate = 0.05f, .volatility = 0.20f},
            barriers::None{},
            payoffs::Call{.strike = 100.0f},
            SimConfig{.initial_spot = 100.0f, .n_paths = 500'000,
                      .n_steps = 100, .T = 1.0f, .risk_free_rate = 0.05f}
        );
        double bs = bs_call(100, 100, 0.05, 0.20, 1.0);
        check("ATM call", result.price, bs, 0.20);  // ~0.2 tolerance for 500k paths
    }

    std::printf("Test: Vanilla European Put convergence\n");
    {
        auto result = price(
            models::GBM{.risk_free_rate = 0.05f, .volatility = 0.20f},
            barriers::None{},
            payoffs::Put{.strike = 100.0f},
            SimConfig{.initial_spot = 100.0f, .n_paths = 500'000,
                      .n_steps = 100, .T = 1.0f, .risk_free_rate = 0.05f}
        );
        double bs = bs_put(100, 100, 0.05, 0.20, 1.0);
        check("ATM put", result.price, bs, 0.20);
    }

    std::printf("Test: ITM call\n");
    {
        auto result = price(
            models::GBM{.risk_free_rate = 0.05f, .volatility = 0.30f},
            barriers::None{},
            payoffs::Call{.strike = 90.0f},
            SimConfig{.initial_spot = 100.0f, .n_paths = 500'000,
                      .n_steps = 100, .T = 0.5f, .risk_free_rate = 0.05f}
        );
        double bs = bs_call(100, 90, 0.05, 0.30, 0.5);
        check("ITM call", result.price, bs, 0.25);
    }

    std::printf("Test: OTM call\n");
    {
        auto result = price(
            models::GBM{.risk_free_rate = 0.05f, .volatility = 0.30f},
            barriers::None{},
            payoffs::Call{.strike = 110.0f},
            SimConfig{.initial_spot = 100.0f, .n_paths = 500'000,
                      .n_steps = 100, .T = 0.5f, .risk_free_rate = 0.05f}
        );
        double bs = bs_call(100, 110, 0.05, 0.30, 0.5);
        check("OTM call", result.price, bs, 0.20);
    }

    std::printf("\n%d passed, %d failed\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
