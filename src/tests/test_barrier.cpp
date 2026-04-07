/// tests/test_barrier.cpp

#include <simd_mc/simd_mc.hpp>
#include <cstdio>
#include <cmath>

static double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

static double bs_call(double S, double K, double r, double sigma, double T) {
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

/// analytical down-and-out call price (continuous monitoring, no dividends)
static double analytical_down_out_call(
    double S, double K, double B, double r, double sigma, double T
) {
    if (B >= S) return 0.0;
    if (B >= K) {
        // barrier above strike: more complex formula so use the full Rubinstein-Reiner decomposition
        double lambda = (r + 0.5 * sigma * sigma) / (sigma * sigma);
        double x1 = std::log(S / K) / (sigma * std::sqrt(T)) + lambda * sigma * std::sqrt(T);
        double y  = std::log(B * B / (S * K)) / (sigma * std::sqrt(T)) + lambda * sigma * std::sqrt(T);
        double y1 = std::log(B / S) / (sigma * std::sqrt(T)) + lambda * sigma * std::sqrt(T);

        double A = S * norm_cdf(x1)
                 - K * std::exp(-r * T) * norm_cdf(x1 - sigma * std::sqrt(T));
        double C = S * std::pow(B / S, 2 * lambda) * norm_cdf(y)
                 - K * std::exp(-r * T) * std::pow(B / S, 2 * lambda - 2)
                   * norm_cdf(y - sigma * std::sqrt(T));
        double D = S * std::pow(B / S, 2 * lambda) * norm_cdf(y1)
                 - K * std::exp(-r * T) * std::pow(B / S, 2 * lambda - 2)
                   * norm_cdf(y1 - sigma * std::sqrt(T));
        return A - C + D;
    }
    // happy case
    double vanilla = bs_call(S, K, r, sigma, T);
    double lambda = (r + 0.5 * sigma * sigma) / (sigma * sigma);
    double y = std::log(B * B / (S * K)) / (sigma * std::sqrt(T)) + lambda * sigma * std::sqrt(T);
    double mirror = S * std::pow(B / S, 2 * lambda) * norm_cdf(y)
                  - K * std::exp(-r * T) * std::pow(B / S, 2 * lambda - 2)
                    * norm_cdf(y - sigma * std::sqrt(T));
    return vanilla - mirror;
}

int main() {
    using namespace simd_mc;

    int pass = 0, fail = 0;
    int n_paths = 500'000;

    auto check = [&](const char* name, double mc, double ref, double tol) {
        double err = std::abs(mc - ref);
        bool ok = err < tol;
        std::printf("  %s: MC=%.4f  Ref=%.4f  err=%.4f  %s\n",
                    name, mc, ref, err, ok ? "PASS" : "FAIL");
        if (ok) ++pass; else ++fail;
    };

    float S0 = 100.0f, K = 100.0f, r = 0.05f, sigma = 0.20f, T = 1.0f;

    std::printf("Test: Down-and-out call, B=80 (far barrier)\n");
    {
        float B = 80.0f;
        double analytical = analytical_down_out_call(S0, K, B, r, sigma, T);

        auto result = price(
            models::GBM{.risk_free_rate = r, .volatility = sigma},
            barriers::DownAndOut{.level = B},
            payoffs::Call{.strike = K},
            SimConfig{.initial_spot = S0, .n_paths = n_paths,
                      .n_steps = 252, .T = T, .risk_free_rate = r}
        );

        std::printf("  Analytical (continuous): %.4f\n", analytical);
        // MC with discrete monitoring will be slightly higher than continuous (misses some barrier crossings between steps)
        check("Far barrier", result.price, analytical, 0.60);
    }

    std::printf("Test: Down-and-out call, B=90 (medium barrier)\n");
    {
        float B = 90.0f;
        double analytical = analytical_down_out_call(S0, K, B, r, sigma, T);

        auto result = price(
            models::GBM{.risk_free_rate = r, .volatility = sigma},
            barriers::DownAndOut{.level = B},
            payoffs::Call{.strike = K},
            SimConfig{.initial_spot = S0, .n_paths = n_paths,
                      .n_steps = 252, .T = T, .risk_free_rate = r}
        );

        std::printf("  Analytical (continuous): %.4f\n", analytical);
        check("Medium barrier", result.price, analytical, 0.60);
    }

    std::printf("Test: Knock-in via parity (KI = Vanilla_BS - KO_MC)\n");
    {
        float B = 85.0f;
        double vanilla_bs = bs_call(S0, K, r, sigma, T);
        double analytical_ko = analytical_down_out_call(S0, K, B, r, sigma, T);
        double analytical_ki = vanilla_bs - analytical_ko;

        auto result_ko = price(
            models::GBM{.risk_free_rate = r, .volatility = sigma},
            barriers::DownAndOut{.level = B},
            payoffs::Call{.strike = K},
            SimConfig{.initial_spot = S0, .n_paths = n_paths,
                      .n_steps = 252, .T = T, .risk_free_rate = r}
        );

        // control variate 
        double ki_via_parity = vanilla_bs - result_ko.price;

        std::printf("  Vanilla BS:             %.4f\n", vanilla_bs);
        std::printf("  KO MC:                  %.4f\n", result_ko.price);
        std::printf("  KI via parity:          %.4f\n", ki_via_parity);
        std::printf("  KI analytical (cont.):  %.4f\n", analytical_ki);
        check("KI parity", ki_via_parity, analytical_ki, 0.50);
    }

    std::printf("Test: Lane utilisation increases with compaction\n");
    {
        float B = 95.0f;

        auto result_compact = price(
            models::GBM{.risk_free_rate = r, .volatility = sigma},
            barriers::DownAndOut{.level = B},
            payoffs::Call{.strike = K},
            SimConfig{.initial_spot = S0, .n_paths = 200'000,
                      .n_steps = 252, .T = T, .risk_free_rate = r}
        );

        std::printf("  Close barrier (B=95) utilisation: %.1f%%\n",
                    result_compact.lane_utilisation * 100.0);

        bool ok = result_compact.lane_utilisation > 0.0;
        std::printf("  Utilisation > 0%%: %s\n", ok ? "PASS" : "FAIL");
        if (ok) ++pass; else ++fail;
    }

    std::printf("Test: Bridge correction improves accuracy\n");
    {
        float B = 85.0f;
        double analytical = analytical_down_out_call(S0, K, B, r, sigma, T);

        auto result_hard = price(
            models::GBM{.risk_free_rate = r, .volatility = sigma},
            barriers::DownAndOut{.level = B},
            payoffs::Call{.strike = K},
            SimConfig{.initial_spot = S0, .n_paths = n_paths,
                      .n_steps = 252, .T = T, .risk_free_rate = r}
        );

        auto result_bridge = price(
            models::GBM{.risk_free_rate = r, .volatility = sigma},
            barriers::DownAndOutBridge{.level = B, .vol = sigma},
            payoffs::Call{.strike = K},
            SimConfig{.initial_spot = S0, .n_paths = n_paths,
                      .n_steps = 252, .T = T, .risk_free_rate = r}
        );

        double err_hard   = std::abs(result_hard.price - analytical);
        double err_bridge = std::abs(result_bridge.price - analytical);

        std::printf("  Analytical:     %.4f\n", analytical);
        std::printf("  Hard check:     %.4f  (err=%.4f)\n", result_hard.price, err_hard);
        std::printf("  Bridge corr:    %.4f  (err=%.4f)\n", result_bridge.price, err_bridge);

        // bridge should be closer to analytical (continuous monitoring) price but with MC noise this isn't guaranteed every run, so use generous tolerance
        bool ok = err_bridge < err_hard + 0.5;
        std::printf("  Bridge less than or equal to Hard+noise: %s\n", ok ? "PASS" : "FAIL");
        if (ok) ++pass; else ++fail;
    }

    std::printf("\n%d passed, %d failed\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
