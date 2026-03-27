#pragma once
/// simd_mc/core/sim_config.hpp

#include <cmath>
#include <thread>

namespace simd_mc {

struct SimConfig {
    float initial_spot   = 100.0f;
    int   n_paths        = 100'000;
    int   n_steps        = 252;
    float T              = 1.0f;
    float risk_free_rate = 0.05f;
    int   n_threads      = 0;

    [[nodiscard]] float dt()              const { return T / static_cast<float>(n_steps); }
    [[nodiscard]] float discount_factor() const { return std::exp(-risk_free_rate * T); }
    [[nodiscard]] int   thread_count()    const {
        if (n_threads > 0) return n_threads;
        int hw = static_cast<int>(std::thread::hardware_concurrency());
        return hw > 0 ? hw : 1;
    }
};

struct SimulationResult {
    double price            = 0.0;
    double std_error        = 0.0;
    int    paths_simulated  = 0;
    double lane_utilisation = 0.0;
    bool   truncated        = false;

    double welford_mean = 0.0;
    double welford_m2   = 0.0;
};

/// merge two Welford partial results (Chan's parallel algorithm) > numerically stable combination of two independent Welford accumulators
inline SimulationResult welford_merge(const SimulationResult& a, const SimulationResult& b) {
    if (a.paths_simulated == 0) return b;
    if (b.paths_simulated == 0) return a;

    int    n_ab = a.paths_simulated + b.paths_simulated;
    double delta = b.welford_mean - a.welford_mean;
    double mean_ab = a.welford_mean + delta * b.paths_simulated / n_ab;
    double m2_ab = a.welford_m2 + b.welford_m2
                 + delta * delta * static_cast<double>(a.paths_simulated)
                   * static_cast<double>(b.paths_simulated) / n_ab;

    double variance = (n_ab > 1) ? m2_ab / (n_ab - 1) : 0.0;
    double std_err  = (n_ab > 1) ? std::sqrt(variance / n_ab) : 0.0;

    double total_steps = a.lane_utilisation * a.paths_simulated
                       + b.lane_utilisation * b.paths_simulated;
    double util = (n_ab > 0) ? total_steps / n_ab : 0.0;

    double discount = 1.0;  // already applied in each partial result, the partial results already have discount_factor applied, so mean_ab already reflects discounted prices

    return SimulationResult{
        .price            = mean_ab,
        .std_error        = std_err * std::abs(a.price > 0 ? a.price / a.welford_mean : 1.0),
        .paths_simulated  = n_ab,
        .lane_utilisation = util,
        .truncated        = a.truncated || b.truncated,
        .welford_mean     = mean_ab,
        .welford_m2       = m2_ab
    };
}

} // namespace simd_mc
