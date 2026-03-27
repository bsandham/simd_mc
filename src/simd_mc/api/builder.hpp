#pragma once
/// simd_mc/api/builder.hpp
/// splits n_paths across hardware threads, each running an independent
/// engine with a different RNG seed. results are merged via Chan's parallel Welford algorithm for numerically stable variance combination

#include "../core/simd_compat.hpp"
#include "../core/sim_config.hpp"
#include "../engine/monte_carlo_engine.hpp"
#include "../refill/policies.hpp"
#include "../compaction/strategies.hpp"
#include "../rng/philox.hpp"
#include <algorithm>
#include <vector>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace simd_mc {

template <typename Model, typename Barrier, typename PayoffFn>
SimulationResult price(
    Model    model,
    Barrier  barrier,
    PayoffFn payoff,
    SimConfig config
) {
    using V = FloatV;

    model.set_timestep(config.dt());

    int n_threads = config.thread_count();
    int paths_per_thread = config.n_paths / n_threads;
    int remainder = config.n_paths % n_threads;

    float dt = config.dt();
    float discount = config.discount_factor();

    std::vector<SimulationResult> partial(n_threads);

    #ifdef _OPENMP
    #pragma omp parallel num_threads(n_threads)
    #endif
    {
        #ifdef _OPENMP
        int tid = omp_get_thread_num();
        #else
        int tid = 0;
        #endif

        int my_paths = paths_per_thread + (tid < remainder ? 1 : 0);

        refill::AntitheticPreferred refill_policy{
            .initial_spot = config.initial_spot,
            .total_steps  = config.n_steps,
        };

        compaction::Adaptive compact_strategy{
            .threshold = std::max(1, static_cast<int>(V::size()) / 4)
        };

        rng::Philox<V> random_engine(static_cast<uint32_t>(42 + tid * 1000003));

        auto engine = MonteCarloEngine<
            V, Model, Barrier, PayoffFn,
            refill::AntitheticPreferred,
            compaction::Adaptive,
            rng::Philox<V>
        >(model, barrier, payoff, refill_policy, compact_strategy, random_engine);

        partial[tid] = engine.run(my_paths, config.n_steps, dt, discount);
    }

    // all partial results have undiscounted welford_mean/m2 and discounted price/std_error. we merge the undiscounted stats
    double merged_mean = 0.0;
    double merged_m2   = 0.0;
    int    merged_n    = 0;
    double total_util  = 0.0;
    bool   any_trunc   = false;

    for (int t = 0; t < n_threads; ++t) {
        const auto& p = partial[t];
        if (p.paths_simulated == 0) continue;

        int n_a = merged_n;
        int n_b = p.paths_simulated;
        int n_ab = n_a + n_b;

        if (n_a == 0) {
            merged_mean = p.welford_mean;
            merged_m2   = p.welford_m2;
        } else {
            double delta = p.welford_mean - merged_mean;
            merged_mean += delta * n_b / n_ab;
            merged_m2  += p.welford_m2
                        + delta * delta * static_cast<double>(n_a)
                          * static_cast<double>(n_b) / n_ab;
        }

        merged_n   = n_ab;
        total_util += p.lane_utilisation * p.paths_simulated;
        any_trunc   = any_trunc || p.truncated;
    }

    double variance = (merged_n > 1) ? merged_m2 / (merged_n - 1) : 0.0;
    double std_err  = (merged_n > 1) ? std::sqrt(variance / merged_n) : 0.0;
    double util     = (merged_n > 0) ? total_util / merged_n : 0.0;

    return SimulationResult{
        .price            = merged_mean * discount,
        .std_error        = std_err * discount,
        .paths_simulated  = merged_n,
        .lane_utilisation = util,
        .truncated        = any_trunc,
        .welford_mean     = merged_mean,
        .welford_m2       = merged_m2
    };
}

} // namespace simd_mc
