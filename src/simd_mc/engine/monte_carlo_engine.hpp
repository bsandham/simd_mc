#pragma once
/// simd_mc/engine/monte_carlo_engine.hpp

#include "../core/simd_compat.hpp"
#include "../core/lane_register.hpp"
#include "../core/sim_config.hpp"
#include "../concepts/all.hpp"
#include <cmath>

namespace simd_mc {

template <
    typename V,
    typename Model,
    typename Barrier,
    typename PayoffFn,
    typename Refill,
    typename Compaction,
    typename RNG
>
    requires concepts::DiffusionModel<Model, V>
          && concepts::BarrierCondition<Barrier, V>
          && concepts::Payoff<PayoffFn, V>
          && concepts::RefillPolicy<Refill, V>
          && concepts::CompactionStrategy<Compaction, V>
          && concepts::RandomEngine<RNG, V>
class MonteCarloEngine {
    using Mask   = typename V::mask_type;
    using Scalar = typename V::value_type;
    static constexpr std::size_t W = V::size();

public:
    MonteCarloEngine(
        Model model, Barrier barrier, PayoffFn payoff,
        Refill refill, Compaction compaction, RNG rng
    ) : model_(model), barrier_(barrier), payoff_(payoff),
        refill_(refill), compaction_(compaction), rng_(rng) {}

    [[nodiscard]] SimulationResult run(
        int n_paths, int n_steps, float dt, float discount_factor
    ) {
        LaneRegister<V> lanes{};
        lanes.clear();
        Mask all_vacant(true);
        refill_.fill(lanes, all_vacant);
        lanes.alive = Mask(true);
        lanes.needs_refill = Mask(false);
        int    paths_done = 0;
        double welford_mean = 0.0;
        double welford_m2   = 0.0;
        long total_live_steps = 0;
        long total_steps      = 0;
        bool was_truncated    = false;

        while (paths_done < n_paths) {

            V Z = rng_.next_normal();
            V U = rng_.next_uniform();

            // negate Z for antithetic-tagged lanes
            Z = apply_antithetic(lanes, Z);

            V spot_before = lanes.spot;
            Mask was_alive = lanes.alive;
            lanes.spot = model_.evolve(lanes.spot, Z, dt);

            lanes.alive = barrier_.check(
                spot_before, lanes.spot, U, dt, lanes.alive
            );

            Mask steps_done = decrement_counters(lanes);

            Mask completed = lanes.alive && steps_done;
            if (any_of(completed)) {
                V payout = payoff_.evaluate(lanes.spot, lanes.survival_weight);

                for (std::size_t i = 0; i < W; ++i) {
                    if (completed[i]) {
                        double p = static_cast<double>(static_cast<Scalar>(payout[i]));
                        paths_done++;
                        double delta = p - welford_mean;
                        welford_mean += delta / paths_done;
                        double delta2 = p - welford_mean;
                        welford_m2 += delta * delta2;
                    }
                }

                lanes.alive = lanes.alive && (!completed);
            }

            // newly_dead = was alive before this step, but not alive after
            Mask newly_dead = was_alive && (!lanes.alive) && (!steps_done);
            for (std::size_t i = 0; i < W; ++i) {
                if (newly_dead[i]) {
                    paths_done++;
                    double delta = 0.0 - welford_mean;
                    welford_mean += delta / paths_done;
                    double delta2 = 0.0 - welford_mean;
                    welford_m2 += delta * delta2;
                }
            }
            stdx::where(float_mask_to_int(newly_dead), lanes.steps_remaining) = IntV(0);
            lanes.needs_refill = (!lanes.alive) || completed;

            total_live_steps += static_cast<long>(W - popcount(lanes.needs_refill));
            total_steps++;

            if (compaction_.maybe_compact(lanes)) {
                refill_.fill(lanes, lanes.needs_refill);

                for (std::size_t i = 0; i < W; ++i) {
                    if (lanes.needs_refill[i])
                        lanes.alive[i] = true;
                }
                lanes.needs_refill = Mask(false);
            }

            if (total_steps > static_cast<long>(n_paths) * n_steps * 2L
                              / static_cast<long>(W)) {
                was_truncated = true;
                break;
            }
        }

        double variance = (paths_done > 1)
            ? welford_m2 / (paths_done - 1)
            : 0.0;
        double std_err = (paths_done > 1)
            ? std::sqrt(variance / paths_done)
            : 0.0;
        double utilisation = (total_steps > 0)
            ? static_cast<double>(total_live_steps) / (total_steps * static_cast<long>(W))
            : 0.0;

        return SimulationResult{
            .price            = welford_mean * discount_factor,
            .std_error        = std_err * discount_factor,
            .paths_simulated  = paths_done,
            .lane_utilisation = utilisation,
            .truncated        = was_truncated,
            .welford_mean     = welford_mean,
            .welford_m2       = welford_m2
        };
    }

private:
    Model      model_;
    Barrier    barrier_;
    PayoffFn   payoff_;
    Refill     refill_;
    Compaction compaction_;
    RNG        rng_;

    /// negate Z for antithetic lanes
    static V apply_antithetic(const LaneRegister<V>& lanes, V Z) {
        Mask is_anti;
        for (std::size_t i = 0; i < W; ++i)
            is_anti[i] = (lanes.role[i] == LaneRole::Antithetic);
        return select(is_anti, -Z, Z);
    }

    /// decrement step counters — 3 SIMD instructions:
    /// vpcmpd  (steps > 0 > k mask)
    /// vpsubd  (steps - 1, masked)
    /// vpcmpeqd (steps == 0 > done mask)
    static Mask decrement_counters(LaneRegister<V>& lanes) {
        IntV zero(0);
        IntV one(1);
        auto positive = lanes.steps_remaining > zero;                   // vpcmpd
        stdx::where(positive, lanes.steps_remaining) =
            lanes.steps_remaining - one;                                // vpsubd{k}
        IntMaskV done_i = (lanes.steps_remaining == zero);              // vpcmpeqd
        return int_mask_to_float(done_i);                               // bit_cast (no-op)
    }
};

} // namespace simd_mc
