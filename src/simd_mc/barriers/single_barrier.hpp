#pragma once
/// simd_mc/barriers/single_barrier.hpp

#include "../core/simd_compat.hpp"

namespace simd_mc::barriers {

enum class Direction { Down, Up };

template <Direction Dir, bool UseBridgeCorrection = false>
struct SingleBarrier {
    float level = 0.0f;
    float vol   = 0.0f;

    template <typename V>
    [[nodiscard]] auto check(
        V spot_before, V spot_after, V uniforms,
        typename V::value_type dt,
        typename V::mask_type prior_alive
    ) const -> typename V::mask_type {

        using Mask = typename V::mask_type;

        // compiles to vcmpps + mask register op
        Mask survived;
        if constexpr (Dir == Direction::Down) {
            survived = spot_after > V(level);
        } else {
            survived = spot_after < V(level);
        }

        // brownian bridge crossing probability (compiled away when false)
        if constexpr (UseBridgeCorrection) {
            V barrier_v = V(level);
            V log_before = simd_log(spot_before / barrier_v);
            V log_after  = simd_log(spot_after  / barrier_v);
            V sigma_sq_dt = V(vol * vol * dt);

            V cross_prob = simd_exp(
                V(-2.0f) * log_before * log_after / sigma_sq_dt
            );

            Mask bridge_kill = uniforms < cross_prob;
            survived = survived && !bridge_kill;
        }

        return prior_alive && survived;
    }
};

using DownAndOut       = SingleBarrier<Direction::Down, false>;
using UpAndOut         = SingleBarrier<Direction::Up,   false>;
using DownAndOutBridge = SingleBarrier<Direction::Down, true>;
using UpAndOutBridge   = SingleBarrier<Direction::Up,   true>;

} // namespace simd_mc::barriers
