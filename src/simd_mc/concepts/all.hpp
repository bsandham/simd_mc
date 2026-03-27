#pragma once
/// simd_mc/concepts/all.hpp

#include "../core/simd_compat.hpp"
#include "../core/lane_register.hpp"
#include <concepts>

namespace simd_mc::concepts {

template <typename M, typename V>
concept DiffusionModel = requires(M model, V spot, V normals, typename V::value_type dt) {
    { model.evolve(spot, normals, dt) } -> std::same_as<V>;
};

template <typename B, typename V>
concept BarrierCondition = requires(
    B barrier, V spot_before, V spot_after, V uniforms,
    typename V::value_type dt, typename V::mask_type prior_alive
) {
    { barrier.check(spot_before, spot_after, uniforms, dt, prior_alive) }
        -> std::same_as<typename V::mask_type>;
};

template <typename P, typename V>
concept Payoff = requires(P payoff, V terminal_spot, V survival_weight) {
    { payoff.evaluate(terminal_spot, survival_weight) } -> std::same_as<V>;
};

template <typename R, typename V>
concept RefillPolicy = requires(R policy, LaneRegister<V>& lanes, typename V::mask_type vacated) {
    { policy.fill(lanes, vacated) } -> std::same_as<void>;
};

template <typename C, typename V>
concept CompactionStrategy = requires(C strategy, LaneRegister<V>& lanes) {
    { strategy.maybe_compact(lanes) } -> std::same_as<bool>;
};

template <typename E, typename V>
concept RandomEngine = requires(E engine) {
    { engine.next_normal()  } -> std::same_as<V>;
    { engine.next_uniform() } -> std::same_as<V>;
};

} // namespace simd_mc::concepts
