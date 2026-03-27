#pragma once
/// simd_mc/core/lane_register.hpp

#include "simd_compat.hpp"
#include <array>
#include <cstdint>

namespace simd_mc {

enum class LaneRole : uint8_t {
    Independent,
    HasPartner,
    Antithetic,
    Completed,
    Dead
};

/// template V is a stdx::simd type
template <typename V>
struct LaneRegister {
    using value_type = typename V::value_type;
    using mask_type  = typename V::mask_type;
    static constexpr std::size_t width = V::size();

    // hot
    V         spot;
    V         survival_weight;
    IntV      steps_remaining;           // vpaddd for decrement, vpcmpeqd for done check

    // warm
    std::array<uint64_t, width> path_id{};
    std::array<int,      width> partner_lane{};
    std::array<LaneRole, width> role{};

    mask_type alive;
    mask_type needs_refill;

    void clear() {
        spot = V(value_type(0));
        survival_weight = V(value_type(1));
        steps_remaining = IntV(0);
        alive = mask_type(true);
        needs_refill = mask_type(false);
        path_id.fill(0);
        partner_lane.fill(-1);
        role.fill(LaneRole::Independent);
    }
};

} // namespace simd_mc
