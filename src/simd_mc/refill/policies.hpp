#pragma once
/// simd_mc/refill/policies.hpp

#include "../core/lane_register.hpp"
#include <cstdint>

namespace simd_mc::refill {

struct Independent {
    float    initial_spot = 100.0f;
    int      total_steps  = 252;
    uint64_t next_path_id = 0;

    template <typename V>
    void fill(LaneRegister<V>& lanes, typename V::mask_type vacated) {
        for (std::size_t i = 0; i < V::size(); ++i) {
            if (!vacated[i]) continue;
            lanes.spot[i]            = initial_spot;
            lanes.steps_remaining[i] = total_steps;
            lanes.path_id[i]         = next_path_id++;
            lanes.partner_lane[i]    = -1;
            lanes.role[i]            = LaneRole::Independent;
            lanes.survival_weight[i] = 1.0f;
        }
    }
};

struct AntitheticPreferred {
    float    initial_spot = 100.0f;
    int      total_steps  = 252;
    uint64_t next_path_id = 0;

    template <typename V>
    void fill(LaneRegister<V>& lanes, typename V::mask_type vacated) {
        for (std::size_t i = 0; i < V::size(); ++i) {
            if (!vacated[i]) continue;

            int partner = find_available_partner<V>(lanes, i);

            if (partner >= 0) {
                // antithetic starts at S₀ with full path length, this gives partial anticorrelation (only for the overlapping future steps), but doesn't introduce barrier bias
                // phase correction (cloning partner's spot) is biased for barrier options because it skips early barrier checks
                lanes.spot[i]            = initial_spot;
                lanes.steps_remaining[i] = total_steps;
                lanes.path_id[i]         = next_path_id++;
                lanes.partner_lane[i]    = partner;
                lanes.role[i]            = LaneRole::Antithetic;
                lanes.survival_weight[i] = 1.0f;

                lanes.partner_lane[partner] = static_cast<int>(i);
                lanes.role[partner]         = LaneRole::HasPartner;
            } else {
                lanes.spot[i]            = initial_spot;
                lanes.steps_remaining[i] = total_steps;
                lanes.path_id[i]         = next_path_id++;
                lanes.partner_lane[i]    = -1;
                lanes.role[i]            = LaneRole::Independent;
                lanes.survival_weight[i] = 1.0f;
            }
        }
    }

private:
    template <typename V>
    static int find_available_partner(const LaneRegister<V>& lanes, std::size_t exclude) {
        for (std::size_t j = 0; j < V::size(); ++j) {
            if (j == exclude) continue;
            if (lanes.role[j] == LaneRole::Independent &&
                lanes.steps_remaining[j] > 0) {
                return static_cast<int>(j);
            }
        }
        return -1;
    }
};

} // namespace simd_mc::refill
