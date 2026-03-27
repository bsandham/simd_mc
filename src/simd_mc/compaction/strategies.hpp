#pragma once
/// simd_mc/compaction/strategies.hpp
/// EveryStep and Adaptive compaction strategies.
///
/// on avx-512 uses VCOMPRESSPS / VPCOMPRESSD for hot registers. else portable scalar shuffle.

#include "../core/lane_register.hpp"
#include "../core/simd_compat.hpp"

#ifdef __AVX512F__
#include <immintrin.h>
#endif

namespace simd_mc::compaction {

namespace detail {

template <typename V>
void compact_lanes_portable(LaneRegister<V>& lanes) {
    using T = typename V::value_type;
    constexpr std::size_t W = V::size();

    std::array<T,        W> new_spot{};
    std::array<T,        W> new_weight{};
    std::array<int32_t,  W> new_steps{};
    std::array<uint64_t, W> new_path_id{};
    std::array<int,      W> new_partner{};
    std::array<LaneRole, W> new_role{};

    std::size_t write = 0;
    for (std::size_t read = 0; read < W; ++read) {
        if (!lanes.needs_refill[read]) {
            new_spot[write]    = static_cast<T>(lanes.spot[read]);
            new_weight[write]  = static_cast<T>(lanes.survival_weight[read]);
            new_steps[write]   = static_cast<int32_t>(lanes.steps_remaining[read]);
            new_path_id[write] = lanes.path_id[read];
            new_partner[write] = -1;
            new_role[write]    = (lanes.role[read] == LaneRole::HasPartner)
                                    ? LaneRole::Independent
                                    : lanes.role[read];
            ++write;
        }
    }

    std::size_t n_live = write;
    for (std::size_t i = n_live; i < W; ++i) {
        new_spot[i] = T(0); new_weight[i] = T(1); new_steps[i] = 0;
        new_path_id[i] = 0; new_partner[i] = -1; new_role[i] = LaneRole::Dead;
    }

    for (std::size_t i = 0; i < W; ++i) {
        lanes.spot[i] = new_spot[i]; lanes.survival_weight[i] = new_weight[i];
        lanes.steps_remaining[i] = new_steps[i]; lanes.path_id[i] = new_path_id[i];
        lanes.partner_lane[i] = new_partner[i]; lanes.role[i] = new_role[i];
    }

    for (std::size_t i = 0; i < W; ++i) {
        lanes.needs_refill[i] = (i >= n_live);
        lanes.alive[i]        = (i < n_live);
    }
}

#ifdef __AVX512F__
/// avx-512 hardware compaction for 16wide vectors
/// VCOMPRESSPS for spot/weight, VPCOMPRESSD for steps_remaining
template <typename V>
void compact_lanes_avx512(LaneRegister<V>& lanes) {
    constexpr std::size_t W = V::size();

    uint16_t live_bits = 0;
    for (std::size_t i = 0; i < W; ++i)
        if (!lanes.needs_refill[i]) live_bits |= (1u << i);

    __mmask16 live_mask = _cvtu32_mask16(live_bits);
    int n_live = __builtin_popcount(live_bits);

    // VCOMPRESSPS: spot
    __m512 spot_raw = __builtin_bit_cast(__m512, lanes.spot);
    alignas(64) float spot_buf[16];
    _mm512_mask_compressstoreu_ps(spot_buf, live_mask, spot_raw);
    for (int i = n_live; i < 16; ++i) spot_buf[i] = 0.0f;
    lanes.spot = __builtin_bit_cast(V, _mm512_load_ps(spot_buf));

    // VCOMPRESSPS: survival_weight
    __m512 wt_raw = __builtin_bit_cast(__m512, lanes.survival_weight);
    alignas(64) float wt_buf[16];
    _mm512_mask_compressstoreu_ps(wt_buf, live_mask, wt_raw);
    for (int i = n_live; i < 16; ++i) wt_buf[i] = 1.0f;
    lanes.survival_weight = __builtin_bit_cast(V, _mm512_load_ps(wt_buf));

    // VPCOMPRESSD: steps_remaining
    __m512i steps_raw = __builtin_bit_cast(__m512i, lanes.steps_remaining);
    alignas(64) int32_t steps_buf[16];
    _mm512_mask_compressstoreu_epi32(steps_buf, live_mask, steps_raw);
    for (int i = n_live; i < 16; ++i) steps_buf[i] = 0;
    lanes.steps_remaining = __builtin_bit_cast(IntV, _mm512_load_epi32(steps_buf));

    // scalar shuffle
    std::array<uint64_t, W> new_path_id{};
    std::array<int,      W> new_partner{};
    std::array<LaneRole, W> new_role{};

    std::size_t write = 0;
    for (std::size_t read = 0; read < W; ++read) {
        if (live_bits & (1u << read)) {
            new_path_id[write] = lanes.path_id[read];
            new_partner[write] = -1;
            new_role[write]    = (lanes.role[read] == LaneRole::HasPartner)
                                    ? LaneRole::Independent
                                    : lanes.role[read];
            ++write;
        }
    }
    for (std::size_t i = n_live; i < W; ++i) {
        new_path_id[i] = 0; new_partner[i] = -1; new_role[i] = LaneRole::Dead;
    }
    lanes.path_id      = new_path_id;
    lanes.partner_lane = new_partner;
    lanes.role         = new_role;

    for (std::size_t i = 0; i < W; ++i) {
        lanes.needs_refill[i] = (static_cast<int>(i) >= n_live);
        lanes.alive[i]        = (static_cast<int>(i) < n_live);
    }
}
#endif

/// dispatch: avx-512 hardware path when available and width matches
template <typename V>
void compact_lanes(LaneRegister<V>& lanes) {
#ifdef __AVX512F__
    if constexpr (V::size() == 16) {
        compact_lanes_avx512(lanes);
    } else {
        compact_lanes_portable(lanes);
    }
#else
    compact_lanes_portable(lanes);
#endif
}

} // namespace detail

struct EveryStep {
    template <typename V>
    bool maybe_compact(LaneRegister<V>& lanes) {
        if (none_of(lanes.needs_refill)) return false;
        detail::compact_lanes(lanes);
        return true;
    }
};

struct Adaptive {
    int threshold = 4;

    template <typename V>
    bool maybe_compact(LaneRegister<V>& lanes) {
        int dead_count = popcount(lanes.needs_refill);
        if (dead_count < threshold) return false;
        detail::compact_lanes(lanes);
        return true;
    }
};

} // namespace simd_mc::compaction
