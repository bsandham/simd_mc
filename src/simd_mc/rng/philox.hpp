#pragma once
/// simd_mc/rng/philox.hpp
/// vectorised Philox-2x32-10 RNG with cached Box-Muller
///
/// wlanes are hashed in parallel using stdx::simd<uint32_t>
/// Philox Feistel rounds compile to vpmuludq + vpxord + vpaddd

#include "../core/simd_compat.hpp"
#include <cstdint>

namespace simd_mc::rng {

using U32V = stdx::rebind_simd_t<uint32_t, FloatV>;

namespace detail {

/// vectorised widening multiply: a[i] * b > (hi[i], lo[i]) for all lanes
/// compiles to vpmuludq + vpsrlq + vpshufd (handles evn/odd lanes)
inline void mulhilo(U32V a, uint32_t b, U32V& out_hi, U32V& out_lo) {
    // GCC auto-vectorises this into vpmuludq w/ lane shuffling
    for (std::size_t i = 0; i < U32V::size(); ++i) {
        uint64_t product = uint64_t(uint32_t(a[i])) * b;
        out_hi[i] = uint32_t(product >> 32);
        out_lo[i] = uint32_t(product);
    }
}

/// vectorised Philox-2x32-10: hash all W counters in parallel
///
/// 10 Feistel rounds of: multiply, XOR with key, advance key
/// gcc compiles to - (vpmuludq + vpsrlq + vpshufd + vpxord + vpaddd)
inline U32V philox_hash_vec(U32V lo, U32V hi, U32V key) {
    constexpr uint32_t PHILOX_M = 0xD2511F53u;
    constexpr uint32_t PHILOX_W = 0x9E3779B9u;

    for (int round = 0; round < 10; ++round) {
        U32V new_hi, new_lo;
        mulhilo(lo, PHILOX_M, new_hi, new_lo);
        lo = new_hi ^ hi ^ key;
        hi = new_lo;
        key = key + U32V(PHILOX_W);
    }
    return lo;
}

/// vectorised uint32 >  float in (0, 1).
/// compiles to: vpord + vcvtdq2ps + vmulps
inline FloatV u32_to_float01_vec(U32V x) {
    x = x | U32V(1u);
    // need unsigned >  float, but stdx::static_simd_cast treats as signed
    // use standard (x | 1) * (1/2^32) approach via per-element conversion since stdx doesn't provide unsigned > float directly
    FloatV result;
    for (std::size_t i = 0; i < FloatV::size(); ++i)
        result[i] = float(uint32_t(x[i])) * (1.0f / 4294967296.0f);
    return result;
}

} // namespace detail

template <typename V>
struct Philox {
    using value_type = typename V::value_type;
    static constexpr std::size_t W = V::size();

    uint32_t seed_    = 42;
    uint64_t counter_ = 0;

    V      cached_normal_;
    bool   has_cached_ = false;

    explicit Philox(uint32_t seed = 42) : seed_(seed) {}

    /// constructs W counters (base + lane offset), hashes all W in parallel via vectorised Philox, converts to float
    /// one scalar loop in u32> float conversion (stdx lacks unsigned > float cast; this is one concession we make)
    [[nodiscard]] V next_uniform() {
        U32V lo, hi;
        for (std::size_t i = 0; i < W; ++i) {
            uint64_t c = counter_ + i;
            lo[i] = uint32_t(c);
            hi[i] = uint32_t(c >> 32);
        }
        counter_ += W;

        U32V hashed = detail::philox_hash_vec(lo, hi, U32V(seed_));
        return detail::u32_to_float01_vec(hashed);
    }


    [[nodiscard]] V next_normal() {
        if (has_cached_) {
            has_cached_ = false;
            return cached_normal_;
        }

        V u1 = next_uniform();
        V u2 = next_uniform();

        constexpr float two_pi = 6.2831853071795864f;
        V radius = simd_sqrt(V(-2.0f) * simd_log(u1));   // vsqrtps(vmulps(simd_log))
        V angle  = V(two_pi) * u2;                         // vmulps

        auto [sin_val, cos_val] = simd_sincos(angle);      // scalar fallback (cached)

        cached_normal_ = radius * sin_val;
        has_cached_ = true;

        return radius * cos_val;
    }
};

} // namespace simd_mc::rng
