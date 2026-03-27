#pragma once
/// simd_mc/models/gbm.hpp
/// gbm: S(t+dt) = S(t) * exp((r-q-½σ²)dt + σ√dt Z)
/// simd_compat.hpp compiles to a vfmadd chain

#include "../core/simd_compat.hpp"
#include <cmath>

namespace simd_mc::models {

struct GBM {
    float risk_free_rate = 0.05f;
    float dividend_yield = 0.0f;
    float volatility     = 0.20f;

    float drift_per_step = 0.0f;
    float vol_sqrt_dt    = 0.0f;

    constexpr void set_timestep(float dt) {
        float mu = risk_free_rate - dividend_yield;
        drift_per_step = (mu - 0.5f * volatility * volatility) * dt;
        vol_sqrt_dt    = volatility * std::sqrt(dt);
    }

    template <typename V>
    [[nodiscard]] V evolve(V spot, V Z, float /*dt*/) const {
        V exponent = V(drift_per_step) + V(vol_sqrt_dt) * Z;
        return spot * simd_exp(exponent);
    }
};

} // namespace simd_mc::models
