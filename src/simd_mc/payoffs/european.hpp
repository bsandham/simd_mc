#pragma once
/// simd_mc/payoffs/european.hpp
/// european call/put payoff. max() compiles to vmaxps

#include "../core/simd_compat.hpp"

namespace simd_mc::payoffs {

enum class OptionType { Call, Put };

template <OptionType Type>
struct European {
    float strike = 100.0f;

    template <typename V>
    [[nodiscard]] V evaluate(V terminal_spot, V survival_weight) const {
        V zero = V(0.0f);
        V intrinsic;
        if constexpr (Type == OptionType::Call) {
            intrinsic = simd_max(terminal_spot - V(strike), zero);
        } else {
            intrinsic = simd_max(V(strike) - terminal_spot, zero);
        }
        return intrinsic * survival_weight;
    }
};

using Call = European<OptionType::Call>;
using Put  = European<OptionType::Put>;

} // namespace simd_mc::payoffs
