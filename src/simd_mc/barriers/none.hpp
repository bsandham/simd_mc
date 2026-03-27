#pragma once
#include "../core/simd_compat.hpp"

namespace simd_mc::barriers {

struct None {
    template <typename V>
    [[nodiscard]] auto check(V, V, V, typename V::value_type, typename V::mask_type prior_alive
    ) const -> typename V::mask_type {
        return prior_alive;
    }
};

} // namespace simd_mc::barriers
