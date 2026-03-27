#pragma once
/// simd_mc/core/simd_compat.hpp
/// SIMD types and operations built on std::experimental::simd - parallelism 2
///
/// simd_exp: 8x vfmadd Horner chain, vcvttps2dq floor, __builtin_bit_cast
/// simd_log: IEEE 754 bit extraction via __builtin_bit_cast, vcvtdq2ps, Horner chain
/// simd_sqrt: vsqrtps delegates to stdx::sqrt
/// simd_max: vmaxps delegates to stdx::max

#include <experimental/simd>
#include <cstdint>
#include <cmath>

namespace simd_mc {

namespace stdx = std::experimental;

using FloatV = stdx::native_simd<float>;
using IntV   = stdx::rebind_simd_t<int32_t, FloatV>;
using MaskV  = stdx::native_simd_mask<float>;
using IntMaskV = stdx::native_simd_mask<int32_t>;

inline constexpr int simd_width = FloatV::size();

using stdx::any_of;
using stdx::none_of;
using stdx::all_of;
using stdx::popcount;

// on AVX-512, both are the same k-register. __builtin_bit_cast is zero-cost
inline MaskV  int_mask_to_float(IntMaskV m) { return __builtin_bit_cast(MaskV, m); }
inline IntMaskV float_mask_to_int(MaskV m) { return __builtin_bit_cast(IntMaskV, m); }

inline FloatV select(MaskV mask, FloatV a, FloatV b) {
    FloatV result = b;
    stdx::where(mask, result) = a;
    return result;
}

inline float reduce(FloatV v) {
    return stdx::reduce(v);
}

namespace fast {

// GCC's native_simd<float> and rebind_simd_t<int32_t> share the same ZMM
// register. __builtin_bit_cast just tells the compiler to reinterpret the type

inline IntV   float_as_int(FloatV v) { return __builtin_bit_cast(IntV, v); }
inline FloatV int_as_float(IntV v)   { return __builtin_bit_cast(FloatV, v); }

// vcvttps2dq (float > int truncate) + vcvtdq2ps (int > float) + masked vsubps

inline FloatV floor(FloatV x) {
    IntV   xi        = stdx::static_simd_cast<IntV>(x);
    FloatV truncated = stdx::static_simd_cast<FloatV>(xi);
    stdx::where(x < truncated, truncated) = truncated - FloatV(1.0f);
    return truncated;
}
// vcvtdq2ps / vcvttps2dq

inline FloatV int_to_float(IntV x)   { return stdx::static_simd_cast<FloatV>(x); }
inline IntV   float_to_int(FloatV x) { return stdx::static_simd_cast<IntV>(x); }

} // namespace fast

/// vectorised exp(x)
///
/// range reduction: n = floor(x/ln2 + 0.5) < > fast::floor
/// reduced argument: r = x - n*ln2 < > 2x vfmsub
/// polynomial: degree-5 Horner < > 6x vfmadd
/// reconstruction: 2^n via bit manip < > 1x vpslld + __builtin_bit_cast
inline FloatV simd_exp(FloatV x) {
    constexpr float log2e  = 1.44269504088896341f;
    constexpr float ln2_hi = 0.693145751953125f;
    constexpr float ln2_lo = 1.42860682030941723e-6f;
    constexpr float c1 = 1.0f, c2 = 0.5f, c3 = 0.1666666716f;
    constexpr float c4 = 0.0416664853f, c5 = 0.0083312546f, c6 = 0.0013842773f;

    x = stdx::max(x, FloatV(-87.3f));                           // vmaxps
    x = stdx::min(x, FloatV(88.7f));                            // vminps

    FloatV n = fast::floor(x * FloatV(log2e) + FloatV(0.5f));   // 3 SIMD ops
    FloatV r = x - n * FloatV(ln2_hi) - n * FloatV(ln2_lo);     // vfmsub chain

    FloatV poly = FloatV(c6);
    poly = poly * r + FloatV(c5);
    poly = poly * r + FloatV(c4);
    poly = poly * r + FloatV(c3);
    poly = poly * r + FloatV(c2);
    poly = poly * r + FloatV(c1);
    poly = poly * r + FloatV(1.0f);

    // 2^n: convert n to int (vcvttps2dq), add bias, shift to exponent field
    IntV int_n      = fast::float_to_int(n);                     // vcvttps2dq
    IntV scale_bits = (int_n + IntV(127)) << 23;                 // vpaddd + vpslld
    FloatV scale    = fast::int_as_float(scale_bits);            // ret (no-op)

    return poly * scale;                                         // vmulps
}

/// vectorised log(x) — zero scalar loops.
///
/// extract exponent and mantissa via bit ops (IEEE 754 decomposition)
/// sub: u = (m-1)/(m+1), u in set [0, 1/3)
/// polynomial: log(m) = 2u(1 + u²/3 + u⁴/5 + u⁶/7 + u⁸/9) via Horner
///
/// bit extraction via __builtin_bit_cast
/// int-to-float via vcvtdq2ps
inline FloatV simd_log(FloatV x) {
    constexpr float ln2 = 0.6931471805599453f;

    IntV xi            = fast::float_as_int(x);                          // ret (no-op)
    IntV exponent      = ((xi >> 23) & IntV(0xFF)) - IntV(127);          // vpsrad + vpandd + vpsubd
    IntV mantissa_bits = (xi & IntV(0x007FFFFF)) | IntV(0x3F800000);     // vpandd + vpord
    FloatV m           = fast::int_as_float(mantissa_bits);              // ret (no-op)
    FloatV e           = fast::int_to_float(exponent);                   // vcvtdq2ps

    // u = (m-1)/(m+1) in set [0, 1/3) — much better convergence than (m-1) in set [0, 1)
    FloatV u  = (m - FloatV(1.0f)) / (m + FloatV(1.0f));                // vsubps + vaddps + vdivps
    FloatV u2 = u * u;                                                   // vmulps

    // Horner on u²: 4 vfmadd instructions
    FloatV poly = FloatV(1.0f / 9.0f);
    poly = poly * u2 + FloatV(1.0f / 7.0f);
    poly = poly * u2 + FloatV(1.0f / 5.0f);
    poly = poly * u2 + FloatV(1.0f / 3.0f);
    poly = poly * u2 + FloatV(1.0f);

    return e * FloatV(ln2) + FloatV(2.0f) * u * poly;                   // vfmadd chain
}

/// vsqrtps — single instruction
inline FloatV simd_sqrt(FloatV x) { return stdx::sqrt(x); }

/// vmaxps — single instruction
inline FloatV simd_max(FloatV a, FloatV b) { return stdx::max(a, b); }

/// returns {sin(x), cos(x)} for x in set [0, 2π) (Box-Muller input range) 
///
/// quadrant decomposition: q = floor(x / (π/2)), reduce to [0, π/2), then apply minimax polynomials for sin and cos on the reduced range
struct SinCosResult { FloatV sin_val; FloatV cos_val; };

inline SinCosResult simd_sincos(FloatV x) {
    constexpr float two_over_pi = 0.6366197723675814f;
    constexpr float pio2_hi     = 1.5707963267f;
    constexpr float pio2_lo     = 9.48966e-11f;

    // quadrant: q = floor(x / (π/2)) in set {0,1,2,3}
    IntV q = fast::float_to_int(x * FloatV(two_over_pi));           // vcvttps2dq
    q = q & IntV(3);                                                 // vpandd
    FloatV qf = fast::int_to_float(q);                               // vcvtdq2ps

    // reduced angle t in [0, π/2) — extended precision subtraction
    FloatV t  = (x - qf * FloatV(pio2_hi)) - qf * FloatV(pio2_lo);
    FloatV t2 = t * t;

    // sin(t) apprx t + t³·P(t²) — Horner on t²
    FloatV sp = FloatV(-1.984126984e-4f);                           
    sp = sp * t2 + FloatV(8.333333333e-3f);                          
    sp = sp * t2 + FloatV(-1.666666667e-1f);                         
    FloatV s = t + t * t2 * sp;                                      // 3× vfmadd

    // cos(t) apprx 1 + t²·Q(t²) — Horner on t²
    FloatV cp = FloatV(-1.388888889e-3f);                           
    cp = cp * t2 + FloatV(4.166666667e-2f);                       
    cp = cp * t2 + FloatV(-5.0e-1f);                                
    FloatV c = FloatV(1.0f) + t2 * cp;                               // 3× vfmadd

    //   q=0: sin= s, cos= c    q=1: sin= c, cos=-s
    //   q=2: sin=-s, cos=-c    q=3: sin=-c, cos= s
    auto q0 = int_mask_to_float(q == IntV(0));                       // vpcmpeqd >  bit_cast
    auto q1 = int_mask_to_float(q == IntV(1));
    auto q2 = int_mask_to_float(q == IntV(2));
    auto q3 = int_mask_to_float(q == IntV(3));

    FloatV sin_r(0.0f);
    stdx::where(q0, sin_r) =  s;
    stdx::where(q1, sin_r) =  c;
    stdx::where(q2, sin_r) = -s;
    stdx::where(q3, sin_r) = -c;

    FloatV cos_r(0.0f);
    stdx::where(q0, cos_r) =  c;
    stdx::where(q1, cos_r) = -s;
    stdx::where(q2, cos_r) = -c;
    stdx::where(q3, cos_r) =  s;

    return {sin_r, cos_r};
}

} // namespace simd_mc
