# Stream-Compacting Barrier Option Monte Carlo Using `std::simd`

## 1. Introduction

`simd_mc` is a C++23 header-only library that prices barrier options via Monte Carlo simulation using `std::experimental::simd`. The central contribution is solving the dead lane problem: the systematic underutilisation of SIMD registers when simulated paths are knocked out by a barrier. The solution is in-register stream compaction, a technique borrowed from GPU programming but never previously applied to CPU-side financial Monte Carlo in C++.

I built this to explore what `std::experimental::simd` can actually do when pushed to the hardware level. The result is a Monte Carlo engine where every hot-path operation compiles to a named SIMD instruction, verified in assembly: zero scalar `expf`, `logf`, `sinf`, `cosf`, or `sqrtf` calls.

The project sits at the intersection of three domains: modern C++ language evolution, low-level performance engineering, and quantitative derivatives pricing. It is designed to function both as a usable library and as a teaching artifact for the C++ community.

---

## 2. Prior Art and Competitive Landscape

The individual pieces exist. Nobody has assembled them for this specific problem.

SIMD Monte Carlo for vanilla options exists in Intel's oneMKL examples, but these never encounter the barrier check. Barrier option Monte Carlo exists in QuantLib and dozens of GitHub projects, but all are scalar. Stream compaction for Monte Carlo exists on GPUs, in path tracing (SIGGRAPH), neutron transport (WARP), and GPU option pricing papers (CUDA). SIMD lane refilling exists on the CPU for database queries (Lang et al., VLDB 2018), using AVX-512's `VEXPANDPS` instruction.

The gap is the combination: bringing a GPU technique (stream compaction) to CPU SIMD registers, for a financial application (barrier options), using the newest C++ standard library feature (`std::simd`). Each ingredient exists somewhere; the recipe does not.

Why not just use a GPU? Latency. A GPU introduces several microseconds of kernel launch overhead before computation begins. For a risk engine repricing a single barrier option on a market tick, the CPU path is faster in total wall-clock time despite lower throughput. GPU Monte Carlo excels at batch pricing; CPU SIMD Monte Carlo excels at single-option latency-sensitive repricing. Both matter on a derivatives desk.

---

## 3. The Financial Problem

A barrier option is a path-dependent exotic derivative whose payoff depends on whether the underlying price breached a predetermined level at any point during the option's lifetime. A knock-out call with strike K and barrier B pays `max(S_T - K, 0)` at expiry, unless the price ever dropped below B, in which case it pays zero.

Barrier options are among the most widely traded exotics. They offer 20-50% premium savings over vanillas because the barrier condition reduces the probability of payout. They appear in autocallable notes, turbo warrants, accumulators, and yield enhancement instruments.

Monte Carlo simulation is often the only feasible pricing approach for discrete barrier monitoring (checking the barrier at daily closes), stochastic volatility models, multiple barriers, and correlation products. The Greeks for barrier options must be recomputed on every market tick via bump-and-reprice, which means the pricing engine is called not once but hundreds of thousands of times per risk cycle. This is where inner-loop performance translates directly into wall-clock time savings.

---

## 4. The Dead Lane Problem

In a standard SIMD Monte Carlo, W paths evolve in parallel across a single register. For vanilla options this gives a clean W-fold speedup. For barrier options, paths that hit the barrier "die" but continue occupying SIMD lanes, performing meaningless arithmetic.

On a 16-wide AVX-512 register pricing a barrier 15% below spot with 252 daily steps:

- By step 50: roughly 30% of lanes dead, 70% utilisation
- By step 125: roughly 55% of lanes dead, 45% utilisation
- By step 200: roughly 75% of lanes dead, 25% utilisation

The expected 16-fold speedup collapses to 5-7 times in practice. This is the CPU SIMD analogue of GPU warp divergence, and GPUs solve it with stream compaction: dead threads are replaced with fresh work.

The stream-compacting engine applies this same technique. When lanes die, it compresses survivors to the left side of the register and refills vacated right-side lanes with fresh paths. Lane utilisation stays above 90% throughout the simulation. On AVX-512, the compress step maps to a single `VCOMPRESSPS` instruction. On AVX2, a portable scalar shuffle provides the same result.

---

## 5. Why C++ and Why `std::simd`

The dead lane problem requires reasoning about hardware execution at the level of individual register lanes while simultaneously maintaining type safety and portability. This tension between low-level hardware control and high-level generic programming is what C++ does better than any other mainstream language.

`std::simd` (adopted into C++26 via P1928R15, available as `std::experimental::simd` in GCC 11+) provides a portable abstraction over hardware SIMD registers. The same code compiles to AVX-512 on Intel, AVX2 on older hardware, or NEON on ARM. No source changes. The compiler selects the optimal instructions for the target.

In practice, `std::experimental::simd` handles arithmetic, comparisons, sqrt, and max natively (single hardware instructions). But it does not vectorise transcendental functions: `stdx::exp` and `stdx::cos` fall back to scalar `libm` calls. The library addresses this with hand-written minimax polynomial approximations for `exp`, `log`, and `sincos` that compile to pure `vfmadd` chains. Documenting where `std::simd` falls short and what I had to work around is itself a contribution to the C++ ecosystem.

---

## 6. Architecture

Six C++20 concepts define the engine's extension points. Every policy call is inlined at compile time; the inner loop has zero virtual dispatch, zero heap allocation, and zero unpredictable branches.

| Concept | Responsibility | Example |
|---------|---------------|---------|
| `DiffusionModel` | Advance spot prices one step | `models::GBM` |
| `BarrierCondition` | Determine which lanes survive | `barriers::DownAndOutBridge` |
| `Payoff` | Compute terminal value | `payoffs::Call` |
| `RefillPolicy` | What fills recovered lanes | `refill::AntitheticPreferred` |
| `CompactionStrategy` | When to compact | `compaction::Adaptive` |
| `RandomEngine` | SIMD-width random draws | `rng::Philox` |

Adding a new model means writing one struct with an `evolve()` method. Nothing else changes. The builder function hides the template machinery so the call site reads like a problem specification:

```cpp
auto result = simd_mc::price(
    models::GBM{.risk_free_rate = 0.05f, .volatility = 0.20f},
    barriers::DownAndOutBridge{.level = 80.0f, .vol = 0.20f},
    payoffs::Call{.strike = 100.0f},
    SimConfig{.initial_spot = 100.0f, .n_paths = 500'000, .n_steps = 252, .T = 1.0f}
);
```

---

## 7. Implemented Optimisations

The engine went through three major iterations. Each was driven by assembly evidence and profiling, not guesswork.

**Vectorised transcendentals.** `simd_exp` uses Cephes-style range reduction with a degree-5 minimax Horner polynomial. `simd_log` uses IEEE 754 exponent extraction via `__builtin_bit_cast` (compiles to `ret`, zero instructions) and a rational polynomial via the (m-1)/(m+1) substitution. `simd_sincos` uses quadrant-based decomposition with minimax polynomials and SIMD masked blending. All compile to `vfmadd` chains. Zero scalar library calls.

**Vectorised RNG.** The Philox-2x32-10 hash runs on `stdx::rebind_simd_t<uint32_t, FloatV>`, hashing all W lanes in parallel via `vpmuludq` + `vpxord`. Box-Muller caches the sin branch and returns it on the next call, halving the number of uniform draws and transcendental evaluations.

**SIMD step counter.** `steps_remaining` is an `IntV` register, decremented with `vpcmpd` + masked `vpsubd` + `vpcmpeqd` instead of a W-element scalar loop.

**AVX-512 native compaction.** `VCOMPRESSPS` for `spot` and `survival_weight`, `VPCOMPRESSD` for `steps_remaining`. Three SIMD instructions replace roughly 100 scalar operations per compaction event. Falls back to a portable scalar shuffle on AVX2 via `if constexpr`.

**OpenMP multi-threading.** `price()` splits paths across hardware cores, each running an independent engine with a unique Philox seed. Results merged via Chan's parallel Welford algorithm. Verified: prices and standard errors match between single-threaded and multi-threaded runs.

**Brownian bridge correction.** The analytical crossing probability conditioned on step endpoints catches barrier breaches between observation dates. Compile-time toggle via template parameter. Result: 3.2 times accuracy improvement at identical path count.

**Additional.** Welford's online variance (numerically stable standard error). Clean knock-out detection via `was_alive` mask captured before the barrier check. Antithetic variance reduction in recovered lanes. Adaptive compaction threshold. Truncation detection flag.

---

## 8. Results

### 8.1 Test Environment

GCC 13 (MSYS2 UCRT64), C++23, `-O3 -march=native -fopenmp`, AVX2 8-wide, Windows 11.

### 8.2 Optimisation Impact

| Benchmark (500k paths, B=90) | v0.1 | v0.3 | Speedup |
|------------------------------|------|------|---------|
| Wall-clock time | 1,963 ms | 31 ms | 63 times |
| Scalar `expf` calls in engine | 16 per step | 0 | -- |
| Scalar `sinf`/`cosf` calls | 32 per step | 0 | -- |

### 8.3 Latency at Various Path Counts

| Paths | Price | Std Error | Time | Use Case |
|-------|-------|-----------|------|----------|
| 10,000 | 10.66 | 0.73 | 0.7 ms | Sub-ms indicative |
| 50,000 | 9.44 | 0.31 | 3.1 ms | Tick-by-tick hedging |
| 100,000 | 9.16 | 0.21 | 6.0 ms | Intraday risk |
| 500,000 | 9.06 | 0.09 | 31 ms | Production risk reporting |

### 8.4 Barrier Pricing

All knock-out prices are below the corresponding vanilla, as required. MC prices sit above the continuous-monitoring analytical values — this is correct and expected because discrete daily monitoring misses between-step crossings. The bias scales with barrier proximity: 0.04 at B=80 (20% below spot), 0.25 at B=90 (10% below spot).

| Barrier | Price | Lane Utilisation | Time (ms) |
|---------|-------|-----------------|-----------|
| B=70 (far) | 10.39 | 98.5% | 42 |
| B=80 | 10.23 | 95.9% | 40 |
| B=90 | 9.06 | 93.7% | 31 |
| B=95 (close) | 6.11 | 92.5% | 19 |
| B=98 (very close) | 3.55 | 90.4% | 11 |

Lane utilisation stays above 90% across all barrier distances. Without stream compaction, the close-barrier cases would collapse to 25-45%.

### 8.5 Brownian Bridge Correction

| Method | Price | Error vs Analytical |
|--------|-------|-------------------|
| Hard barrier check | 10.08 | 0.075 |
| Bridge correction | 9.96 | 0.023 |

Same paths, same steps, same RNG. One additional `exp`/`log` per step per lane. The bridge correction reduces error by a factor of 3.2.

### 8.6 Convergence

Std error halves when paths quadruple, confirming the 1/sqrt(N) convergence rate. The price converges toward the continuous-monitoring analytical value from above. The remaining gap is the discrete monitoring bias, not engine error. The OpenMP parallel Welford merge introduces no statistical bias.

---

## 9. Future Work

### 9.1 Accuracy Improvements

**Control variate (vanilla BS correction).** Use the analytical Black-Scholes vanilla price to correct the barrier estimate. Since both share the same random draws, their errors are highly correlated. Typical variance reduction: 50-80%, equivalent to 2-5 times more paths for free.

**Quasi-Monte Carlo (Sobol sequences).** Changes the convergence rate from O(1/sqrt(N)) to O((log N)^d / N). For 252-step paths, this can mean 10-100 times fewer paths for the same accuracy. Non-trivial interaction with compaction: refilled lanes must draw from fresh Sobol dimensions.

**Importance sampling.** Girsanov drift tilting shifts the probability measure so more paths approach the barrier region, then reweights by the likelihood ratio. The `survival_weight` field in `LaneRegister` already exists for this purpose.

### 9.2 Financial Extensions

**Greeks.** The biggest gap between demo and tool. A `GreeksEngine` wrapper calling `price()` with bumped parameters for delta, gamma, vega, theta. Requires double precision in the differencing layer to avoid catastrophic cancellation.

**Heston stochastic volatility.** Nobody prices barriers under flat vol in production. The Monte Carlo framework accommodates stochastic vol naturally by swapping the GBM evolution for a Heston step function.

**Double barriers.** Simultaneous upper and lower barriers (range accruals, double knock-out notes). The dead lane problem is worse because paths die from both directions.

**Autocallable notes.** The largest exotic by volume in European retail. Knock out on periodic observation dates with coupon accumulation. Fits the existing concept architecture as a new `Payoff` policy.

**Asian features, worst-of baskets.** Running average accumulators and multi-asset engines. Multi-asset barriers amplify the dead lane problem, which is where stream compaction provides its largest relative benefit.

### 9.3 Infrastructure

**Double precision.** Parameterise the engine on the scalar type. Required for safe Greeks differencing. `stdx::native_simd<double>` gives 4-wide on AVX2, 8-wide on AVX-512.

**C++26 `std::simd` migration.** Mechanical find-and-replace when GCC 16 ships.

**Multilevel Monte Carlo.** Giles' 2008 method telescopes the estimator across resolution levels. The different MLMC levels map naturally to different compaction strategies: coarsest levels need no compaction, finest levels need aggressive compaction.

---

## 10. Hardware

**AVX-512** is the primary target. 16-wide float32, `VCOMPRESSPS` for single-instruction compaction, 8 hardware mask registers for the engine's alive/completed/needs_refill masks.

**AVX2** is the most common in production trading infrastructure. 8-wide float32. The portable compaction fallback runs here. All SIMD math functions work identically.

**ARM SVE/NEON** works with a recompile. `std::experimental::simd` targets NEON at 4-wide. SVE predication maps naturally to the engine's masked control flow.

**GPU** is not the target. The engine trades throughput for latency, which is the right tradeoff for tick-by-tick barrier hedging on a derivatives desk.

---

## 11. What I Learned

The dead lane problem is not specific to single-barrier options. It appears in any path-dependent simulation where paths terminate early: double barriers, autocallables, American exercise, credit default models. Every one of these is a candidate for stream compaction on CPU SIMD hardware.

Writing vectorised polynomial approximations that match `libm` accuracy within MC tolerances taught me more about IEEE 754 floating-point than any textbook. The `__builtin_bit_cast` between `stdx::simd<float>` and `stdx::simd<int32_t>` compiling to `ret` (literally zero instructions) was the most satisfying assembly output I have ever seen.

The antithetic phase correction that I implemented, discovered was biased for barrier options, and reverted taught me that correctness always comes before cleverness. The double-harvest bug that was masked at narrow SIMD widths by the compaction timing taught me that testing at a single configuration is not testing.

---

## References

M. Broadie, P. Glasserman, S. Kou. A continuity correction for discrete barrier options. Mathematical Finance, 7(4):325-349, 1997.

L. Andersen, R. Brotherton-Ratcliffe. Exact exotics. Risk, 9(10):85-89, 1996.

M. Giles. Multilevel Monte Carlo path simulation. Operations Research, 56(3):607-617, 2008.

M. Joshi. C++ Design Patterns and Derivatives Pricing. Cambridge University Press, 2003.

M. Kretz. Extending C++ for explicit data-parallel programming via SIMD vector types. PhD thesis, Goethe University Frankfurt, 2015.

M. Kretz. Merge data-parallel types from the Parallelism TS 2. P1928R15, WG21, 2025.

H. Lang et al. Data blocks: Hybrid OLTP and OLAP on compressed storage. SIGMOD, 2018.

D. Roger, U. Assarsson, N. Holzschuch. Efficient stream reduction on the GPU. GPGPU Workshop, 2007.

J. Salmon et al. Parallel random numbers: As easy as 1, 2, 3. SC11, 2011.

E. G. Haug. The Complete Guide to Option Pricing Formulas. McGraw-Hill, 2007.
