# `simd_mc`

## C++23 | `std::experimental::simd`

A header-only C++23 library that prices barrier options via Monte Carlo simulation using `std::experimental::simd` (Parallelism TS 2, GCC 11+). Solves the dead lane problem - the systematic underutilisation of SIMD registers when simulated paths are knocked out by a barrier - via an in-register stream compaction implementation stolen from the people who figured it out for the GPU :)

<div align="center">

| Feature | Assembly-verified SIMD |
|:--------|:------:|
| Vectorised `exp()` — minimax polynomial | Yes, 8× `vfmadd` |
| Vectorised `log()` — IEEE 754 decomposition, (m-1)/(m+1) rational polynomial | Yes |
| Vectorised Philox-2x32-10 RNG | Yes, `vpmuludq` + `vpxord` |
| SIMD step counter — `IntV` register | Yes, `vpcmpd` + masked `vpsubd` |
| Zero-cost bit reinterpretation — `__builtin_bit_cast`` | Yes, compiles to `ret |
| SIMD floor | Yes, `vcvttps2dq` + `vcvtdq2ps` + masked `vsubps` |
| Brownian bridge barrier correction (compile-time toggle via `if constexpr`) | Yes |
| Antithetic variance reduction in recovered lanes | Yes |
| Adaptive compaction threshold | Yes |
| Welford's online variance (numerically stable std error) | Yes |
</div>

---

## API Usage

### Price Function
```cpp
#include <simd_mc/simd_mc.hpp>

auto result = simd_mc::price(
    simd_mc::models::GBM{.risk_free_rate = 0.05f, .volatility = 0.20f},
    simd_mc::barriers::DownAndOutBridge{.level = 80.0f, .vol = 0.20f},
    simd_mc::payoffs::Call{.strike = 100.0f},
    simd_mc::SimConfig{
        .initial_spot   = 100.0f,
        .n_paths        = 1'000'000,
        .n_steps        = 252,
        .T              = 1.0f,
        .risk_free_rate = 0.05f
    }
);
// result.price            — discounted option value
// result.std_error        — Monte Carlo standard error
// result.paths_simulated  — number of completed paths
// result.lane_utilisation — avg% of live SIMD lanes
```

### Input Parameters

<div align="center">

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `initial_spot` | `float` | Current underlying price S₀ |
| `n_paths` | `int` | Number of Monte Carlo paths (controls accuracy) |
| `n_steps` | `int` | Time steps per path (eg 252 = daily for 1 year) |
| `T` | `float` | Time to maturity in years |
| `risk_free_rate` | `float` | Annualised r |

</div>

### Model Policies

<div align="center">

| Policy | Template Parameter | Example |
|:-------|:-------------------|:--------|
| Diffusion model | `models::GBM` | `{.risk_free_rate = 0.05f, .volatility = 0.20f}` |
| Barrier condition | `barriers::DownAndOut` | `{.level = 80.0f}` |
| Barrier + bridge | `barriers::DownAndOutBridge` | `{.level = 80.0f, .vol = 0.20f}` |
| No barrier | `barriers::None` | `{}` |
| Call payoff | `payoffs::Call` | `{.strike = 100.0f}` |
| Put payoff | `payoffs::Put` | `{.strike = 100.0f}` |

</div>

---

## Design / Approaches

### Background: Barrier Options and SIMD

A knock-out barrier option pays `max(S-K, 0)` at expiry **unless** the underlying price `S` breaches a barrier level `B` at any monitoring date during the option's life. If the barrier is hit, the option is cancelled hence pays 0.

In a Monte Carlo simulation, each path evolves independently according to:

$$ S_{t+\Delta t} = S_t \cdot \exp\left[\left(r - \tfrac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t}\; Z\right], \quad Z \sim \mathcal{N}(0,1) $$

With SIMD, `W` paths evolve in parallel across a single register. But when a path hits the barrier, its lane "dies", which means it continues occupying the register but is useless.

### The Problem

On a `W=8` AVX2 register pricing a barrier 5% below spot with 252 daily steps, lane utilisation can be **25–45%** by mid-simulation.

This is the CPU SIMD analogue of **GPU warp divergence**. GPUs solve it with stream compaction - dead threads are replaced with fresh work.

I chose to take a look at this as I am not aware this implementation exists elsewhere (open-source at least). It was also just cool to play around with SIMD in C++23 before it "graduates" out of its experimental package in C++26.

### Stream Compaction

When lanes die, we compress survivors to the left side of the register and refill vacated right-side lanes with fresh paths e.g.:

```
Before compaction:  [LIVE] [DEAD] [LIVE] [DEAD] [LIVE] [DEAD] [DEAD] [LIVE]
After compaction:   [LIVE] [LIVE] [LIVE] [LIVE] [NEW]  [NEW]  [NEW]  [NEW]
```

Ensuring lane utilisation stays as close as possible to 100% throughout the simulation.

### Brownian Bridge Correction

Discrete monitoring (checking the barrier at 252 daily steps) misses between-step crossings. The Brownian bridge correction finds the analytical probability that the continuous path crossed the barrier between two discrete observations:

$$ P(\text{cross}\mid S_{t}, S_{t+\Delta t}) = \exp\left(\frac{-2\ln(S_t/B)\ln(S_{t+\Delta t}/B)}{\sigma^2 \Delta t}\right) $$

This is accessible as an option at compile time - `DownAndOutBridge` vs `DownAndOut`. When disabled, the bridge code compiles to nothing.

### Knock-In via Parity

Knock-in barriers are priced via:

$$ V_{\text{knock-in}} = V_{\text{vanilla}}^{\text{BS}} - V_{\text{knock-out}}^{\text{MC}} $$

The analytical Black-Scholes vanilla price is exact, so the knock-in estimate inherits only the Monte Carlo error of the knock-out.

---

## Assembly Verification

Verify with:
```bash
g++ -std=c++23 -O3 -march=native -S -I src your_file.cpp -o output.s
grep -c "call.*expf" output.s    # Should be 0
grep -c "call.*logf" output.s    # Should be 0
```

### Instruction Breakdown (AVX2 8-wide)

<div align="center">

| Operation | Instruction(s) | Count | Scalar Loops Identified |
|:----------|:---------------|:-----:|:------------:|
| GBM evolve: `drift + vol*Z` | `vfmadd132ps` | 1 | 0 |
| `exp()` Horner polynomial | `vfmadd132ps` / `vfmadd213ps` | 8 | 0 |
| `exp()` floor (range reduction) | `vcvttps2dq` + `vcvtdq2ps` | 2 | 0 |
| `exp()` 2^n reconstruction | `vpslld` + `__builtin_bit_cast` → `ret` | 1 | 0 |
| `log()` IEEE 754 decomposition | `vpsrad` + `vpandd` + `vpord` | 3 | 0 |
| `log()` int-to-float | `vcvtdq2ps` | 1 | 0 |
| `log()` Horner polynomial | `vfmadd` chain | 4 | 0 |
| Barrier check: `spot > B` | `vcmpps` → k-register | 1 | 0 |
| Mask logic: `alive && survived` | `kandw` | 1 | 0 |
| Payoff: `max(S-K, 0)` | `vmaxps` | 1 | 0 |
| Step decrement | `vpcmpd` + masked `vpsubd` + `vpcmpeqd` | 3 | 0 |
| Philox RNG (10 Feistel rounds) | `vpmuludq` + `vpxord` + `vpaddd` | ~40 | 0 |
| `sqrt()` | `vsqrtps` | 1 | 0 |
| Bit reinterpretation | `__builtin_bit_cast` → `ret` | 0 | 0 |

</div>

The only scalar on the per-step path is `sincos` in Box-Muller - mitigated by caching (fires once per `2W` normals, not every step). Maybe someone could take a look at this...

---

## Results

### Testing

<div align="center">

| Component | Specification |
|:----------|:-------------|
| Compiler | GCC 13.2 (MSYS2 UCRT64) |
| Standard | C++23 (`-std=c++23`) |
| Flags | `-O3 -march=native -ffast-math` |
| ISA | AVX2, 8-wide (`native_simd<float>::size() = 8`) |
| OS | Windows 11, MSYS2 |
| Paths | 500,000 |
| Steps | 252 (daily monitoring, T=1 year) |
| Parameters | S₀=100, K=100, r=5%, σ=20%, T=1y |

</div>

### Vanilla Convergence (No Barrier)

<div align="center">

| Test | MC Price | Black-Scholes | Error | Result |
|:-----|:---------|:-------------|:------|:------:|
| ATM Call (S₀=K=100) | 10.4445 | 10.4506 | 0.0061 | Yes |
| ATM Put (S₀=K=100) | 5.5706 | 5.5735 | 0.0029 | Yes |
| ITM Call (K=90, σ=30%) | 15.4809 | 15.4860 | 0.0051 | Yes |
| OTM Call (K=110, σ=30%) | 5.5813 | 5.5871 | 0.0058 | Yes |

</div>

All errors are in the thousandths - consistent with 500k-path MC statistical noise.

### Barrier Pricing vs Continuous-Monitoring Analytical

<div align="center">

| Test | Barrier | MC Price | Analytical (cont.) | Error | Notes |
|:-----|:--------|:---------|:-------------------|:------|:------|
| Far barrier | B=80 | 10.3916 | 10.3513 | 0.0403 | MC > analytical — expected (discrete monitoring bias) |
| Medium barrier | B=90 | 8.9104 | 8.6655 | 0.2449 | 6× larger bias at closer barrier |

</div>

You can see that the MC price is above the continuous-monitoring analytical price. This is fine, as discrete daily monitoring misses between-step barrier crossings, so fewer paths are knocked out, inflating the price.
The bias scales with barrier proximity as predicted by Broadie-Glasserman-Kou theory. https://www.columbia.edu/~sk75/mfBGK.pdf.

### Knock-In Parity

<div align="center">

| Component | Value |
|:----------|:------|
| Vanilla Black-Scholes (exact) | 10.4506 |
| Knock-Out MC (B=85) | 10.0240 |
| **Knock-In via parity** | **0.4266** |
| Knock-In analytical (cont.) | 0.5013 |
| Error | 0.0747 |

</div>

Error is inherited entirely from the knock-out estimate. Notable as it validates use of a control variate.

### Brownian Bridge Correction

<div align="center">

| Method | Price | Error vs Analytical | Improvement |
|:-------|:------|:-------------------|:------------|
| Hard barrier check | 10.0240 | 0.0747 | - |
| **Brownian bridge correction** | **9.9725** | **0.0233** | **3.2×** |
| Analytical (continuous) | 9.9493 | - | - |

</div>

One additional `exp`/`log` per step per lane. The bridge-corrected price is within 0.02 of the continuous-monitoring analytical.

### Lane Utilisation

At B=95 (barrier 5% below spot price), lane utilisation measured at **92.5%**. Without stream compaction, this would likely be 25–45% by mid-simulation.

### Barrier Proximity Sweep (Crossover Benchmark)

<div align="center">

| Barrier | Price | Std Error | Util % | Time (ms) |
|:--------|:------|:----------|:-------|:----------|
| 70 | 10.4522 | 0.0329 | 99.6 | 2990 |
| 75 | 10.4182 | 0.0330 | 99.6 | 2953 |
| 80 | 10.3628 | 0.0330 | 99.6 | 2751 |
| 85 | 10.0288 | 0.0331 | 99.5 | 2429 |
| 90 | 8.9717 | 0.0329 | 99.4 | 1963 |
| 92 | 8.0831 | 0.0325 | 99.3 | 1729 |
| 95 | 6.2314 | 0.0303 | 99.1 | 1269 |
| 97 | 4.4744 | 0.0270 | 98.6 | 907 |
| 98 | 3.4230 | 0.0243 | 98.2 | 691 |

</div>

**Observations:**
- Price drops monotonically as barrier approaches spot (more knock-outs → less payoff)
- Execution time drops with closer barriers (knocked-out paths complete faster, replaced by shorter-lived paths)
- Utilisation stays above 98% across all barrier distances — the stream compaction engine keeps SIMD lanes productive :)

---

## Architecture

Six C++20 concepts define the engine's extension points, where every policy call is inlined at compile time.

<div align="center">

| Concept | Responsibility | Example | Hot Path? |
|:--------|:--------------|:--------|:---------:|
| `DiffusionModel` | Advance spot prices one step | `models::GBM` | Yes |
| `BarrierCondition` | Determine which lanes survive | `barriers::DownAndOutBridge` | Yes |
| `Payoff` | Compute terminal value | `payoffs::Call` | Yes |
| `RefillPolicy` | Decide what fills recovered lanes | `refill::AntitheticPreferred` | No |
| `CompactionStrategy` | Decide when to compact | `compaction::Adaptive` | No |
| `RandomEngine` | Produce SIMD-width random draws | `rng::Philox` | Yes |

</div>

Adding a new model (e.g. Heston, local vol) is easy as you just need to write one struct with an `evolve()` method.

### SIMD Type Mapping

<div align="center">

| C++ Type | Hardware | Instruction Examples |
|:---------|:---------|:--------------------|
| `FloatV` = `stdx::native_simd<float>` | ZMM/YMM register | `vmulps`, `vaddps`, `vfmadd132ps` |
| `IntV` = `stdx::rebind_simd_t<int32_t, FloatV>` | ZMM/YMM register | `vpaddd`, `vpsubd`, `vpmuludq` |
| `MaskV` = `stdx::native_simd_mask<float>` | k-register (AVX-512) / YMM (AVX2) | `kandw`, `korw`, `knotw` |

</div>

---

## Build

### Requirements
- GCC 11+ (ships `std::experimental::simd` in libstdc++)
- C++23 mode (`-std=c++23`)

### Compile
```bash
# Direct compilation
g++ -std=c++23 -O3 -march=native -I src benchmarks/basic_barrier.cpp -o basic_barrier -lm

# CMake (MSYS2/MinGW)
mkdir -p out/build && cd out/build
cmake ../.. && ninja

# Run tests
./test_vanilla
./test_barrier
```

### Project Structure
```
simd_mc/
├── src/simd_mc/           # Header-only library
│   ├── core/              # simd_compat.hpp, lane_register.hpp, sim_config.hpp
│   ├── concepts/          # 6 policy concepts
│   ├── models/            # GBM diffusion
│   ├── barriers/          # Down/Up × Out × Hard/Bridge, None
│   ├── payoffs/           # European Call/Put
│   ├── refill/            # Independent, AntitheticPreferred
│   ├── compaction/        # EveryStep, Adaptive
│   ├── rng/               # Vectorised Philox + cached Box-Muller
│   ├── engine/            # MonteCarloEngine template
│   └── api/               # price() convenience function
├── src/tests/             # 9 tests (4 vanilla, 5 barrier)
├── benchmarks/            # basic_barrier, barrier proximity sweep
├── CMakeLists.txt         # GCC, C++23, -O3 -march=native -ffast-math
└── CMakeSettings.json     # MSYS2 UCRT64 Visual Studio config
```

---

## Design Influences

<div align="center">

| Source | What We Take |
|:-------|:-------------|
| **QuantLib** (Ballabio) | Conceptual separation of instrument, process, engine, payoff |
| **Joshi** (*C++ Design Patterns and Derivatives Pricing*) | Template Method for MC loop, Strategy for statistics |
| **HFT tradition** (Ghosh, Sapir) | Zero-overhead compile-time dispatch, no virtuals on hot path |
| **Giles** (MLMC, 2008) | Hierarchical MC, level-dependent strategy |
| **Salmon et al.** (Philox, SC11) | Counter-based RNG for SIMD-parallel generation |
| **Lang et al.** (VLDB 2018) | AVX-512 stream compaction for database query vectorisation |
| **Roger et al.** (GPU stream compaction, 2007) | Warp-level compaction for rendering |
| **Broadie, Glasserman, Kou** (1997) | Brownian bridge barrier correction |
| **Andersen, Brotherton-Ratcliffe** (1996) | Bridge-corrected barrier monitoring |

</div>

## Useful Literature

[BGK97] M. Broadie, P. Glasserman, and S. Kou. A continuity correction for discrete barrier options. *Mathematical Finance*, 7(4):325–349, 1997.

[ABR96] L. Andersen and R. Brotherton-Ratcliffe. Exact exotics. *Risk*, 9(10):85–89, 1996.

[DAO22] T. Dao, D. Fu, S. Ermon, A. Rudra, and C. Ré. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. *NeurIPS*, 2022.

[GIL08] M. Giles. Multilevel Monte Carlo path simulation. *Operations Research*, 56(3):607–617, 2008.

[JOS03] M. Joshi. *C++ Design Patterns and Derivatives Pricing*. Cambridge University Press, 2003.

[LAN18] H. Lang, T. Mühlbauer, F. Funke, P. Boncz, T. Neumann, and A. Kemper. Data blocks: Hybrid OLTP and OLAP on compressed storage using both vectorization and compilation. In *SIGMOD*, 2018.

[ROG07] D. Roger, U. Assarsson, and N. Holzschuch. Efficient stream reduction on the GPU. In *Workshop on General Purpose Processing on Graphics Processing Units*, 2007.

[SAL11] J. Salmon, M. Moraes, R. Dror, and D. Shaw. Parallel random numbers: As easy as 1, 2, 3. In *SC11: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis*, 2011.

[HAU07] E. G. Haug. *The Complete Guide to Option Pricing Formulas*. McGraw-Hill, 2nd edition, 2007.

---

## License

MIT
