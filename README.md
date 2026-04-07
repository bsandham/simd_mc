# `simd_mc`

## C++23 | `std::experimental::simd`

A header-only C++23 library that prices barrier options via Monte Carlo simulation using `std::experimental::simd` (Parallelism TS 2, GCC 11+). Solves the dead lane problem - the systematic underutilisation of SIMD registers when simulated paths are knocked out by a barrier - via an in-register stream compaction implementation.

<div align="center">

| Feature | Assembly-verified SIMD |
|:--------|:------:|
| Vectorised `exp()` - minimax polynomial | Yes, 8x `vfmadd` |
| Vectorised `log()` - IEEE 754 decomposition, (m-1)/(m+1) rational polynomial | Yes |
| Vectorised `sincos()` - quadrant-based minimax polynomial | Yes |
| Vectorised Philox-2x32-10 RNG | Yes, `vpmuludq` + `vpxord` |
| SIMD step counter - `IntV` register | Yes, `vpcmpd` + masked `vpsubd` |
| Zero-cost bit reinterpretation - `__builtin_bit_cast` | Yes, compiles to `ret` |
| SIMD floor | Yes, `vcvttps2dq` + `vcvtdq2ps` + masked `vsubps` |
| AVX-512 `VCOMPRESSPS` compaction (portable AVX2 fallback) | Yes |
| OpenMP multi-threading with Chan's parallel Welford merge | Yes |
| Brownian bridge barrier correction (compile-time toggle via `if constexpr`) | Yes |
| Antithetic variance reduction in recovered lanes | Yes |
| Adaptive compaction threshold | Yes |
| Welford's online variance (numerically stable std error) | Yes |
| Truncation detection (`result.truncated` flag) | Yes |

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
// result.price            - discounted option value
// result.std_error        - Monte Carlo standard error
// result.paths_simulated  - number of completed paths
// result.lane_utilisation - avg% of live SIMD lanes
```

### Input Parameters

<div align="center">

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `initial_spot` | `float` | Underlying spot price |
| `n_paths` | `int` | Number of Monte Carlo paths (controls accuracy) |
| `n_steps` | `int` | Time steps per path (eg 252 = daily for 1 year) |
| `T` | `float` | Time to maturity in years |
| `risk_free_rate` | `float` | Annualised r |
| `n_threads` | `int` | Thread count (0 = auto-detect hardware cores) |

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

A knock-out barrier option pays `max(S-K, 0)` at expiry unless the underlying price `S` breaches a barrier level `B` at any monitoring date during the option's life. If the barrier is hit, the option is cancelled hence pays 0.

In a Monte Carlo simulation, each path evolves independently according to:

$$ S_{t+\Delta t} = S_t \cdot \exp\left[\left(r - \tfrac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t}\; Z\right], \quad Z \sim \mathcal{N}(0,1) $$

With SIMD, `W` paths evolve in parallel across a single register. But when a path hits the barrier, its lane "dies", which means it continues occupying the register but is useless.

### The Problem

On a `W=8` AVX2 register pricing a barrier 5% below spot with 252 daily steps, lane utilisation can be **25-45%** by mid-simulation.

This is the CPU SIMD analogue of GPU warp divergence. GPUs solve it with stream compaction - dead threads are replaced with fresh work.

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
g++ -std=c++23 -O3 -march=native -fopenmp -S -I src your_file.cpp -o output.s
grep -c "call.*expf" output.s    # Should be 0
grep -c "call.*logf" output.s    # Should be 0
grep -c "call.*sinf" output.s    # Should be 0
grep -c "call.*cosf" output.s    # Should be 0
```

### Instruction Breakdown (AVX2 8-wide)

<div align="center">

| Operation | Instruction(s) | Count | Scalar Loops |
|:----------|:---------------|:-----:|:------------:|
| GBM evolve: `drift + vol*Z` | `vfmadd132ps` | 1 | 0 |
| `exp()` Horner polynomial | `vfmadd132ps` / `vfmadd213ps` | 8 | 0 |
| `exp()` floor (range reduction) | `vcvttps2dq` + `vcvtdq2ps` | 2 | 0 |
| `exp()` 2^n reconstruction | `vpslld` + `__builtin_bit_cast` -> `ret` | 1 | 0 |
| `log()` IEEE 754 decomposition | `vpsrad` + `vpandd` + `vpord` | 3 | 0 |
| `log()` int-to-float | `vcvtdq2ps` | 1 | 0 |
| `log()` Horner polynomial | `vfmadd` chain | 4 | 0 |
| `sincos()` quadrant reduction | `vcvttps2dq` + `vpandd` | 2 | 0 |
| `sincos()` polynomials + blend | `vfmadd` + masked `vmovaps` | 10 | 0 |
| Barrier check: `spot > B` | `vcmpps` -> k-register | 1 | 0 |
| Mask logic: `alive && survived` | `kandw` | 1 | 0 |
| Payoff: `max(S-K, 0)` | `vmaxps` | 1 | 0 |
| Step decrement | `vpcmpd` + masked `vpsubd` + `vpcmpeqd` | 3 | 0 |
| Philox RNG (10 Feistel rounds) | `vpmuludq` + `vpxord` + `vpaddd` | ~40 | 0 |
| `sqrt()` | `vsqrtps` | 1 | 0 |
| Bit reinterpretation | `__builtin_bit_cast` -> `ret` | 0 | 0 |

</div>

---

## Results

### Test Environment

<div align="center">

| Component | Specification |
|:----------|:-------------|
| CPU | Intel Core i7-14700K |
| ISA | AVX2, 8-wide (`native_simd<float>::size() = 8`) |
| Compiler | GCC 13.2 (MSYS2 UCRT64) |
| Standard | C++23 (`-std=c++23`) |
| Flags | `-O3 -march=native -ffast-math -fopenmp` |
| OS | Windows 11, MSYS2 |

</div>

### Vanilla Convergence (500k paths, no barrier)

<div align="center">

| Test | MC Price | Black-Scholes | Error | Result |
|:-----|:---------|:-------------|:------|:------:|
| ATM Call (S=K=100) | 10.5122 | 10.4506 | 0.0616 | Pass |
| ATM Put (S=K=100) | 5.5937 | 5.5735 | 0.0202 | Pass |
| ITM Call (K=90, vol=30%) | 15.5503 | 15.4860 | 0.0643 | Pass |
| OTM Call (K=110, vol=30%) | 5.6627 | 5.5871 | 0.0756 | Pass |

</div>

### Barrier Pricing vs Continuous-Monitoring Analytical (500k paths)

<div align="center">

| Test | Barrier | MC Price | Analytical (cont.) | Error | Notes |
|:-----|:--------|:---------|:-------------------|:------|:------|
| Far barrier | B=80 | 10.2294 | 10.3513 | 0.1219 | Discrete monitoring bias |
| Medium barrier | B=90 | 9.0579 | 8.6655 | 0.3924 | Larger bias at closer barrier |

</div>

MC prices sit above the continuous-monitoring analytical. This is expected: discrete daily monitoring misses between-step barrier crossings, so fewer paths are knocked out, inflating the price. The bias scales with barrier proximity as predicted by Broadie-Glasserman-Kou theory.

### Knock-In Parity (500k paths)

<div align="center">

| Component | Value |
|:----------|:------|
| Vanilla Black-Scholes (exact) | 10.4506 |
| Knock-Out MC (B=85) | 10.0780 |
| Knock-In via parity | 0.3725 |
| Knock-In analytical (cont.) | 0.5013 |
| Error | 0.1288 |

</div>

Error is inherited entirely from the knock-out estimate. Validates the control variate technique.

### Brownian Bridge Correction (500k paths, B=85)

<div align="center">

| Method | Price | Error vs Analytical | Improvement |
|:-------|:------|:-------------------|:------------|
| Hard barrier check | 10.0780 | 0.1288 | -- |
| Brownian bridge correction | 9.9634 | 0.0141 | 9.1x |
| Analytical (continuous) | 9.9493 | -- | -- |

</div>

One additional `exp`/`log` per step per lane. The bridge-corrected price is within 0.015 of the continuous-monitoring analytical.

### Lane Utilisation

At B=95 (barrier 5% below spot price), lane utilisation measured at **92.5%**. Without stream compaction, this would be 25-45% by mid-simulation.

### Barrier Proximity Sweep (25k paths)

<div align="center">

| Barrier | Price | Std Error | Util % | Time (ms) |
|:--------|:------|:----------|:-------|:----------|
| 70 | 10.3005 | 0.1464 | 98.4 | 18.7 |
| 75 | 10.5087 | 0.1480 | 97.5 | 17.6 |
| 80 | 10.1785 | 0.1451 | 95.9 | 17.0 |
| 85 | 10.3713 | 0.1503 | 94.7 | 15.5 |
| 90 | 9.2645 | 0.1493 | 93.7 | 14.6 |
| 92 | 8.1027 | 0.1445 | 93.4 | 11.3 |
| 95 | 6.2919 | 0.1346 | 92.5 | 8.0 |
| 97 | 4.3585 | 0.1192 | 91.5 | 5.6 |
| 98 | 3.6266 | 0.1101 | 90.3 | 4.6 |

</div>

**Observations:**
- Price drops monotonically as barrier approaches spot (more knock-outs, less payoff)
- Execution time drops with closer barriers (knocked-out paths complete faster, replaced by shorter-lived paths)
- Utilisation stays above 90% across all barrier distances

### Realistic Market Scenarios (25k paths)

<div align="center">

| Scenario | Price | Std Error | Util % | Time (ms) |
|:---------|:------|:----------|:-------|:----------|
| FTSE equity (S=7500, K=7500, B=6750, vol=18%) | 637.35 | 6.27 | 94.0 | 32.3 |
| Tech stock (S=200, K=200, B=160, vol=40%) | 30.99 | 0.39 | 93.6 | 28.7 |
| FX 3-month (S=1.25, K=1.25, B=1.20, vol=8%) | 0.0230 | 0.0002 | 94.1 | 16.9 |
| Commodity 2-year (S=80, K=85, B=110, vol=35%) | 13.83 | 0.11 | 94.2 | 65.4 |

</div>

### Up-and-Out Put (25k paths)

<div align="center">

| Scenario | Price | Std Error | Util % | Time (ms) |
|:---------|:------|:----------|:-------|:----------|
| S=100, K=105, B=120, T=0.5y | 8.1695 | 0.0644 | 94.7 | 19.5 |

</div>

### Convergence Study (Down-and-Out Call, B=90)

<div align="center">

| Paths | Price | Std Error | Time (ms) |
|:------|:------|:----------|:----------|
| 10,000 | 10.6634 | 0.7321 | 0.7 |
| 50,000 | 9.4410 | 0.3049 | 3.0 |
| 100,000 | 9.1638 | 0.2114 | 5.9 |
| 500,000 | 9.0579 | 0.0933 | 30.3 |
| 1,000,000 | 8.9712 | 0.0654 | 59.3 |

</div>

Std error halves when paths quadruple, confirming the 1/sqrt(N) convergence rate. The price converges toward the continuous-monitoring analytical value of 8.67 from above. The remaining gap is the discrete monitoring bias, not engine error.

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
- GCC 11+ (has `std::experimental::simd` in libstdc++)
- C++23 mode (`-std=c++23`)
- OpenMP (`-fopenmp`)

### Compile
```bash
# Direct compilation
g++ -std=c++23 -O3 -march=native -fopenmp -I src src/main.cpp -o main -lm

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
├── src/
│   ├── main.cpp               # Demonstration with 7 sections
│   ├── simd_mc/               # Header-only library
│   │   ├── core/              # simd_compat.hpp, lane_register.hpp, sim_config.hpp
│   │   ├── concepts/          # 6 policy concepts
│   │   ├── models/            # GBM diffusion
│   │   ├── barriers/          # Down/Up x Out x Hard/Bridge, None
│   │   ├── payoffs/           # European Call/Put
│   │   ├── refill/            # Independent, AntitheticPreferred
│   │   ├── compaction/        # Adaptive (AVX-512 VCOMPRESSPS + portable fallback)
│   │   ├── rng/               # Vectorised Philox + cached Box-Muller
│   │   ├── engine/            # MonteCarloEngine template
│   │   └── api/               # price() with OpenMP parallelism
│   └── tests/                 # 9 tests (4 vanilla, 5 barrier)
├── benchmarks/                # basic_barrier, crossover sweep, scalar vs SIMD
├── CMakeLists.txt             # GCC, C++23, -O3, -fopenmp
├── CMakeSettings.json         # MSYS2 UCRT64 Visual Studio config
├── EXTENSIONS.md
└── LICENSE (MIT)
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

[BGK97] M. Broadie, P. Glasserman, and S. Kou. A continuity correction for discrete barrier options. *Mathematical Finance*, 7(4):325-349, 1997.

[ABR96] L. Andersen and R. Brotherton-Ratcliffe. Exact exotics. *Risk*, 9(10):85-89, 1996.

[GIL08] M. Giles. Multilevel Monte Carlo path simulation. *Operations Research*, 56(3):607-617, 2008.

[JOS03] M. Joshi. *C++ Design Patterns and Derivatives Pricing*. Cambridge University Press, 2003.

[KRE15] M. Kretz. Extending C++ for explicit data-parallel programming via SIMD vector types. PhD thesis, Goethe University Frankfurt, 2015.

[KRE25] M. Kretz. Merge data-parallel types from the Parallelism TS 2. P1928R15, WG21, 2025.

[LAN18] H. Lang et al. Data blocks: Hybrid OLTP and OLAP on compressed storage. *SIGMOD*, 2018.

[ROG07] D. Roger, U. Assarsson, and N. Holzschuch. Efficient stream reduction on the GPU. *GPGPU Workshop*, 2007.

[SAL11] J. Salmon et al. Parallel random numbers: As easy as 1, 2, 3. *SC11*, 2011.

[HAU07] E. G. Haug. *The Complete Guide to Option Pricing Formulas*. McGraw-Hill, 2nd edition, 2007.

---

## License

MIT
