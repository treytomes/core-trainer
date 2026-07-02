# Mini-GPT from scratch

The same GPT-2-style transformer as **CoreTrainer.Examples.MiniGpt**, but built with **no
tensor library** — every matrix multiply, softmax, LayerNorm, attention pattern, their
**gradients**, and the Adam update are hand-written and steppable. It exists to reveal the
algorithms TorchSharp hides behind `.backward()` and `optimizer.step()`.

Before training, it runs a **gradient check** that compares the hand-derived analytic
gradients against numerical finite differences — proving the backward pass is correct.

```bash
dotnet run -c Release --project examples/CoreTrainer.Examples.MiniGptScratch
```

📖 **Full write-up covering both the library and from-scratch versions:**
[../CoreTrainer.Examples.MiniGpt/README.md](../CoreTrainer.Examples.MiniGpt/README.md)

---

## Beyond the tutorial: a numeric-precision testbed

Because there is no tensor library, the numeric type is *ours* — so the whole model can be
run at different precisions by flipping one alias in [RealType.cs](RealType.cs):

```csharp
global using Real = double;   // or System.Single (float), or the custom Fixed type
```

Everything downstream refers to `Real` and does its math through `RealMath`, so this single
line re-runs the entire transformer — weights, activations, **gradients, and Adam** — at the
chosen precision. This turns the example into an experiment about *how much numeric precision
a transformer actually needs*.

Two orthogonal levers, plus BitNet-style ternary weights, all composable:

- **Precision** (`Real` alias): `double`, `float`, **`Fixed`** (64-bit), or **`Fixed32`**
  (32-bit) — hand-built fixed-point types (a signed integer at scale `2^F`, i.e. "integer with a
  scale", the same idea as the ternary weights generalized to a full number type). `Fixed32`'s
  multiply maps onto a single native SIMD instruction (`vpmuldq` / `vmull_s32`), which is the
  whole reason to drop from 64 to 32 bits.
- **Ternary weights** (`UseBitLinear` in [appsettings.yaml](appsettings.yaml)): BitNet-b1.58
  `{-1, 0, +1}` block weights, trained with latent full-precision weights + straight-through
  estimator, exported by a "compile" step that packs 5 trits/byte (~1.6 bits/weight, ~40×
  smaller than double) and reports how many weights trained to exactly zero.
- **Strict 64-bit** (`Fixed.Int64Only`): forbid the `Int128` multiply intermediate, so all
  arithmetic stays within 64-bit integers — modeling real fixed-point hardware (MCUs, DSPs,
  FPGAs) that lacks an FPU and has at most a 64-bit integer multiply.

### Headline result: a transformer trained entirely in 64-bit integer arithmetic

With `Real = Fixed`, `Int64Only = true`, and `F = 20`, the model **trains from scratch and
generates text using nothing but 64-bit integer math** — no floating point in the linear
algebra (only `exp`/`sqrt`/`tanh` borrow `double` inside `RealMath`), no `Int128` cheat, no
FPU. It memorizes all five sentences and generates them verbatim. Most "fixed-point neural
net" work quantizes a *pretrained float* model for inference; here the entire training loop,
including backprop and Adam, runs in fixed-point.

### The fixed-point sweet spot (the `F` knob)

`F` (fractional bits) trades **grain** against **range**, and under `Int64Only` it must satisfy
both at once — because the doubled-scale product `a·b` (scale `2^2F`) must fit in int64, so the
representable multiply operand is bounded by `|value| < 2^((63-2F)/2)`:

| F  | grain   | multiply ceiling | outcome |
|----|---------|------------------|---------|
| 16 | 1.5e-5  | ~46,000          | **too coarse** — Adam updates round below the grain, weights freeze, loss stays ~2.9 (near-random) |
| **20** | **9.5e-7** | **~2,896**  | **✓ trains to 5/5 *and* generates** — the sweet spot |
| 24 | 6e-8    | ~181             | trains fine, but **overflows during generation** (autoregressive feedback exceeds the ceiling) |

Too many fractional bits and the value range is too small (generation overflows); too few and
the grain is too coarse to learn (weight updates vanish). F=20 threads both.

### The load-bearing detail: optimizer state must stay double

Adam's moment estimates are kept in `double` regardless of `Real` (see [AdamOptimizer.cs](AdamOptimizer.cs)).
This is not incidental. The second moment `v` is a running mean of *squared* gradients — for a
1e-3 gradient, `v ≈ 1e-6`. If `v` were stored at the model's grain (e.g. fixed-point F=16, grain
1.5e-5) it rounds to zero, and the next step's `mHat / (sqrt(v) + eps)` divides by ~`eps` and
explodes the weights by ~1e8. Keeping optimizer state at higher precision than the weights it
updates is exactly the discipline real mixed-precision training uses — and here it is the
difference between a working fixed-point transformer and one that diverges on the first step.

### Pushing to 32-bit: `Fixed32`

The natural next question after "it trains in 64-bit integers" is *how low can the bit-width
go?* `Fixed32` is the 32-bit sibling of `Fixed`: half the storage, and a multiply that fits a
native `int32×int32→int64` instruction. The catch is a far tighter range/grain budget — only 32
bits total, so `F` fractional bits leave `31−F` integer bits (`|value| < 2^(31−F)`).

Getting a transformer to train at 32-bit fixed-point required four techniques, each fixing a
*measured* failure (not a guessed one — see the diagnostics below). All are standard low-precision
training discipline, rediscovered here from first principles:

- **QK-normalization** — LayerNorm on Q and K before the attention dot product, bounding the
  logits so they cannot overflow the small integer range (the same trick Gemma-2 uses). Without
  it, attention scores grew to fill the range and overflowed.
- **Masked softmax** (`SoftmaxRows` with per-row valid counts) — the causal mask skips future
  cells by *count* instead of writing a `−∞` sentinel, removing the "sentinel minus row-max"
  subtraction that overflowed the low range.
- **Gradient clipping at the source** — LayerNorm's backward multiplies by `1/std`, which spikes
  when a normalized row has near-zero variance. Clamping each `LayerNormBackward` output to a
  fixed limit *before* it enters a residual `Add` (not after — the spiked operand overflows
  inside the Add itself) contains it. Inert in double/float, where gradients sit well within range.
- **Wide-accumulator dot products** (`Mac`) — "compute low, accumulate high": dot products sum
  their `int32×int32` products in a 64-bit register at doubled fractional width (`F=32`) and
  narrow to the model grain *once*, instead of rounding every product before adding. This is the
  textbook fp16-storage / fp32-accumulate rule; it is inert for double/float.

**Result (as of this writing): partial.** With QK-norm + masked softmax + clipping + MAC:

| F  | grain  | range   | outcome |
|----|--------|---------|---------|
| 16 | 1.5e-5 | ±32,768 | loss descends to ~0.6 then **thrashes** (0.6 ↔ 2.0); 0/5 memorized |
| 18 | 3.8e-6 | ±8,192  | loss reaches ~0.07 but still **bounces**; **2/5** memorized — near the threshold |

The clean **double control** (same seed, same LR=0.0005) memorizes 5/5 at loss 0.022, which
proves the config is sound and the fault is genuinely `Fixed32`'s arithmetic. Finer grain (F=16→18)
helped substantially, which pins **storage grain** — not accumulation, not update resolution — as
the dominant remaining factor. Full 5/5 at 32-bit is **unresolved** (see *What remains*, below).

### The load-bearing detail #2: gradient *direction*, not magnitude, is what degrades

A satisfying dead-end worth recording. Two natural theories for the F=16 thrashing — *(a)* Adam's
updates round below the grain and freeze, *(b)* the incoming gradients round below the grain — were
both **disproven by direct measurement.** An update/gradient grain histogram (`AdamUpdateStats`,
logged each epoch as `frozen / ±1grain / resolvable`) showed gradients stayed ~99% resolvable and
updates stayed ~85% resolvable *while the loss climbed*. The histograms measure **magnitude**; the
corruption is in **direction** — thousands of tiny `F=16` rounding errors compounding through the
forward/backward pass so the net gradient points slightly wrong, which Adam then faithfully
normalizes into a confident wrong-way step. (Amusingly, the double control revealed that
update-collapse to the grain floor is the signature of *healthy convergence*, not failure — near a
minimum the true update genuinely is sub-grain.)

### Diagnostics worth stealing

Debugging fixed-point overflow taught a repeatable lesson: **make the code report the actual
values instead of theorizing.** The `Fixed` multiply is `checked` (a range violation throws
loudly rather than wrapping silently) and surfaces the operand *values* in the exception; the
`Fixed32` operators do the same via an explicit int64 range test; an unhandled-exception handler
logs the max `|Raw|` reached before a crash. Together these turned "it crashes somewhere" into
"it's `-9M × -12M` in the Q·K product after one Adam step" — which pinned the optimizer-precision
bug immediately. The same discipline (measure, don't theorize) resolved the F=16 thrashing above:
three plausible theories died to the grain histogram and the double control before the real cause
(direction error from accumulated rounding) was identified.

Each run writes its own timestamped log under `logs/mini-gpt-scratch-<yyyyMMdd-HHmmss>.log`
(configured in `Program.cs`), so successive experiments no longer stack into one file.

### Other performance levers (for scaling up)

- **Add-only ternary matmul**: since ternary weights are `{-1,0,+1}`, the matmul needs no
  multiplies — add on `+1`, subtract on `-1`, skip `0`, scale by γ once at the end. Skips the
  (often majority) zero weights *and* avoids the expensive fixed-point multiply. Used in two of
  the three matmuls per layer (forward, and the input-gradient backward).
- **Parallel attention heads** (`ParallelHeadThreshold`): heads are independent, so the head
  loop parallelizes; gated on a work threshold since threading tiny heads costs more than it
  saves. Its payoff grows with head count and sequence length.
- **Cached quantization**: a `BitLinear`'s ternarized/transposed weights are recomputed once per
  optimizer step (when weights change), not once per forward pass.

---

## What remains to be researched

The project is parked here in a working state (64-bit fixed-point trains 5/5; 32-bit is close but
not there). Open threads, roughly in order of promise:

1. **Close the 32-bit gap.** F=18 reaches 2/5 and F=16 fails, so the lever is grain, not range or
   accumulation. Two untried moves:
   - **`Fixed32` at F=20** (grain 9.5e-7, the value that works in 64-bit) — but range shrinks to
     ±2048, so the QK-norm / clipping guards would be under real pressure. Worth trying now that
     those guards exist; watch for overflow.
   - **Convert the deferred backward scatter-adds** (`dV`, `dQn`, `dKn` in
     [ScratchSelfAttention.cs](ScratchSelfAttention.cs)) to wide accumulation. They accumulate
     across an outer loop into shared cells and were left at `Real` precision when the `Mac` fix
     went in (they need per-head wide buffers). If direction-error from *these* sums is
     significant, cleaning them may be what tips F=18 over to 5/5. Measure first.
   - **A `double` master-weight copy** in the optimizer (accumulate the full-precision update,
     round to `Fixed32` for forward/backward). Standard mixed-precision; not yet needed given the
     grain finding, but the obvious fallback if finer `F` alone can't close it.

2. **Isolate the nonlinearity tables.** GELU/softmax/`tanh` run through `F`-grain lookup tables
   (`Fixed32Math`, `Fixed32LookupTable`). Their contribution to the direction error is unmeasured;
   a run with double-precision transcendentals but `Fixed32` everywhere else would bound it.

3. **Scale the corpus.** All results are on 5 memorized sentences. Whether the fixed-point findings
   hold on a larger corpus (generalization, not memorization) is untested.

4. **Confirm the SIMD payoff.** `Fixed32`'s reason to exist is the native `int32×int32→int64`
   multiply. The wall-clock win over `Fixed`/`double` has not been benchmarked; the matmul is the
   place to measure it.

5. **A different perceptron.** The ternary-weight work points at a broader question the sibling
   `LogicMlp` example takes up: replace the MLP's linear-sum-plus-nonlinearity with a trainable
   **logic network** (bipolar multilinear gates, learned by softmax-over-gate-types + STE), so the
   knowledge lives in *which operation* each unit computes rather than *how much* of each input —
   natively discrete and cullable, the logical endpoint of "weights trained to exactly zero."
