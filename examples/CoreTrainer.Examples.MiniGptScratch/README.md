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

- **Precision** (`Real` alias): `double`, `float`, or **`Fixed`** — a hand-built fixed-point
  type (a signed integer at scale `2^F`, i.e. "integer with a scale", the same idea as the
  ternary weights generalized to a full number type).
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

### Diagnostics worth stealing

Debugging fixed-point overflow taught a repeatable lesson: **make the code report the actual
values instead of theorizing.** The `Fixed` multiply is `checked` (a range violation throws
loudly rather than wrapping silently) and surfaces the operand *values* in the exception; an
unhandled-exception handler logs the max `|Raw|` reached before a crash. Together these turned
"it crashes somewhere" into "it's `-9M × -12M` in the Q·K product after one Adam step" — which
pinned the optimizer-precision bug immediately.

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
