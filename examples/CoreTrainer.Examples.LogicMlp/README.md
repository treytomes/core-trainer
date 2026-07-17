# LogicMlp — a differentiable logic-gate network

A network whose nodes **learn which Boolean gate they are**. Where a perceptron learns continuous
*weights* on a fixed sum, a node here learns a categorical *choice over the 16 two-input Boolean
gates* on a fixed pair of inputs — so its output is a bit, and the trained network is a discrete,
cullable logic circuit. It is the **dual** of [CoreTrainer.Examples.DiffNand](../CoreTrainer.Examples.DiffNand/),
and after Petersen et al., *"Deep Differentiable Logic Gate Networks."*

```bash
dotnet run -c Release --project examples/CoreTrainer.Examples.LogicMlp
```

|                | learns…            | fixes…             |
|----------------|--------------------|--------------------|
| **DiffNand**   | the wiring         | the gate (NAND)    |
| **LogicMlp**   | the gate function  | the wiring (random)|

Both train the same way (softmax + straight-through + temperature annealing) on the same tasks, so
they are directly comparable — and, as the results below show, each hits the *other's* ceiling.

---

## The mechanism: how a gradient selects a gate

Signals are **bipolar** (`-1` = false, `+1` = true), which is load-bearing, not cosmetic:

- Over `{-1,+1}` every 2-input gate has an exact **multilinear polynomial**
  `f(a,b) = α + β·a + γ·b + δ·(a·b)`, smooth on the cube and exact at the corners. All 16 gates share
  the basis `{1, a, b, ab}` and differ only in their coefficient vector (the Walsh–Hadamard transform
  of the gate's truth table), so evaluating all 16 is one small matmul.
- A bipolar signal carries a **sign**, so a node can argue *against* a feature — something `[0,1]`
  logic (like a raw soft-NAND) cannot do, and a likely reason `[0,1]` logic nets train poorly.

Each node holds **16 learnable logits**. Its soft output is the softmax-weighted blend of all 16 gate
polynomials; the gradient flows into the logits (a node's XOR-logit rises whenever "behave more like
XOR here" lowers the loss). **The learnable parameter is not inside a gate — it is the mixture *over*
gates.** On the forward pass a straight-through estimator commits to each node's argmax gate (so the
output *is* the discrete circuit it is scored on) while the gradient flows through the soft blend;
annealing the softmax temperature toward zero sharpens the blend so the two agree. This is the same
STE discipline DiffNand uses for wiring and BitNet uses for ternary weights, here applied to gate
identity.

### The one non-obvious bug: a uniform gate-mixture is a *dead* initialization

Initializing the 16 logits uniformly seems natural and **does not train at all** (loss frozen at
0.25, accuracy at chance). The reason is exact: at any input corner, 8 of the 16 gates output `+1`
and 8 output `-1`, so a uniform blend averages to **0 for every input** — signal dies after one layer
and the gradient vanishes through the depth. The fix is **passthrough-biased init**: bias every gate
toward the "A" passthrough (`f = a`) so the network starts as a stack of identity wires (the
logic-gate analogue of identity-initialized ResNets), signal propagates, and gates differentiate away
from passthrough into real logic during training.

---

## Results: a difficulty ladder, not one hard task

A single hard task is a poor proof of mechanism — the 4-bit adder's carry chain is hard for *every*
method here (the CGP example walls on it too), so its failure alone can't distinguish "the mechanism
is broken" from "the task is just hard." So LogicMlp runs a **ladder** that isolates one capability
per rung (see [LogicTask.cs](LogicTask.cs)). All numbers are the **hardened netlist's exhaustive
accuracy** (a hard forward pass over the entire truth table):

| rung | shape | bit-acc | exact rows | verdict |
|------|-------|---------|-----------|---------|
| bitwise XOR (4-bit) | 8 in → 4 out | 75% | 64/256 | partial |
| **parity (4-bit)** | 4 in → 1 out | **100%** | **16/16** | **SOLVED** |
| parity (8-bit) | 8 in → 1 out | 56% | 144/256 | partial |
| adder (2-bit) | 5 in → 5 out | 89% | 18/32 | partial |
| adder (4-bit) | 9 in → 9 out | 59% | 4/512 | partial |

### The finding: the wall is *routing*, not gate choice

The result inverts the naive difficulty ordering, and that inversion is the signal. **Parity-4 solved
perfectly, but bitwise-XOR-4 — supposedly the easiest rung — did not.** The gate mixes explain why:

- **Parity-4 (solved):** needs *one* XOR-reduction tree feeding *one* output. The net stayed 92%
  passthrough and found the handful of `XOR` gates it needed, in the right places.
- **Bitwise-XOR-4 (75%):** needs *four separate* XOR pairings, each routed to a *different* output,
  through **frozen random wiring**. It found only 3 `XOR` gates and stayed 76% passthrough — it could
  not win the wiring lottery four independent times.

The nodes genuinely learn to pick the right gate where the gradient asks (the mechanism works — parity
is a clean, exact solve). What fails is getting the right *signals to meet at* those gates and the
result to the right output, because **the one degree of freedom that would fix that — the wiring — is
exactly what LogicMlp freezes.** Parity-8, adder-2, and adder-4 all confirm it: each needs many
specific signals routed to many specific places, and fixed wiring can't deliver them.

So LogicMlp and DiffNand hit **complementary** ceilings: DiffNand learns wiring but is stuck with one
gate; LogicMlp learns gates but is stuck with one wiring. Neither degree of freedom alone is enough.

### Learning wiring *and* gates jointly ([JointLogicNet](JointLogicNet.cs))

The obvious next step — free both degrees of freedom (softmax over input wires from a cumulative skip
pool, *and* softmax over gate function, both STE-hardened) — gives a **mixed** result that is itself
informative. Running the same ladder for both architectures (hardened netlist accuracy):

| rung | gate-only | wire+gate |
|------|-----------|-----------|
| bitwise XOR (4-bit) | 75% | **100% SOLVED** |
| parity (4-bit) | **100% SOLVED** | 94% |
| parity (8-bit) | 56% | 47% |
| adder (2-bit) | 89% | 78% |
| adder (4-bit) | 59% | 66% |

The mechanism is **proven**: wire+gate *solves bitwise-XOR-4 exactly* — the very rung gate-only failed
on for lack of routing. Both degrees of freedom train. But adding learnable wiring made most other
rungs **worse**, and the soft-vs-hard gap is the tell: gate-only had soft ≈ hard everywhere, while
wire+gate's soft and hard diverge. Freeing the wiring adds a *second* coupled discrete argmax per gate,
and the optimization landscape got harder faster than the extra capacity helped. **We traded a routing
wall for an optimization wall.** bitwise-XOR was routing-limited (so it won big); the deeper tasks are
optimization-limited (so they regressed).

### The payoff: a logic-gate network as a transformer MLP

The standalone circuits demand 100% to be valid — but a transformer MLP does not. It is only a
per-token feature transform, smoothed by LayerNorm, the residual add, and the next attention layer, so
it needs to be a *useful nonlinearity*, not an exact circuit. Dropping the wire+gate logic network in
as the MLP of the [MiniGpt](../CoreTrainer.Examples.MiniGpt/) transformer (encode `Linear` → bipolar
quantize → logic layers → decode `Linear`, replacing `Linear → GELU → Linear`) and training the whole
model end-to-end on the memorization task gives:

| feed-forward | final loss | memorized |
|--------------|-----------|-----------|
| GELU (baseline) | 0.0217 | 5/5 |
| logic (soft) | 0.0218 | 5/5 |
| logic (hard, STE) | 0.0218 | 5/5 |

**All three are indistinguishable.** Two findings stand out:

1. **The logic MLP matches GELU** — a network that learns its wiring and gates is a viable transformer
   feed-forward, and the continuous↔bipolar encode/decode boundary (BitNet's activation-quantization
   idea) survives the round-trip.
2. **Hard matches soft here, though hardening *hurt* the standalone circuits.** End-to-end training
   lets the rest of the transformer adapt around the hard logic block — the Linear boundaries and
   surrounding layers absorb the STE commitment that broke isolated circuits. The forgiving,
   not-100%-required nature of a language task is exactly why the optimization wall above stops
   mattering here.

Caveats worth stating: this is *memorization* (5 sentences), not generalization or scale; the logic
modes are ~5× slower per epoch than GELU (the soft gate/wire blends are expensive — though a hardened,
culled deployment circuit would be cheap); and the cullability of the transformer's logic blocks has
not yet been measured.

### Cullability (the payoff shared with the ternary-weight work)

Every run's netlist is dominated by **passthroughs and constants** — 79–93% of gates are pure wiring
(`A`/`B`) or dead constants (`TRUE`/`FALSE`), leaving only ~6–20% doing real logic. Like the ternary
weights trained to exactly zero in [MiniGptScratch](../CoreTrainer.Examples.MiniGptScratch/), most of
a trained logic network is prunable structure: the knowledge lives in a small set of real-logic gates,
and the rest is routing that a better architecture would not need to spend gates on.

---

## What remains to be researched

The core convergence is now demonstrated: a learned-wiring, learned-gate logic network trains as a
transformer MLP at parity with GELU (see above). Open threads, roughly in order of promise:

1. **Measure cullability in the transformer.** Run the netlist report on the trained logic MLP blocks:
   how many gates are real logic vs passthrough/constant? If most are real logic, the "the MLP became
   cullable logic" claim is earned; if most are passthrough, the block is mostly wiring and the win is
   thinner. This is the first thing to check.
2. **Generalization and scale.** All results are 5-sentence *memorization*. Test a larger corpus
   (generalization, not overfitting) and larger models to see whether the GELU-parity holds.
3. **Fix the standalone optimization wall.** The joint wire+gate net regressed on deep tasks (an
   optimization, not capacity, problem — see the soft-vs-hard gap). Try a gentler/longer temperature
   anneal, longer soft warmup, or a wire-entropy penalty. Only matters if standalone exact circuits are
   a goal; the transformer use case already works without it.
4. **Extract and run the hardened MLP.** Since hard mode matches soft in the transformer, a trained
   model's logic blocks can be exported as discrete circuits and the culled circuit run directly —
   the cheap-deployment payoff. Verify the exported circuit is bit-identical to the hard-forward model.
