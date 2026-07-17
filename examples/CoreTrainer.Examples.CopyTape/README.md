# CopyTape — isolating head-tracking, and the "usable tape" ceiling

A copy machine, built to answer one question the binary-adder experiment
([CoreTrainer.Examples.AdderTape](../CoreTrainer.Examples.AdderTape/)) raised: when a transformer executes
a sequential algorithm on its scratchpad, **can it keep track of *where it is* — and how far?**

```bash
dotnet run -c Release --project examples/CoreTrainer.Examples.CopyTape
```

---

## Why a copy machine

The adder cliffed on held-out length with *structural* failures — dropped record slots, lost alignment
between the generated step and the operand it should read — not arithmetic errors. The carry rippled
fine; what broke was **head-tracking**: "which cell am I on, advance the head." A Turing machine has an
explicit `(state, head-position)` control; a transformer has the state (it's on the tape, re-read each
step) but **no explicit head-position register** — position is reconstructed each step from content and
the causal mask.

Copy isolates that. Emit the input verbatim (or reversed, or two inputs interleaved) — **no computation
at all**, so a length-generalization failure can *only* be head-tracking. Three modes (config `Mode`)
form a discriminating sub-ladder:

- **plain** — forward copy (+1 head advance). Possibly solvable by an induction head without a real
  position sense (the "too easy" baseline).
- **reverse** — backward copy; must locate the end and walk back. Not a forward induction match.
- **zip** — interleave two inputs (`a0 b0 a1 b1 …`); two heads tracked at once.

Format uses single-character control tokens (`|`, `>`, `.`) so the char tokenizer doesn't fragment them,
and no spaces between bits (short tapes, so autoregressive reliability decay doesn't muddy the signal).
NoPE (no positional embeddings), same as AdderTape — see that README for why absolute positions would
invalidate a length-held-out test. Train on short lengths (4–12 bit), test on long (16/20/24 bit).

---

## Finding: head-tracking is real, degrades with distance, and its reliable range is *capped* (plain, 0.10 M model)

Training a tiny model (dim 64, 2 layers, 0.10 M params) on plain copy, run to convergence:

| epoch | train-len exec-acc | held-out 16b | 20b | 24b | loss |
|-------|--------------------|--------------|-----|-----|------|
| 500   | 67.5% | 0% | 0% | 0% | 0.099 |
| 1000  | 76.1% | 1% | 0% | 0% | 0.051 |
| 1500  | 88.4% | 1% | 0% | 0% | 0.027 |
| 2000  | 94.0% | 2% | 0% | 0% | 0.028 |
| 3000  | **99.4%** | 3% | 0% | 0% | 0.007 |

train-len **saturated at ~100%** while held-out stayed flat at ~2–4% (16b) and dead zero at 20/24b — the
plateau is confirmed, not undertraining. At 0.10 M params the reliable head-tracking distance is capped
below 16 bits.

Read the **failure dumps**, not just the numbers — they show the mechanism:

- **Early training**: held-out output is an input-*independent* `0101…` texture — right length, wrong
  content. The model learned "emit N alternating-ish bits" before learning to copy at all.
- **Later**: output *tracks the input for a prefix, then drifts*. E.g. `got 0111111001010010` vs
  `exp 0111111001010100` — 14 of 16 bits correct, tail wrong; or `got 1110101110110010` vs
  `exp 1110011111011000` — correct for ~4 bits, then decorrelates. The copy pointer holds locally and
  **loses alignment with distance.**

The diagnosis this supports:

1. **Head-tracking is not absent** — the model copies the first several bits correctly, so it *can* bind
   output position to input position.
2. **It degrades with distance** — the error compounds the further the copy runs, because with NoPE the
   alignment is *reconstructed* each step (no stored pointer), so small errors accumulate. On trained
   lengths (≤12 bit) the decay hasn't accumulated enough to break it; past them it drifts off and the
   tail is garbage. Since exec-accuracy needs *every* bit right, held-out reads ~0%.
3. **The reliable range is *capped*, not merely slow — confirmed at convergence.** As training proceeds,
   **train-len climbs to ~100% (67→94→99%) while held-out stays flat (0→2→3%)** and the train/held-out gap
   *widens*. That is the overfitting-to-trained-lengths signature, not the plateau-then-knee of a model
   still learning to generalize. Train-len saturating while held-out never lifts is the definitive form of
   the cap. (The caution against calling a plateau early — from the ternary-precision experiment's late
   knee — was honoured: this was declared only after train-len saturated.)

### The conclusion: NoPE did not remove the length limit — it rebuilt it

The hope behind NoPE was that removing the positional encoding removes the length ceiling baked into
RoPE/absolute schemes. **It doesn't.** With no positional signal, the model builds an *induction-like
attention circuit* to reconstruct "which input position am I aligned to" from content each step — and that
circuit has its own finite precision, which caps its range just as hard. The limit moved from an explicit
trained parameter (RoPE's rotation frequencies) into the *emergent attention dynamics*, where it is harder
to see and no less real. **The context-length problem isn't gone; it was rebuilt from scratch.** The
ceiling is a property of finite-precision *sequential position reconstruction*, not of any particular
positional encoding — which is why it can't be dodged simply by removing the encoding.

## What it means: reliable head-tracking distance *is* the effective context window

This is the reframing the experiment is really about. A model's advertised context length is the number
of tokens it can *attend over*. But for a **sequential algorithm on a tape**, what bounds the computation
is the number of steps over which the model can keep its head-position aligned — the **reliable
head-tracking distance**. Those are different numbers:

> A transformer can have a 128- (or 1M-) token window and still only be able to *execute* a step-by-step
> algorithm for ~14 steps before the head drifts. The advertised context is the tape it can *read*; the
> reliable head-tracking distance is the tape it can *compute on*. The second is smaller, and it is the
> real ceiling on the Turing-tape ladder.

Every rung above copy (addition, multiplication, a counter machine) needs *at least* this much
head-tracking, usually more (a counter machine moves the head both directions). So this cap is not a
copy-specific quirk — it is the load-bearing limit the whole ladder rests on. Copy is where we measure it
cleanly.

## The next question: does model size raise the cap?

If the reliable distance plateaus below the held-out lengths, the open question is whether **a larger
model tracks the head further**, and *why* it would — which distinguishes two hypotheses with opposite
consequences:

- **Capacity-bound** — 0.10 M params / dim 64 may simply lack the representational precision to encode a
  sharp position-count past ~14. If so, a wider/deeper model pushes the reliable distance out, and it
  **grows with parameters**. Optimistic: the tape idea works, it just needs capacity per unit tape.
- **Mechanism-bound** — NoPE reconstructs position via an induction-like attention pattern whose error
  compounds per step *regardless* of width. If so, size just **delays the drift** — the cliff moves
  further out but the shape is unchanged — and the fix is an explicit positional anchor (a relative
  encoding like RoPE, or a content-based moving cursor token), not more parameters.

**The experiment that decides it: a size sweep at fixed everything-else** — dim64/2L → dim128/4L →
dim256/6L — plotting **reliable head-tracking distance vs parameter count**. The *shape* of that curve is
the answer:

- grows ~linearly and keeps pace with length → capacity-bound; scaling is the lever.
- grows sub-linearly and asymptotes (same cliff, further out) → mechanism-bound; scaling won't save it,
  and the head needs an anchor.

Metric discipline for the sweep: watch the **held-out reliable distance** (the correct-prefix length in
the failure dumps), *not* train-len exec-accuracy — a bigger model will memorize trained lengths harder
and improve the wrong number.

If size *does* raise the cap, that is a concrete, quantified statement — "N parameters buys M steps of
usable tape" — which is exactly the kind of engineering result the concept-level "LLMs are Turing-complete
in the limit" literature never provides.

---

## Implementation notes

- Reuses the AdderTape transformer (plain full-precision GELU, NoPE), eval, and checkpointing verbatim;
  only the task differs. `Mode` (plain/reverse/zip), model shape, and the length split are in
  [appsettings.yaml](appsettings.yaml).
- **Per-log sample generation**: every `LogEvery` epochs one held-out example is generated and shown
  (got vs expected), for cheap qualitative feedback between the expensive full evals — this is where the
  `0101…`-texture-then-prefix-drift progression was first visible.
- **Per-length exec-accuracy** (16b/20b/24b separately) so the decay curve is visible, plus a
  failure dump of the longest failures (got vs exp) — the correct-prefix length in those is the reliable
  head-tracking distance, the key quantity.
- Checkpoint/resume (`CheckpointEvery`, `ResumeIfPresent`); a checkpoint is tied to the mode's vocabulary,
  so switching `Mode` needs a fresh start (delete `checkpoints/copy-tape.*`).
