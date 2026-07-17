# AdderTape — the context window as a Turing tape

A transformer trained to execute a sequential algorithm **step by step on its own scratchpad**, tested
by holding out *length*. The first rung of a ladder toward demonstrating universal computation on a
bounded tape.

```bash
dotnet run -c Release --project examples/CoreTrainer.Examples.AdderTape
```

---

## The question this answers

A fixed-depth transformer is a *bounded* computer: one forward pass composes O(depth) sequential steps,
no more. Formally it sits around **uniform TC⁰** (constant-depth threshold circuits), and exact integer
addition's **carry chain** — bit *k*'s carry depends on every lower bit — is an inherently *sequential*
scan that does not fold into constant depth. That is exactly why the parallel adder was so hard to train
in the sibling logic-network experiments (`CoreTrainer.Examples.LogicMlp`, `DiffNand`): we were asking a
constant-depth machine to do a sequential job.

Autoregressive generation breaks that ceiling. Each emitted token is a fresh forward pass that reads
**everything written so far**, so the computation is `(bounded per-step transducer) × (unbounded number
of steps)`, with the context window holding the intermediate state — a bounded control head plus a tape.
The carry chain that is brutal to parallelize is trivial to *ripple* when you have one step per token.

**So: does a transformer given a scratchpad actually execute the algorithm, or just approximate it?**

## The experiment

Train a small GPT on binary-addition **tapes** — the full ripple-carry derivation written out — then test
whether it generalizes to **unseen lengths**. Because longer inputs never appear in training, the only
thing that can transfer is the algorithm itself; memorization is ruled out by construction. Train on
1–4-bit additions; test on 5–8-bit.

**NoPE (no positional embeddings).** The model has token embeddings only; the causal mask supplies the
relative structure. This is essential to the test's validity: with learned absolute positions, positions
beyond the training length would be untrained, so long tapes would fail for *positional* reasons rather
than algorithmic ones. NoPE also length-generalizes best for decoder-only models (Kazemnejad et al.,
2023). The ripple *step* is position-invariant — "add these two bits plus the carry" is the same at bit 2
or bit 20 — so absolute position is not just unnecessary, it is a distractor. (But see the finding below:
NoPE removes *chain* position, which is correct; the format must then supply *within-chunk* structure by
tape locality, not by an index.)

### Two metrics, measuring different things

- **Teacher-forced validation loss** (cheap, one batched forward pass, every `LogEvery`) — next-token loss
  on held-out lengths *given the ground-truth prefix*. A generalization hint, but **misleading alone**: a
  model can score low val-loss yet 0% execution accuracy (good at "next token given the truth", unable to
  ripple unaided). We measured exactly this — val-loss ≈ 0.2 while free-generation held-out was near zero.
- **Free-generation execution accuracy** (expensive, autoregressive, every `EvalEvery`) — from the prompt
  only, the model generates the whole tape itself; we parse the answer from its output and check it equals
  a+b. This is the honest test. Reported **per bit-length** (5b/6b/7b/8b) so the length-decay curve is
  visible, not just an aggregate.

## What we found: the failure is *within-chunk position*, and it is a format problem

The first format wrote the operands as a header block and a separate `sum` line:

```
a 1 1 0        ← operand A, indexed: bit k is at position k of this block
b 1 0 0
work
a1 b1 c0 s0 k1
...
sum 0 0 1 0    ← answer gathered into a positional line
```

Trained on 1–4 bits, it learned trained lengths **perfectly** (train-length execution accuracy 100%) but
the held-out curve was a **cliff, not a slope**:

| bits | 5 | 6 | 7 | 8 |
|------|---|---|---|---|
| exec-acc | 55% | 8% | 4% | 0% |

It generalized *one* step past training, then collapsed. The failure dumps (see below) showed **three**
breakages, and none of them was the carry step itself — the carry rippled fine. They were all operations
that require *position within the current problem*:

1. **Wrong operand bit read** — at step *k* the model must random-access bit *k* of the header block; it
   grabbed the wrong bit.
2. **Wrong step count** — it emitted 7 work lines for an 8-bit number; it cannot *count* to the width to
   know when to stop.
3. **Garbage sum gather** — assembling the `sum` line scatters each digit to slot *k* by position; on long
   tapes this dissolved into contradictory partial sums.

The distinction that resolves it: there are **two kinds of position**, and only one should be removed.

- **Chain position** (which step am I on, the 3rd or the 30th?) — the step must be *invariant* to this.
  NoPE removes it. Correct — that is what lets the algorithm length-generalize.
- **Within-chunk position** (this is bit *k* of the operand; this digit belongs in slot *k*) — the model
  *needs* this, and NoPE stripped it too. That is what broke.

**A Turing head never has this problem, because it never random-accesses.** It reads what is *adjacent*
and consumes the tape *in order*. The first format was subtly *not* tape-like: it forced random access
(reach back to bit *k*), counting (how many bits?), and scatter-gather (the sum line). So the fix is not a
bigger model or positional encodings — it is **making the tape behave like a tape.**

### The streaming tape format (`AdderTape.Render`)

Operands are a **queue consumed one pair per step**; each record echoes the pair it read; the answer is
read straight from the per-step `s` digits (no separate sum line); generation stops when the input runs
out (no counting). For a=3, b=1, 3-bit, LSB-first:

```
1 1        ← operand pairs, one per line — a queue, not an indexed block
1 0
0 0
go
1 1 c0 s0 k1     ← each record: the pair just consumed, carry-in, sum digit, carry-out
1 0 c1 s0 k1
0 0 c1 s1 k0
end 0            ← final carry-out (the high sum bit); then '.' terminates
.
```

Every operation is now a **fixed local offset**: read the current pair (adjacent), read the previous
record's carry-out as this record's carry-in (offset −1), emit. Each old failure mode is removed *by
construction* — no random access (operands echoed in place), no counting (stop when pairs run out), no
gather (answer *is* the `s` digits in order). The residual positional task is only the *streaming*
correspondence "generated record ↔ next input pair", which is local, not indexed. **This is the current
experiment; whether it clears the cliff is the open question.**

### Reading the result

Read the **per-length curve** (5b→8b), not just the aggregate — the shape is the finding:

| shape | meaning |
|-------|---------|
| flat-high across lengths | **Thesis holds** — executes the position-free algorithm on lengths it never trained on. |
| graceful decay (e.g. 5b:90% → 8b:60%) | The algorithm transfers but per-step reliability compounds over length (0.99ᴺ decay) — *extended-depth*, real but decaying. |
| **cliff** (high then ~0 one step past training) | Bounded-depth memorization + a step of slack — what the *first* (indexed) format did. |
| low even on train lengths | Didn't learn the training lengths — more epochs/capacity. (Beware the deceptive plateau-then-knee seen in the ternary experiments; don't call it early.) |

**The most informative move is to find where it breaks.** Push the test length until generalization fails;
that boundary distinguishes *extended-depth* from genuinely *length-invariant* execution, and is more
informative than the success itself.

---

## Implementation notes

- Plain full-precision GELU transformer (this tests **training**, not architecture — nothing exotic).
- **Batched generation** grouped by bit-width (equal prompt lengths batch together): the exec-accuracy
  eval is one forward pass per step over a whole width-group, not per sequence (~15 min → ~1.5 min).
- **Uniform-random pair sampling** (seeded, per width): an earlier *strided* sample biased held-out toward
  low-carry cases and inflated the number — a reminder to check that the eval set is representative, not
  just large. Train and held-out use independent seeds.
- **Per-length exec-accuracy breakdown** so the length-decay curve is visible; a **failure dump** prints
  the most carry-heavy failures (with the generated tape) so *where* generation breaks is inspectable —
  this is how both eval bugs and the position-vs-format finding were caught. A **format self-check** at
  startup asserts `Render` round-trips through `ParseSum` (a metric that can't parse its own ground truth
  is worthless).
- **Checkpointing** (`CheckpointEvery`, `ResumeIfPresent`): saves model + Adam moments + epoch counter,
  resumes automatically — so a long run can be stopped and continued, and bumping `Epochs` and continuing
  is free. (Note: a checkpoint is tied to the tape format's vocabulary; a format change requires a fresh
  start.)
- Config in [appsettings.yaml](appsettings.yaml); the corpus is generated in-process, not read from a file.
  `TrainBits` / `HeldOutBits` are the disjoint length split.

## Debugging lessons worth keeping

Two eval bugs produced *plausible-looking but meaningless* numbers before the metric was trustworthy —
both caught by dumping the model's actual generated tapes rather than trusting the aggregate:

1. Generation stopped at the first newline (a multi-line tape needs no stop token) → exec-acc stuck at 0%.
2. The answer parser took the **last** `sum` line, but past its budget the model starts a *second* tape
   with a garbage sum → exec-acc read ~3% while the model had actually memorized the training set perfectly
   (train-loss 0.0002). A near-zero exec-acc next to a near-zero loss is a *contradiction* that flags a
   broken metric, not a learning failure.

The throughline: **make the code report the actual values.** A dump of real generated tapes turned "it's
not learning" into "the eval is lying, and here's the exact line," twice.

---

## The ladder: toward a bounded-tape universality demonstration

Addition is rung one. The point is not to accumulate algorithms for their own sake but to climb toward
executing a formalism that is *already known to be universal*, each rung adding one computational primitive:

1. **Addition** *(this experiment)* — sequential state (the carry) + a local update. *Can it hold and
   propagate state across steps, on unseen lengths?*
2. **Multiplication** — nested iteration (repeated add-and-shift). *Can the tape drive a loop within a
   loop?*
3. **Comparison / sorting** — data movement and conditional action. *Can it read state and branch?*
4. **A 2-counter (Minsky) machine** — the real target. Its entire instruction set is `increment`,
   `decrement`, `branch-if-zero`, and it is **Turing-complete**. If the transformer can execute an
   *arbitrary* 2-counter program (the program supplied in the prompt) for arbitrary step counts, that is
   not "another algorithm" — it is universal computation, because any computation reduces to one. This is
   a single demonstration that subsumes the pile of individual algorithms.

### Honest claim ceilings (stated up front so we don't over-claim)

- **This is not a proof of Turing-completeness.** That is an infinite claim (unbounded tape, unbounded
  steps); it cannot be established empirically. What is demonstrable is *faithful execution of a
  known-universal formalism up to resource limits.*
- **Finite context = finite tape.** A transformer with a fixed window is formally closer to a
  **linear-bounded automaton** (context-sensitive) than a Turing machine. The *last* rung of the ladder is
  therefore not another algorithm but a **tape-extension mechanism** — scrolling the context, external
  memory, or a "tape ran out, continue" protocol — so the tape can grow.
- **Per-step reliability compounds.** Even a correct step function at 99% per-step accuracy fails over
  enough steps (0.99ᴺ → 0). Where length-generalization breaks may be reliability decay, not a
  representational wall — worth distinguishing when it happens.
- **We hand the model the algorithm** (the scratchpad scaffold), and that is deliberate, not a limitation
  to apologize for: a Turing machine is *defined* by executing a supplied transition function. This tests
  *execution* (the computability-relevant question), not *discovery* (a separate learning-theory question).

The defensible destination: **"a transformer can execute universal computation over bounded tapes, and the
boundedness is a resource limit, not an architectural one."**

## Relation to the rest of the project

This reframes the logic-network thread's central negative result. The parallel adder was hard because we
demanded *parallel depth* for a *sequential* problem, and we tried to put it in the position-wise MLP — a
*combinational* block, when a carry is inherently sequential. The tape puts the sequence where it belongs
(across tokens). The threads rejoin at rung 4: a small learned **logic network** could supply the trivial
per-step transition function (the kind of shallow local circuit `LogicMlp` *can* nail — parity solved
cleanly; it was parallel routing that defeated it), while the **tape** supplies the sequential iteration.
