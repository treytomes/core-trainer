# HeadTrack RUNBOOK — overnight width sweep, and how to pick it up in the morning

**Status at end of 2026-07-13:** built, sanity-checked, ready to run overnight. Not yet run to completion.

---

## What this experiment is (context recovery)

We are building toward a **scrollable-tape Turing machine**: transformer = the finite control, an external
buffer = the tape, the attention window (size ≤ **D**) slides over it via one-cell left/right head moves.
The whole design rests on one primitive — *within a window, can the model track a head through L/R moves
and read the cell under it?* — and on a quantity, **D = the reliable head-tracking distance** (how many
moves it can track before drifting).

Prior results that led here (see [../CoreTrainer.Examples.CopyTape/README.md](../CoreTrainer.Examples.CopyTape/README.md)
and [AdderTape](../CoreTrainer.Examples.AdderTape/README.md)): a NoPE transformer's reliable head-tracking
distance is **capped** and, at 0.10 M params, doesn't reach 16 tokens. Key finding: **NoPE didn't remove the
context-length limit, it rebuilt it in the attention dynamics.** So D is finite — the question is whether it
**scales usefully with model size**. If D grows well with params, the scrollable-tape machine is practical
(N params buys M cells of usable tape); if D ~ constant or log, it's not, and the head needs an explicit
anchor rather than more parameters.

**This run measures D vs model width (EmbedDim)** — the first of several planned sweeps.

## What is being swept, and what is pinned

- **Swept:** `EmbedDim` ∈ {32, 64, 128, 256, 512} (config `WidthSweep`). One fresh model per width, trained to
  convergence, its D measured.
- **Pinned (identical across every width):** 2 layers, 4 heads, block 256, NoPE, 4000 epochs, lr 3e-4,
  train move counts [4..16], eval move counts [4..48], D threshold 90%. *Do not change these mid-sweep* —
  D-vs-width is only interpretable if everything else is fixed.
- **D definition:** the largest eval move-count whose **per-step** trace accuracy ≥ 0.90. Per-step (not
  whole-sequence) so it locates where the head drifts.

## How to run it

```bash
dotnet run -c Release --project examples/CoreTrainer.Examples.HeadTrack
```

- It runs all five widths in sequence, unattended. Expect several hours (each width is 4000 epochs; larger
  widths are slower). This is the intended overnight run.
- **Restartable.** Each width checkpoints to `bin/Release/net10.0/checkpoints/head-track-w<width>.*` every
  500 epochs. If the machine or process dies, just re-run the same command: finished widths' checkpoints
  are at epoch 4000 (skipped instantly), and the interrupted width resumes where it left off.
- **Log:** `bin/Release/net10.0/logs/head-track-<timestamp>.log` (per-run file).
- **To start completely fresh** (e.g. after changing a pinned axis): delete
  `bin/Release/net10.0/checkpoints/head-track-w*.*` first, or the run will resume stale checkpoints.

## Reading the results in the morning

The log emits three machine-readable line types (all prefixed for easy grep):

- `DATA,<params>,<moves>,<accuracy>,<epoch>` — the full accuracy-vs-move-count curve at each eval. The
  final-epoch rows per width are the converged D-curve.
- `DPOINT,<params>,<D>,<epoch>` — the single D value at each eval (watch it grow then plateau with epochs).
- `SWEEP,<width>,<params>,<D>` — **the payoff.** One row per width in the final summary block
  (`=== Width sweep complete: D vs params ===`). This is the D-vs-params table to plot.

Extract the summary table:

```bash
grep "SWEEP," bin/Release/net10.0/logs/head-track-*.log
```

For the D-curve of a specific width (to see the accuracy fall-off shape), grep the final-epoch DATA rows
for that width's param count.

### The graph to make

- **Primary:** D (y) vs params (x, log scale). Shape is the finding:
  - linear / power-law, keeps climbing → **capacity-bound**: scaling buys tape, the machine is practical.
  - flattening / logarithmic → **mechanism-bound**: scaling won't save it, the head needs an explicit anchor.
- **Secondary:** the per-width accuracy-vs-move-count curves overlaid — shows the cliff sharpening or moving
  out with width.

(Plot with the `dataviz` skill or any tool; the `SWEEP,`/`DATA,` rows are CSV-ready.)

## After this sweep — the remaining axes (one campaign each, pinned otherwise)

D is a surface, not a number. Sweep one axis at a time so each graph is attributable:

1. **Width** ← this run.
2. **Depth** (Layers 1/2/4/6/8) — is head-tracking depth-bound rather than width-bound?
3. **Positional scheme** (NoPE vs RoPE) — the axis tied to the original question; expected to move D most.
4. (later) heads, train-move-range.

Each is a config change + fresh checkpoints. Visualize after each, then set up the next.

## Run log

- **2026-07-13 night → 2026-07-14 am:** first overnight attempt. Width 32 trained fine (loss → 0.0004)
  but the process **crashed at epoch 4000's final eval** with `IndexOutOfRangeException` in `Decode` —
  the trained model argmaxed the reserved **pad id** (index = VocabSize) during free generation, and
  `Decode` indexed past `_idToChar`. Latent bug, triggered only once the model was confident enough to
  emit a pad token. **Fixed:** `CharTokenizer.Decode` now skips ids outside the real vocab; the sweep
  loop also catches per-width exceptions (logs to file + continues) so the crash goes to the log and one
  bad width can't kill the others. The crash went to *stderr* originally, which is why the log looked
  like a clean stop — now handled.
- **Width 32 result (26K params), converged:** D = **8 moves** (threshold 90%). Curve:
  `4:99% 8:94% 12:87% 16:80% 20:72% 24:66% 28:61% 32:58% … 48:53%` — a smooth decay toward the 50%
  random floor (consistent with per-step reliability compounding, ~0.99^N), NOT a hard cliff. The
  primitive works; D is well-defined and measurable. Its checkpoint is at epoch 3500, so a restart
  resumes 3500→4000 (~5 min) and completes its now-fixed final eval before moving to width 64.

- **2026-07-15/16: width campaign COMPLETE.** Widths 32/64/128/256 measured. Result: **width saturates**
  — mean per-step accuracy 66 → 73 → 73 → 70 across a 60× param range; rises once (32→64) then flat. See
  **[FINDINGS.md](FINDINGS.md)** for the full table, the metric change (mean-acc replaced brittle
  D@threshold), and caveats (256 trained only 2000 epochs; 512 skipped). Config now trimmed to 2000 epochs
  and `WidthSweep` reduced to [32,64,128,256]. **Next campaign: depth.**

## Gotchas / notes

- `EmbedDim` must stay divisible by `Heads` (4) — the run throws if a swept width violates it.
- Windows locks the build output while the process runs; to rebuild you must stop the run first
  (`taskkill //F //IM CoreTrainer.Examples.HeadTrack.exe`).
- Sanity-checked 2026-07-13: self-check passes, curves + DATA lines render, untrained model correctly
  reads D=0 (~50% per-step, random baseline). The machinery is verified; only the training-time result
  is pending.
- Memory: [[turing-completeness-ladder]] has the full arc and the capacity-vs-mechanism framing.
