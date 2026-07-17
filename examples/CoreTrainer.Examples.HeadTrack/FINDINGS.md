# HeadTrack findings

Measuring **D**, the reliable head-tracking distance — how many one-cell L/R head moves a NoPE
transformer can follow over a windowed tape before it loses track — and how D scales with model size.
This is the primitive the scrollable-tape Turing-machine design rests on (transformer = control, external
buffer = tape, window ≤ D). See [RUNBOOK.md](RUNBOOK.md) for the experimental setup and how to run it.

## Metric note: mean per-step accuracy, not D@threshold

We initially reported **D = the largest move-count whose per-step accuracy ≥ 90%**. That number turned
out to be **too brittle to compare models**: it is a single threshold-crossing point, so two models with
near-identical accuracy curves can report very different D just because one grazes 90% at move-12 and the
other misses by a fraction. The width sweep showed exactly this — D flipped 12 → 8 → 4 between widths
whose actual curves were nearly on top of each other.

The robust metric is **mean per-step accuracy across all eval move-counts (4…48)** — an area-under-the-curve
summary that uses the whole curve. Both are logged (`meanAcc` and `D`); **meanAcc is the primary metric.**

## Campaign 1: width (EmbedDim), all else pinned

Fixed: 2 layers, 4 heads, block 256, NoPE, train move-counts 4–16, eval 4–48. (Widths 32/64/128 trained
4000 epochs; 256 trained 2000 after we trimmed the epoch budget — see caveat.)

| EmbedDim | params | mean per-step acc | D@90% |
|----------|--------|-------------------|-------|
| 32  | 26 K  | 66.0% | 8 |
| 64  | 101 K | 73.2% | 12 |
| 128 | 399 K | 73.1% | 8 |
| 256 | 1.6 M | 70.1% | 4 |

### Finding: **width saturates — it does not buy head-tracking distance**

Across a **60× parameter range**, mean accuracy goes **66 → 73 → 73 → 70**. It rises once (32→64, ~+7
points) and is then flat-to-slightly-down. Every 4× width increase after the first bought essentially
nothing. This is the strong form of **mechanism-bound, not capacity-bound**: beyond ~100 K parameters the
head-tracking limit is set by the *architecture* (a NoPE model reconstructing position through attention),
and adding width cannot move it.

Every width produces the same shape of curve — near-perfect at 4 moves, smooth decay toward the 50% random
floor by ~48 moves (consistent with per-step reliability compounding, ~0.99ᴺ) — the curves just don't
separate much past width 64.

### Honest caveats

- **Width 256 (70.1%) is below 64/128, but it trained for only 2000 epochs** where 64/128 ran 4000 (we
  trimmed the budget mid-campaign after seeing meanAcc plateau by ~epoch 500–1000). Its 70.1% may be mild
  undertraining rather than genuine degradation — its curve is healthy (98% @ 4 moves, smooth decay), so
  "converged on the plateau, a hair low" is at least as likely as "large width is worse." We did **not**
  re-run 256 to 4000 to disambiguate; the headline (no useful growth) holds either way, but do not claim
  "width hurts" from this alone.
- Width 512 was **not run** — the plateau was established by 64/128 and confirmed at 16× params by 256, so
  32× was not worth the ~day of wall-clock.
- Single seed per width. The mean-accuracy differences among 64/128/256 (73.2/73.1/70.1) are within what a
  seed change could plausibly move; treat 64–256 as "flat," not as a rank order.
- **Epoch-2000 == epoch-4000 for every width** (recovered from the per-500-epoch DATA logs: 32→66.0,
  64→73.2, 128→73.1 identical at both). So the metric was flat from ~epoch 1000, the epoch trim to 2000
  loses nothing, and the 256@2000-vs-others@4000 comparison is fair — no width run gained anything from the
  extra epochs. This also makes the depth campaign (2000 epochs) directly comparable to width.

## What this implies for the next campaigns

Width saturating does **not** mean the architecture is capped — it means width was the wrong dial. The
open question is whether another axis behaves differently:

- **Depth (next):** head-tracking is a *sequential accumulation* ("where am I after N moves"). Depth adds
  compositional steps per forward pass; width only adds representational precision, which this task
  apparently stops needing past ~100 K params. If depth keeps climbing where width flatlined, the headline
  becomes **"head-tracking scales with depth, not width"** — a sharper and more useful result.
- **Positional scheme (later):** NoPE vs RoPE — the axis tied to the original question. Expected to shift
  the whole curve (explicit relative position vs reconstructed).

Discipline: one axis per campaign, everything else pinned; report **meanAcc**; watch that the metric is a
whole-curve summary, not a threshold crossing.
