# TODO

Tracked follow-ups from the project code review. Items #1–#4 (configuration wiring:
color scheme loading, GUI honoring `CoreWarsOptions`, grid sizing, shared warrior
placement) and #6 (N-warrior support) have been addressed.

## #5 — Remove dead public API surface

These members have no callers and no tests:

- `GridConfig.GridToAddress` — [src/CoreTrainer.Graphics/GridConfig.cs](src/CoreTrainer.Graphics/GridConfig.cs)
- `GridConfig.TotalCells` — only used by a test now; confirm before removing

**Resolved:** `SimulationController.Warrior0StartAddress` / `Warrior1StartAddress`
were removed during the N-warrior work (#6); `Initialize` now takes warrior/position
lists, so the per-warrior start-address properties no longer exist.

**Keep:** `GridConfig.ScreenToGrid` — intended for the upcoming cell-hover tooltip
feature (see visualizer-design.md Phase 6). Leaving it in anticipation of that work.

**Action:** delete the orphaned members, or add the consuming feature/test that
justifies each. Speculative API with no caller tends to rot silently.

## #7 — `CoreRenderer` redraws everything every frame

[src/CoreTrainer.Graphics/Rendering/CoreRenderer.cs](src/CoreTrainer.Graphics/Rendering/CoreRenderer.cs)

- `Render` draws all cells (8000 at default size) plus ~181 grid lines every frame,
  regardless of what changed.

**Deferred by decision** — render-texture / dirty-cell optimization was explicitly
postponed. Revisit only if profiling shows the draw loop missing the frame budget,
or when larger cores / higher step rates make it visible. Likely approaches:
cache grid lines to a render texture, and/or track dirty cells and repaint only
those.

## NAND warrior — richer codec inputs (future)

The trainable NAND warrior (`CoreTrainer.Networking.Nand`) currently plans to feed
the network only the warrior's own state (PC, process count) and the ±N cell window
contents. Once the basic setup works, consider adding per-cell context inputs:

- **Cell owner** — which warrior last wrote each cell in the window (lets the network
  distinguish its own code from an enemy's).
- **Warrior presence** — whether an active process of another warrior sits on a cell.

Both are already tracked by `Mars` (cell ownership) or derivable from the process
queues, so this is a codec-input expansion, not an engine change. Deferred until the
basic state + window codec is proven.

## NAND warrior — autonomous (program-driven) sub-processes (future)

Today a brain is keyed per-warrior, not per-process: `Mars.ExecuteInstruction`
consults `_brains[warriorIndex]` for every dequeued process and unconditionally
overwrites the PC cell with the brain's decision before executing. So a brain warrior
pilots *every* one of its processes — an SPL'd child is also brain-driven, and the
brain cannot "compile" a tight loop into memory and let it run free.

Letting a brain spawn autonomous, program-driven sub-processes would be a real
feature, motivated by a genuine **cost asymmetry**: a brain step is expensive (encode →
evaluate the network → decode) while a program-driven process is nearly free. A warrior
that could delegate a cheap bomber/imp to memory and keep its brain free to scout would
be strictly more capable.

The blocker is signaling: the emitted `Instruction` has no spare "autonomous" bit, and
adding one shouldn't break the 16-opcode (4-bit) bijection. Likely substrate is a
per-process flag — change the process queue from `Queue<int>` to carry `(pc, autonomous)`
and have `ExecuteInstruction` check the *process* flag rather than only the warrior's
brain. The open question is the trigger (new opcode vs. convention vs. an explicit
flag). Deferred: the all-brain model is sufficient to get a training loop running, and
this is far easier to design once we've seen what evolved brains actually struggle with.
