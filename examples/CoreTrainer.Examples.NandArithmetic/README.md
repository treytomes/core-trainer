# NAND Arithmetic — Training System Demo

A standalone, self-contained example of the Cartesian Genetic Programming (CGP) training
system used by the Core Wars NAND warriors — applied here to a much simpler, easier-to-
follow problem: **evolving a network of NAND gates to add and subtract numbers.**

There is no Core Wars here. The point is to see the evolution machinery work in isolation.

## Run it

```bash
dotnet run -c Release --project examples/CoreTrainer.Examples.NandArithmetic
```

You'll see the starting accuracy (~50%, random), then accuracy climbing generation by
generation, and finally a few concrete sample evaluations.

## What it demonstrates

The same three pieces that power the Core Wars warriors, with a different goal:

| Piece | Role here |
|-------|-----------|
| `NandGenome`  | the evolvable description of a NAND-gate circuit |
| `NandNetwork` | decodes a genome and evaluates it for a set of input bits |
| `CgpEvolution`| the 1+λ strategy that searches for a better genome |

**The only problem-specific part is the fitness function.** Here it scores "how many
output bits match the correct answer." In Core Wars it scores "how well the warrior does
in battle." Everything else — the genome, the network, the search loop — is identical.
That separation is the main lesson of this example.

## How it works

- **Inputs:** the two operands (`BitWidth` bits each) plus one bit selecting add vs.
  subtract.
- **Outputs:** `BitWidth + 1` bits — the result plus a carry/borrow bit. Subtraction uses
  two's complement, so every output pattern is a valid number.
- **Fitness:** the fraction of output bits that are correct, averaged over a fixed set of
  sampled test cases. Scoring per-*bit* (not per-*answer*) gives the search a smooth
  gradient — a circuit that gets most bits right scores higher than one that doesn't, even
  before any full answer is correct.

## Tuning

All knobs are `const`s at the top of [Program.cs](Program.cs):

- `BitWidth` — **drop this to 4** to watch a smaller circuit train faster. Full 8-bit
  add/sub is a genuinely hard structure to discover from random wiring with a plain 1+λ
  search, so expect steady improvement rather than perfection in a short run; the
  low-order bits are learned readily, the high-order carry chain much more slowly.
- `NodeCount` — how many NAND gates the network may wire together.
- `Generations`, `Lambda`, `MutationRate` — the search budget and behavior.
- `TestCases` — how many sampled operations are used to score each genome.

The run is reproducible: the test set and evolution both use fixed seeds.

## A note on results

This example is honest about the difficulty: it reliably *learns* (accuracy rises well
above the random 50% baseline) but does not trivially reach 100% on 8-bit arithmetic in a
short run. That is the real character of evolutionary search on a hard combinational
target — and exactly why the Core Wars warriors, facing an even harder objective, are a
research-grade experiment rather than a solved problem.
