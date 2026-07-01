# Mini-GPT — a small transformer, two ways

Two standalone examples that build the **same** GPT-2-style transformer and train it to
memorize a handful of sentences — one using the TorchSharp tensor library, one hand-rolled
from scratch with no library at all. They are meant to be read as a pair:

| Project | What it is | What it shows |
|---------|-----------|---------------|
| **CoreTrainer.Examples.MiniGpt** | The transformer in ~5 files using **TorchSharp** | How you'd actually build one: concise, autograd + Adam handled for you |
| **CoreTrainer.Examples.MiniGptScratch** | The identical model with **no dependencies** | What TorchSharp hides — every matmul, softmax, **gradient**, and the Adam update, all explicit |

Neither example has anything to do with Core Wars. They exist to make the transformer
architecture — and the training math underneath it — concrete.

## Run them

```bash
# The library version
dotnet run -c Release --project examples/CoreTrainer.Examples.MiniGpt

# The from-scratch version (also runs a gradient check first)
dotnet run -c Release --project examples/CoreTrainer.Examples.MiniGptScratch
```

Both train for a couple of minutes on CPU, then print a memorization test: each sentence is
prompted with its first 10 characters and the model must reproduce the rest verbatim.
Success looks like `Memorization: 5/5 sentences reproduced verbatim.`

## The task: memorize, on purpose

Real language models are trained *not* to memorize — they must generalize. These examples
do the opposite deliberately: with a fixed handful of sentences, **success is exact
memorization.** That makes the "did it learn?" test unambiguous (greedy decoding reproduces
each sentence character-for-character) and lets a tiny model train to near-zero loss on a
laptop. Dropout is off; we train until the loss collapses.

Tokenization is **character-level** — the vocabulary is just the distinct characters in the
corpus, so there is no out-of-vocabulary case and no external tokenizer.

## How the transformer works

Both projects implement the standard decoder-only GPT recipe. Following one forward pass:

1. **Embeddings.** Each character id becomes a learned vector; a learned *position* vector
   is added so the model knows token order (attention alone is order-blind). The sum is the
   **residual stream** — one vector per position that every layer reads from and writes back
   to.
2. **Causal self-attention.** Each position projects its vector into a *query*, *key*, and
   *value*. Query·key dot products score how relevant each earlier position is; a **causal
   mask** forbids attending to future positions (so position *t* depends only on tokens
   0..*t*); softmax turns the scores into weights; the output is the weighted blend of
   values. Done in several **heads** in parallel, each attending for a different reason.
3. **MLP.** A position-wise two-layer network (expand ×4, GELU, contract) processes what
   each position gathered. Attention *moves* information between positions; the MLP
   *transforms* it within a position.
4. **Residuals + LayerNorm.** Each sub-layer *adds* its output to the stream (`x + sub(x)`)
   rather than replacing it, and LayerNorm keeps activations well-scaled. This is what makes
   deep stacks trainable. The block (attention + MLP) is stacked several times.
5. **Head.** A final projection turns each position's vector into one score per vocabulary
   entry — the next-token prediction. Training compares every position's prediction to the
   actual next character (cross-entropy); generation takes the last position's argmax,
   appends it, and repeats.

### File map (both projects mirror this structure)

| Concept | TorchSharp (`MiniGpt`) | From scratch (`MiniGptScratch`) |
|---------|------------------------|----------------------------------|
| Attention | `CausalSelfAttention.cs` | `ScratchSelfAttention.cs` (+ explicit backward) |
| Block | `TransformerBlock.cs` | `ScratchBlock.cs` (+ explicit backward) |
| Full model | `MiniGpt.cs` | `ScratchGpt.cs` (+ embedding backward) |
| Tokenizer | `CharTokenizer.cs` | `CharTokenizer.cs` |
| Ops | *(hidden in TorchSharp)* | `Ops.cs`, `Matrix.cs`, `Linear.cs`, `AdamOptimizer.cs` |
| Driver | `Program.cs` | `Program.cs` (+ gradient check) |

## What the from-scratch version adds

The scratch project is the interesting one, because it makes visible the two things a tensor
library does silently:

- **The backward pass (autograd).** `.backward()` in TorchSharp is replaced by a hand-derived
  gradient for *every* operation: `dX = dY·Wᵀ` for linear layers, the softmax Jacobian, the
  LayerNorm gradient, GELU's derivative, the full dQ/dK/dV through the attention mask and
  softmax, and a scatter-add back into the embedding tables. These are in `Ops.cs`,
  `Linear.cs`, and the `Scratch*` classes.
- **The optimizer.** `optimizer.step()` is replaced by `AdamOptimizer.cs` — the first/second
  moment estimates, bias correction, and per-parameter adaptive update, spelled out.

### The gradient check

Hand-rolled backprop is exactly where silent bugs hide: a wrong sign or a missing term still
*sort of* trains, just worse, so you can't tell from the loss alone. Before training, the
scratch program runs a **gradient check** — it perturbs individual weights and compares the
analytic gradient from the backward pass against a numerical finite-difference estimate,
`(L(w+h) − L(w−h)) / 2h`. Agreement to several significant figures proves the math is right.
A healthy run reports something like:

```
=== Gradient check (analytic vs numerical) ===
  base loss 2.0671, checked 20 weights, max relative error 2.755E-7
  PASS: analytic gradients match numerical (max rel err < 1e-4).
```

This is the standard technique for validating a from-scratch backprop, and it's the reason
you can trust that the two projects are genuinely computing the same thing.

## Tuning

Knobs are `const`s at the top of each `Program.cs`: `EMBED_DIM`, `HEADS`, `LAYERS`,
`BLOCK_SIZE`, `EPOCHS`, `LEARNING_RATE`, `SEED`. Both are reproducible (fixed seed). The
scratch version is intentionally a little smaller (fewer dimensions/layers) since every
operation runs in plain C# loops rather than a vectorized backend, but it reaches the same
memorization result.

## Why two versions

The TorchSharp example is how you would really build a transformer — terse, fast, and with
the hard parts (autograd, optimized kernels) handled for you. The scratch example is how you
*understand* one: nothing is hidden, every number is steppable in a debugger, and the
gradient check proves the whole backward pass is correct. Read `MiniGpt` for the shape; read
`MiniGptScratch` for the substance.
