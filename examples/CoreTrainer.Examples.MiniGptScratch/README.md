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
