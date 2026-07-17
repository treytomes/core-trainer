using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace CoreTrainer.Examples.AdderTape;

/// <summary>
/// One GPT-2-style transformer block: pre-LayerNorm masked self-attention and a position-wise GELU MLP,
/// each wrapped in a residual connection. Plain full precision — this experiment tests the training
/// (context-as-tape), not the architecture, so nothing exotic here.
/// </summary>
public sealed class TransformerBlock : nn.Module<Tensor, Tensor>
{
	private readonly LayerNorm _ln1;
	private readonly CausalSelfAttention _attention;
	private readonly LayerNorm _ln2;
	private readonly Sequential _mlp;

	public TransformerBlock(int embedDim, int heads) : base(nameof(TransformerBlock))
	{
		_ln1 = nn.LayerNorm(embedDim);
		_attention = new CausalSelfAttention(embedDim, heads);
		_ln2 = nn.LayerNorm(embedDim);
		_mlp = nn.Sequential(
			nn.Linear(embedDim, 4 * embedDim),
			nn.GELU(),
			nn.Linear(4 * embedDim, embedDim));
		RegisterComponents();
	}

	public override Tensor forward(Tensor x)
	{
		using var scope = NewDisposeScope();
		x = x + _attention.forward(_ln1.forward(x));
		x = x + _mlp.forward(_ln2.forward(x));
		return x.MoveToOuterDisposeScope();
	}
}
