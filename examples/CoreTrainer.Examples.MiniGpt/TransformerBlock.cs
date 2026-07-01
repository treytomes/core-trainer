using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace CoreTrainer.Examples.MiniGpt;

/// <summary>
/// One GPT-2-style transformer block: pre-layer-norm masked self-attention and a position-wise
/// MLP, each wrapped in a residual connection.
/// </summary>
/// <remarks>
/// Pre-LN (normalize before each sub-layer, add the residual after) is the arrangement GPT-2 uses;
/// it keeps gradients well-scaled through deep stacks and trains more stably than the original
/// post-LN transformer. The MLP expands to 4x the embedding dimension and back — the conventional
/// ratio — with a GELU nonlinearity in between.
/// </remarks>
public sealed class TransformerBlock : nn.Module<Tensor, Tensor>
{
	#region Fields

	private readonly LayerNorm _ln1;
	private readonly CausalSelfAttention _attention;
	private readonly LayerNorm _ln2;
	private readonly Sequential _mlp;

	#endregion

	#region Constructors

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

	#endregion

	#region Methods

	/// <summary>
	/// Runs the block over a batch of sequences (shape <c>[batch, time, embedDim]</c>), returning the
	/// same shape.
	/// </summary>
	public override Tensor forward(Tensor x)
	{
		using var scope = NewDisposeScope();

		// Residual around pre-LN attention, then residual around pre-LN MLP.
		x = x + _attention.forward(_ln1.forward(x));
		x = x + _mlp.forward(_ln2.forward(x));
		return x.MoveToOuterDisposeScope();
	}

	#endregion
}
