using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace CoreTrainer.Examples.MiniGpt;

/// <summary>
/// Multi-head causal (masked) self-attention, the core mixing operation of a GPT-style block.
/// </summary>
/// <remarks>
/// A single fused linear projects each position into concatenated query, key, and value vectors,
/// which are split into <c>heads</c> independent attention heads. Each position attends only to
/// itself and earlier positions — enforced by a lower-triangular mask that sets future scores to
/// negative infinity before the softmax — so the model stays autoregressive: the prediction at
/// position <c>t</c> can never peek at tokens after <c>t</c>. This is what lets the same forward
/// pass both train on a whole sequence at once and generate left to right at inference.
/// </remarks>
public sealed class CausalSelfAttention : nn.Module<Tensor, Tensor>
{
	#region Fields

	private readonly int _heads;
	private readonly int _headDim;

	/// <summary>
	/// Fused projection to [Q | K | V].
	/// </summary>
	private readonly nn.Module<Tensor, Tensor> _qkv;

	/// <summary>
	/// Output projection back to the embedding dimension.
	/// </summary>
	private readonly nn.Module<Tensor, Tensor> _proj;

	#endregion

	#region Constructors

	/// <param name="embedDim">Model/embedding dimension.</param>
	/// <param name="heads">Number of attention heads; must divide <paramref name="embedDim"/>.</param>
	/// <param name="useBitLinear">Use BitNet ternary-weight linears in place of full-precision ones.</param>
	public CausalSelfAttention(int embedDim, int heads, bool useBitLinear = false) : base(nameof(CausalSelfAttention))
	{
		if (embedDim % heads != 0)
			throw new ArgumentException($"embedDim ({embedDim}) must be divisible by heads ({heads}).");

		_heads = heads;
		_headDim = embedDim / heads;
		_qkv = Linears.Make(embedDim, 3 * embedDim, useBitLinear);
		_proj = Linears.Make(embedDim, embedDim, useBitLinear);

		RegisterComponents();
	}

	#endregion

	#region Methods

	/// <summary>
	/// Applies masked self-attention to a batch of sequences (shape <c>[batch, time, embedDim]</c>)
	/// and returns the mixed representation of the same shape.
	/// </summary>
	public override Tensor forward(Tensor x)
	{
		using var scope = NewDisposeScope();

		var batch = x.shape[0];
		var time = x.shape[1];
		var embedDim = x.shape[2];

		// Project to Q, K, V and reshape each into [batch, heads, time, headDim].
		var qkv = _qkv.forward(x);
		var parts = qkv.chunk(3, dim: -1);
		var q = SplitHeads(parts[0], batch, time);
		var k = SplitHeads(parts[1], batch, time);
		var v = SplitHeads(parts[2], batch, time);

		// Scaled dot-product attention scores: [batch, heads, time, time].
		var scores = q.matmul(k.transpose(-2, -1)) / Math.Sqrt(_headDim);

		// Causal mask: forbid attending to future positions (upper triangle above the diagonal).
		var mask = ones([time, time], device: x.device).triu(diagonal: 1).to_type(ScalarType.Bool);
		scores = scores.masked_fill(mask, double.NegativeInfinity);

		var weights = scores.softmax(dim: -1);
		var attended = weights.matmul(v); // [batch, heads, time, headDim]

		// Recombine heads back to [batch, time, embedDim] and project.
		var merged = attended.transpose(1, 2).contiguous().reshape(batch, time, embedDim);
		return _proj.forward(merged).MoveToOuterDisposeScope();
	}

	/// <summary>Reshapes [batch, time, embedDim] into per-head [batch, heads, time, headDim].</summary>
	private Tensor SplitHeads(Tensor t, long batch, long time) =>
		t.reshape(batch, time, _heads, _headDim).transpose(1, 2);

	#endregion
}
