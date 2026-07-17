using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace CoreTrainer.Examples.HeadTrack;

/// <summary>
/// Multi-head causal (masked) self-attention — the cross-position mixing operation. This is the path by
/// which a step on the scratchpad can read earlier steps (e.g. copy the previous line's carry forward),
/// so it is the mechanism that makes the "context as tape" execution possible.
/// </summary>
public sealed class CausalSelfAttention : nn.Module<Tensor, Tensor>
{
	private readonly int _heads;
	private readonly int _headDim;
	private readonly Linear _qkv;
	private readonly Linear _proj;

	public CausalSelfAttention(int embedDim, int heads) : base(nameof(CausalSelfAttention))
	{
		if (embedDim % heads != 0)
			throw new ArgumentException($"embedDim ({embedDim}) must be divisible by heads ({heads}).");

		_heads = heads;
		_headDim = embedDim / heads;
		_qkv = nn.Linear(embedDim, 3 * embedDim);
		_proj = nn.Linear(embedDim, embedDim);
		RegisterComponents();
	}

	public override Tensor forward(Tensor x)
	{
		using var scope = NewDisposeScope();

		var batch = x.shape[0];
		var time = x.shape[1];
		var embedDim = x.shape[2];

		var qkv = _qkv.forward(x);
		var parts = qkv.chunk(3, dim: -1);
		var q = SplitHeads(parts[0], batch, time);
		var k = SplitHeads(parts[1], batch, time);
		var v = SplitHeads(parts[2], batch, time);

		var scores = q.matmul(k.transpose(-2, -1)) / Math.Sqrt(_headDim);

		// Causal mask: forbid attending to future positions (upper triangle above the diagonal).
		var mask = ones([time, time], device: x.device).triu(diagonal: 1).to_type(ScalarType.Bool);
		scores = scores.masked_fill(mask, double.NegativeInfinity);

		var weights = scores.softmax(dim: -1);
		var attended = weights.matmul(v);

		var merged = attended.transpose(1, 2).contiguous().reshape(batch, time, embedDim);
		return _proj.forward(merged).MoveToOuterDisposeScope();
	}

	private Tensor SplitHeads(Tensor t, long batch, long time) =>
		t.reshape(batch, time, _heads, _headDim).transpose(1, 2);
}
