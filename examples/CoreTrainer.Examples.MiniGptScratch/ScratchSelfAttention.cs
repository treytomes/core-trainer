namespace CoreTrainer.Examples.MiniGptScratch;

/// <summary>
/// Multi-head causal self-attention, hand-rolled with an explicit backward pass. Mirrors the
/// TorchSharp <c>CausalSelfAttention</c> but with every matmul, mask, softmax, and their gradients
/// written out.
/// </summary>
/// <remarks>
/// Forward, per head <c>h</c> over a length-<c>T</c> sequence: split the fused Q/K/V projections into
/// heads, form scores <c>Q·Kᵀ/√d</c>, mask out future positions (upper triangle) to −∞, softmax each
/// row, and blend the values <c>weights·V</c>. Backward reverses each step; the interesting piece is
/// that the same score matrix feeds both the softmax and (through it) Q and K, so their gradients are
/// reconstructed from the per-head attention weights saved during the forward pass.
/// </remarks>
public sealed class ScratchSelfAttention
{
	#region Fields

	private readonly int _heads;
	private readonly int _headDim;
	private readonly double _scale;
	private readonly ILinearLayer _qkv;   // fused projection to [Q | K | V], width 3·embedDim
	private readonly ILinearLayer _proj;  // output projection

	// Saved forward activations for the backward pass.
	private Matrix? _q, _k, _v;                 // [time, embedDim]
	private Matrix[]? _attnWeights;             // per head: [time, time]
	private Matrix? _attnConcat;                // [time, embedDim] input to _proj
	private int _time;

	#endregion

	#region Constructors

	public ScratchSelfAttention(int embedDim, int heads, Random init, LinearFactory linear)
	{
		if (embedDim % heads != 0)
			throw new ArgumentException($"embedDim ({embedDim}) must be divisible by heads ({heads}).");

		_heads = heads;
		_headDim = embedDim / heads;
		_scale = 1.0 / Math.Sqrt(_headDim);
		_qkv = linear(embedDim, 3 * embedDim, init, true);
		_proj = linear(embedDim, embedDim, init, true);
	}

	#endregion

	#region Methods

	/// <summary>Collects the learnable parameters of both projections.</summary>
	public IEnumerable<Parameter> Parameters() => _qkv.Parameters().Concat(_proj.Parameters());

	/// <summary>The linear projections in this block, for inspection (e.g. the ternary compile step).</summary>
	public IEnumerable<ILinearLayer> LinearLayers()
	{
		yield return _qkv;
		yield return _proj;
	}

	/// <summary>
	/// Forward pass over one sequence <c>x</c> (shape <c>[time, embedDim]</c>), returning the mixed
	/// representation of the same shape. Saves the per-head attention weights and Q/K/V for backward.
	/// </summary>
	public Matrix Forward(Matrix x)
	{
		_time = x.Rows;
		var embedDim = x.Cols;

		var qkv = _qkv.Forward(x);                       // [time, 3·embedDim]
		(_q, _k, _v) = SplitQkv(qkv, embedDim);

		_attnWeights = new Matrix[_heads];
		_attnConcat = new Matrix(_time, embedDim);

		for (var h = 0; h < _heads; h++)
		{
			var offset = h * _headDim;

			// Scores[i,j] = (Q_i · K_j) · scale, then mask future (j > i) to -inf and softmax rows.
			var scores = new Matrix(_time, _time);
			for (var i = 0; i < _time; i++)
			{
				for (var j = 0; j < _time; j++)
				{
					if (j > i)
					{
						scores[i, j] = double.NegativeInfinity;
						continue;
					}
					var dot = 0.0;
					for (var d = 0; d < _headDim; d++)
						dot += _q[i, offset + d] * _k[j, offset + d];
					scores[i, j] = dot * _scale;
				}
			}

			var weights = Ops.SoftmaxRows(scores);       // [time, time], rows sum to 1
			_attnWeights[h] = weights;

			// Blend values: out_i = Σ_j weights[i,j] · V_j, written into this head's slice.
			for (var i = 0; i < _time; i++)
				for (var d = 0; d < _headDim; d++)
				{
					var acc = 0.0;
					for (var j = 0; j <= i; j++)
						acc += weights[i, j] * _v[j, offset + d];
					_attnConcat[i, offset + d] = acc;
				}
		}

		return _proj.Forward(_attnConcat);
	}

	/// <summary>
	/// Backward pass. Given the gradient w.r.t. the block output, accumulates parameter gradients and
	/// returns the gradient w.r.t. the input <c>x</c>.
	/// </summary>
	public Matrix Backward(Matrix dOut)
	{
		if (_q is null || _k is null || _v is null || _attnWeights is null || _attnConcat is null)
			throw new InvalidOperationException("Backward called before Forward.");

		var embedDim = _q.Cols;

		// Back through the output projection to the concatenated attention output.
		var dConcat = _proj.Backward(dOut);              // [time, embedDim]

		// Gradients accumulate into the split Q/K/V, then get recombined for the fused qkv backward.
		var dQ = new Matrix(_time, embedDim);
		var dK = new Matrix(_time, embedDim);
		var dV = new Matrix(_time, embedDim);

		for (var h = 0; h < _heads; h++)
		{
			var offset = h * _headDim;
			var weights = _attnWeights[h];

			// dWeights[i,j] = Σ_d dConcat[i,d] · V[j,d]; and dV[j,d] += Σ_i weights[i,j]·dConcat[i,d].
			var dWeights = new Matrix(_time, _time);
			for (var i = 0; i < _time; i++)
			{
				for (var j = 0; j <= i; j++)
				{
					var dw = 0.0;
					for (var d = 0; d < _headDim; d++)
					{
						var g = dConcat[i, offset + d];
						dw += g * _v[j, offset + d];
						dV[j, offset + d] += weights[i, j] * g;
					}
					dWeights[i, j] = dw;
				}
			}

			// Back through the row-wise softmax to the (masked) scores.
			var dScores = Ops.SoftmaxRowsBackward(dWeights, weights);

			// Scores[i,j] = scale · Σ_d Q[i,d]·K[j,d]  ->  dQ, dK (only over unmasked j ≤ i).
			for (var i = 0; i < _time; i++)
				for (var j = 0; j <= i; j++)
				{
					var ds = dScores[i, j] * _scale;
					for (var d = 0; d < _headDim; d++)
					{
						dQ[i, offset + d] += ds * _k[j, offset + d];
						dK[j, offset + d] += ds * _q[i, offset + d];
					}
				}
		}

		// Recombine dQ|dK|dV into the fused [time, 3·embedDim] gradient and back through the qkv linear.
		var dQkv = new Matrix(_time, 3 * embedDim);
		for (var r = 0; r < _time; r++)
			for (var c = 0; c < embedDim; c++)
			{
				dQkv[r, c] = dQ[r, c];
				dQkv[r, embedDim + c] = dK[r, c];
				dQkv[r, 2 * embedDim + c] = dV[r, c];
			}

		return _qkv.Backward(dQkv);
	}

	/// <summary>Splits a fused [time, 3·embedDim] projection into its Q, K, V slices.</summary>
	private (Matrix Q, Matrix K, Matrix V) SplitQkv(Matrix qkv, int embedDim)
	{
		var q = new Matrix(_time, embedDim);
		var k = new Matrix(_time, embedDim);
		var v = new Matrix(_time, embedDim);
		for (var r = 0; r < _time; r++)
			for (var c = 0; c < embedDim; c++)
			{
				q[r, c] = qkv[r, c];
				k[r, c] = qkv[r, embedDim + c];
				v[r, c] = qkv[r, 2 * embedDim + c];
			}
		return (q, k, v);
	}

	#endregion
}
