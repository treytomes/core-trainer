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

	/// <summary>
	/// Estimated per-head work (score + blend ≈ 2·time²·headDim scalar ops) at or above which the head
	/// loop is run in parallel; below it the threads cost more than they save, so it stays serial. Set
	/// once at startup from configuration via <see cref="ConfigureParallelism"/>. Default int.MaxValue =
	/// always serial, so behavior is unchanged until explicitly configured.
	/// </summary>
	private static long _parallelHeadThreshold = long.MaxValue;

	private readonly int _heads;
	private readonly int _headDim;
	private readonly Real _scale;
	private readonly ILinearLayer _qkv;   // fused projection to [Q | K | V], width 3·embedDim
	private readonly ILinearLayer _proj;  // output projection

	// QK-normalization keeps attention logits O(1) so they cannot grow unbounded and overflow low-range
	// fixed-point. Each head's Q and K row (length headDim) is LayerNorm-normalized before the dot
	// product, parameter-free (unit scale, zero shift), reusing the gradient-checked LayerNorm ops.
	// Always on: benign in double/float, essential in Fixed32; being unconditional lets the double-mode
	// gradient check validate the added backward.
	private readonly Matrix _qkGamma; // [1, headDim] ones
	private readonly Matrix _qkBeta;  // [1, headDim] zeros

	// Saved forward activations for the backward pass.
	private Matrix? _qn, _kn, _v;               // POST-norm Q, K and (un-normalized) V, [time, embedDim]
	private Matrix? _qRaw, _kRaw;               // PRE-norm Q, K, for the QK-norm backward
	private double[][]? _qMean, _qInvStd;       // per-head LayerNorm stats for Q: [head][time]
	private double[][]? _kMean, _kInvStd;       // per-head LayerNorm stats for K
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
		_scale = (Real)(1.0 / Math.Sqrt(_headDim));
		_qkv = linear(embedDim, 3 * embedDim, init, true);
		_proj = linear(embedDim, embedDim, init, true);

		// Parameter-free QK-norm: unit scale, zero shift.
		_qkGamma = new Matrix(1, _headDim);
		_qkGamma.Fill((Real)1);
		_qkBeta = new Matrix(1, _headDim);
	}

	#endregion

	#region Methods

	/// <summary>
	/// Sets the per-head work threshold at or above which the head loops run in parallel (see
	/// <see cref="_parallelHeadThreshold"/>). A threshold of 0 always parallelizes; a very large value
	/// never does. Call once at startup from configuration.
	/// </summary>
	public static void ConfigureParallelism(long headWorkThreshold) => _parallelHeadThreshold = headWorkThreshold;

	/// <summary>
	/// True when this layer's per-head work is large enough to justify parallelizing across heads. Work
	/// is estimated as the score+blend cost ≈ 2·time²·headDim; below the configured threshold the loop
	/// stays serial (thread overhead would dominate at small sizes / few heads).
	/// </summary>
	private bool ShouldParallelizeHeads()
	{
		if (_heads <= 1)
			return false;
		var perHeadWork = 2L * _time * _time * _headDim;
		return perHeadWork >= _parallelHeadThreshold;
	}

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
		(_qRaw, _kRaw, _v) = SplitQkv(qkv, embedDim);

		// Normalized Q/K (filled per head in ForwardHead) and per-head LayerNorm stats for backward.
		_qn = new Matrix(_time, embedDim);
		_kn = new Matrix(_time, embedDim);
		_qMean = new double[_heads][];
		_qInvStd = new double[_heads][];
		_kMean = new double[_heads][];
		_kInvStd = new double[_heads][];

		_attnWeights = new Matrix[_heads];
		_attnConcat = new Matrix(_time, embedDim);

		// Heads are independent: each reads Q/K/V and writes only its own column slice of _attnConcat
		// and its own _attnWeights[h], so the loop parallelizes with no shared writes. Gated on a work
		// threshold because for few/small heads the thread overhead outweighs the work.
		if (ShouldParallelizeHeads())
			Parallel.For(0, _heads, ForwardHead);
		else
			for (var h = 0; h < _heads; h++)
				ForwardHead(h);

		return _proj.Forward(_attnConcat);
	}

	/// <summary>Computes one head's masked attention and writes its slice of <c>_attnConcat</c>.</summary>
	private void ForwardHead(int h)
	{
		var offset = h * _headDim;

		// QK-norm: LayerNorm this head's Q and K rows (over headDim) into the normalized buffers, saving
		// stats for backward. This bounds the logits so they cannot overflow low-range fixed-point.
		(_qMean![h], _qInvStd![h]) = NormalizeHeadSlice(_qRaw!, _qn!, offset);
		(_kMean![h], _kInvStd![h]) = NormalizeHeadSlice(_kRaw!, _kn!, offset);

		// Scores[i,j] = (Qn_i · Kn_j) · scale for j ≤ i (causal). Future cells (j > i) are left at 0 and
		// excluded from the softmax via the per-row valid count — no -inf sentinel is written, which
		// removes a whole overflow class in low-range fixed-point (the sentinel-minus-rowmax subtraction).
		var scores = new Matrix(_time, _time);
		var validCounts = new int[_time];
		for (var i = 0; i < _time; i++)
		{
			validCounts[i] = i + 1; // row i attends to columns 0..i
			for (var j = 0; j <= i; j++)
			{
				// Accumulate the raw Qn·Kn dot at doubled fractional width (Mac), then apply the 1/√d
				// scale once. The wide accumulator both removes the per-term rounding that corrupts the
				// gradient direction at low precision AND holds the unscaled dot without overflowing the
				// fixed-point integer part — so the old per-term scale fold (needed only to keep the
				// accumulator small) is no longer required, and scaling once is strictly more accurate.
				var mac = new Mac();
				for (var d = 0; d < _headDim; d++)
					mac.MultiplyAdd(_qn![i, offset + d], _kn![j, offset + d]);
				scores[i, j] = mac.Result * _scale;
			}
		}

		var weights = Ops.SoftmaxRows(scores, validCounts); // [time, time], each row sums to 1 over 0..i
		_attnWeights![h] = weights;

		// Blend values: out_i = Σ_j weights[i,j] · V_j, written into this head's slice.
		for (var i = 0; i < _time; i++)
			for (var d = 0; d < _headDim; d++)
			{
				var mac = new Mac();
				for (var j = 0; j <= i; j++)
					mac.MultiplyAdd(weights[i, j], _v![j, offset + d]);
				_attnConcat![i, offset + d] = mac.Result;
			}
	}

	/// <summary>
	/// Backward pass. Given the gradient w.r.t. the block output, accumulates parameter gradients and
	/// returns the gradient w.r.t. the input <c>x</c>.
	/// </summary>
	public Matrix Backward(Matrix dOut)
	{
		if (_qn is null || _kn is null || _v is null || _attnWeights is null || _attnConcat is null)
			throw new InvalidOperationException("Backward called before Forward.");

		var embedDim = _qn.Cols;

		// Back through the output projection to the concatenated attention output.
		var dConcat = _proj.Backward(dOut);              // [time, embedDim]

		// Gradients accumulate into the split Q/K/V, then get recombined for the fused qkv backward.
		var dQ = new Matrix(_time, embedDim);
		var dK = new Matrix(_time, embedDim);
		var dV = new Matrix(_time, embedDim);

		// Each head accumulates only into its own column slice [offset, offset+headDim) of dQ/dK/dV, so
		// distinct heads never touch the same cell — the += accumulations are race-free by construction,
		// and the loop parallelizes like the forward. Same work-threshold gate.
		if (ShouldParallelizeHeads())
			Parallel.For(0, _heads, h => BackwardHead(h, dConcat, dQ, dK, dV));
		else
			for (var h = 0; h < _heads; h++)
				BackwardHead(h, dConcat, dQ, dK, dV);

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

	/// <summary>
	/// Backprops one head: from <paramref name="dConcat"/> through the value blend, the softmax, and the
	/// scores, accumulating into this head's column slice of <paramref name="dQ"/>/<paramref name="dK"/>/
	/// <paramref name="dV"/> (disjoint per head, hence safe to run concurrently).
	/// </summary>
	private void BackwardHead(int h, Matrix dConcat, Matrix dQ, Matrix dK, Matrix dV)
	{
		var offset = h * _headDim;
		var weights = _attnWeights![h];

		// dWeights[i,j] = Σ_d dConcat[i,d] · V[j,d]; and dV[j,d] += Σ_i weights[i,j]·dConcat[i,d].
		var dWeights = new Matrix(_time, _time);
		for (var i = 0; i < _time; i++)
		{
			for (var j = 0; j <= i; j++)
			{
				// dw[i,j] = Σ_d dConcat·V is a clean per-cell dot — accumulate it wide (Mac), same as the
				// forward dots. The dV scatter-add below accumulates across the i loop into a shared cell,
				// not a per-cell dot, so it stays at Real precision for now (a known remaining compounding
				// site, deferred with the other scatter-adds until measurement says it is needed).
				var mac = new Mac();
				for (var d = 0; d < _headDim; d++)
				{
					var g = dConcat[i, offset + d];
					mac.MultiplyAdd(g, _v![j, offset + d]);
					dV[j, offset + d] += weights[i, j] * g;
				}
				dWeights[i, j] = mac.Result;
			}
		}

		// Back through the row-wise softmax to the (masked) scores.
		var dScores = Ops.SoftmaxRowsBackward(dWeights, weights);

		// Scores[i,j] = scale · Σ_d Qn[i,d]·Kn[j,d]  ->  gradients w.r.t. the NORMALIZED Qn, Kn (over
		// unmasked j ≤ i). Collected into small [time, headDim] buffers so they can then be routed back
		// through the QK-norm LayerNorm to get gradients w.r.t. the raw Q, K.
		var dQn = new Matrix(_time, _headDim);
		var dKn = new Matrix(_time, _headDim);
		for (var i = 0; i < _time; i++)
			for (var j = 0; j <= i; j++)
			{
				var ds = dScores[i, j] * _scale;
				for (var d = 0; d < _headDim; d++)
				{
					dQn[i, d] += ds * _kn![j, offset + d];
					dKn[j, d] += ds * _qn![i, offset + d];
				}
			}

		// Back through QK-norm: dQn/dKn (w.r.t. normalized) -> dQ/dK (w.r.t. raw), via the
		// gradient-checked LayerNormBackward on each head's raw slice. Parameter-free, so the γ/β grad
		// buffers are scratch (discarded). Writes into this head's column slice of dQ/dK.
		WriteHeadSliceBackward(_qRaw!, offset, _qMean![h], _qInvStd![h], dQn, dQ);
		WriteHeadSliceBackward(_kRaw!, offset, _kMean![h], _kInvStd![h], dKn, dK);
	}

	/// <summary>
	/// LayerNorm-normalizes rows over the <c>[offset, offset+headDim)</c> columns of <paramref name="src"/>
	/// into <paramref name="dst"/> at the same offset, returning the per-row mean/inv-std stats. Reuses
	/// the gradient-checked <see cref="Ops.LayerNorm"/> by copying the slice through a small buffer.
	/// </summary>
	private (double[] Mean, double[] InvStd) NormalizeHeadSlice(Matrix src, Matrix dst, int offset)
	{
		var slice = new Matrix(_time, _headDim);
		for (var i = 0; i < _time; i++)
			for (var d = 0; d < _headDim; d++)
				slice[i, d] = src[i, offset + d];

		var (normed, mean, invStd) = Ops.LayerNorm(slice, _qkGamma, _qkBeta);

		for (var i = 0; i < _time; i++)
			for (var d = 0; d < _headDim; d++)
				dst[i, offset + d] = normed[i, d];

		return (mean, invStd);
	}

	/// <summary>
	/// Routes a normalized-slice gradient <paramref name="dNormed"/> back through the QK-norm LayerNorm
	/// to the raw slice, adding the result into this head's column slice of <paramref name="dRawDst"/>.
	/// </summary>
	private void WriteHeadSliceBackward(Matrix src, int offset, double[] mean, double[] invStd, Matrix dNormed, Matrix dRawDst)
	{
		var slice = new Matrix(_time, _headDim);
		for (var i = 0; i < _time; i++)
			for (var d = 0; d < _headDim; d++)
				slice[i, d] = src[i, offset + d];

		// Scratch γ/β grads (QK-norm is parameter-free; these are discarded).
		var gGrad = new Matrix(1, _headDim);
		var bGrad = new Matrix(1, _headDim);
		var dRaw = Ops.LayerNormBackward(dNormed, slice, _qkGamma, mean, invStd, gGrad, bGrad);

		for (var i = 0; i < _time; i++)
			for (var d = 0; d < _headDim; d++)
				dRawDst[i, offset + d] += dRaw[i, d];
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
