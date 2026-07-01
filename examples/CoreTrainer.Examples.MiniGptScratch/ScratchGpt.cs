namespace CoreTrainer.Examples.MiniGptScratch;

/// <summary>
/// A small GPT-2-style decoder-only transformer, assembled entirely from hand-rolled pieces with an
/// explicit forward and backward pass and no tensor library. Mirrors the TorchSharp <c>MiniGpt</c>
/// architecture: token + positional embeddings, a stack of causal transformer blocks, a final layer
/// norm, and a linear head to per-token logits.
/// </summary>
/// <remarks>
/// Sequences are processed one at a time (no batch dimension); over a training step the caller sums
/// gradients across sequences by running forward/backward on each before a single optimizer step.
/// The embedding backward is a scatter-add: each position's output gradient is added back onto the
/// row of the token-embedding table for that token and onto the row of the position-embedding table
/// for that slot.
/// </remarks>
public sealed class ScratchGpt
{
	#region Fields

	private readonly int _blockSize;
	private readonly int _embedDim;
	private readonly Parameter _tokenEmbedding;    // [vocab, embedDim]
	private readonly Parameter _positionEmbedding; // [blockSize, embedDim]
	private readonly ScratchBlock[] _blocks;
	private readonly Parameter _lnfGamma, _lnfBeta;
	private readonly Linear _head;                 // [embedDim, vocab], no bias

	// Saved forward state for backward.
	private int[]? _tokenIds;
	private int _time;
	private Matrix? _finalNormInput;
	private double[]? _lnfMean, _lnfInvStd;

	#endregion

	#region Constructors

	/// <param name="linear">
	/// Factory for the transformer blocks' linear projections — pass <see cref="Linear"/> for a
	/// full-precision model or <see cref="BitLinear"/> for BitNet-style ternary weights. The token/
	/// position embeddings and the LM head deliberately stay full precision either way (matching
	/// BitNet, which leaves the most precision-sensitive layers unquantized).
	/// </param>
	public ScratchGpt(int vocabSize, int blockSize, int embedDim, int heads, int layers, Random init, LinearFactory linear)
	{
		_blockSize = blockSize;
		_embedDim = embedDim;

		_tokenEmbedding = new Parameter(RandomMatrix(vocabSize, embedDim, 0.02, init));
		_positionEmbedding = new Parameter(RandomMatrix(blockSize, embedDim, 0.02, init));

		_blocks = new ScratchBlock[layers];
		for (var i = 0; i < layers; i++)
			_blocks[i] = new ScratchBlock(embedDim, heads, init, linear);

		_lnfGamma = OnesRow(embedDim);
		_lnfBeta = new Parameter(new Matrix(1, embedDim));
		_head = new Linear(embedDim, vocabSize, init, hasBias: false);
	}

	#endregion

	#region Properties

	/// <summary>Maximum context length.</summary>
	public int BlockSize => _blockSize;

	#endregion

	#region Methods

	/// <summary>Collects every learnable parameter, for the optimizer.</summary>
	public IReadOnlyList<Parameter> Parameters()
	{
		var ps = new List<Parameter> { _tokenEmbedding, _positionEmbedding };
		foreach (var block in _blocks)
			ps.AddRange(block.Parameters());
		ps.Add(_lnfGamma);
		ps.Add(_lnfBeta);
		ps.AddRange(_head.Parameters());
		return ps;
	}

	/// <summary>
	/// The linear layers inside the transformer blocks — the ones that are ternary under BitNet-style
	/// training. Excludes the embeddings and the LM head, which stay full precision. Used by the
	/// ternary compile step to export and measure the quantized weights.
	/// </summary>
	public IEnumerable<ILinearLayer> BlockLinearLayers()
	{
		foreach (var block in _blocks)
			foreach (var layer in block.LinearLayers())
				yield return layer;
	}

	/// <summary>
	/// Forward pass for one sequence of token ids, returning per-position logits <c>[time, vocab]</c>.
	/// Saves the token ids and final-norm state for the backward pass.
	/// </summary>
	public Matrix Forward(int[] tokenIds)
	{
		if (tokenIds.Length > _blockSize)
			throw new ArgumentException($"Sequence length {tokenIds.Length} exceeds block size {_blockSize}.");

		_tokenIds = tokenIds;
		_time = tokenIds.Length;

		// Embed: x_t = tokenEmbedding[token_t] + positionEmbedding[t].
		var x = new Matrix(_time, _embedDim);
		for (var t = 0; t < _time; t++)
			for (var c = 0; c < _embedDim; c++)
				x[t, c] = _tokenEmbedding.Value[tokenIds[t], c] + _positionEmbedding.Value[t, c];

		foreach (var block in _blocks)
			x = block.Forward(x);

		var (normed, mean, invStd) = Ops.LayerNorm(x, _lnfGamma.Value, _lnfBeta.Value);
		_finalNormInput = x;
		_lnfMean = mean;
		_lnfInvStd = invStd;

		return _head.Forward(normed);
	}

	/// <summary>
	/// Backward pass from the gradient of the loss w.r.t. the logits. Accumulates every parameter
	/// gradient, including scatter-adds into the two embedding tables.
	/// </summary>
	public void Backward(Matrix dLogits)
	{
		if (_tokenIds is null || _finalNormInput is null || _lnfMean is null || _lnfInvStd is null)
			throw new InvalidOperationException("Backward called before Forward.");

		var dNormed = _head.Backward(dLogits);
		var dx = Ops.LayerNormBackward(
			dNormed, _finalNormInput, _lnfGamma.Value, _lnfMean, _lnfInvStd, _lnfGamma.Grad, _lnfBeta.Grad);

		for (var i = _blocks.Length - 1; i >= 0; i--)
			dx = _blocks[i].Backward(dx);

		// Embedding backward: scatter each position's gradient back to its token row and slot row.
		for (var t = 0; t < _time; t++)
			for (var c = 0; c < _embedDim; c++)
			{
				var g = dx[t, c];
				_tokenEmbedding.Grad[_tokenIds[t], c] += g;
				_positionEmbedding.Grad[t, c] += g;
			}
	}

	/// <summary>
	/// Greedy autoregressive generation: starting from <paramref name="prompt"/> token ids, append the
	/// argmax next token <paramref name="maxNewTokens"/> times, cropping context to the block size.
	/// </summary>
	public List<int> Generate(IReadOnlyList<int> prompt, int maxNewTokens)
	{
		var ids = new List<int>(prompt);
		for (var step = 0; step < maxNewTokens; step++)
		{
			var context = ids.Count > _blockSize
				? ids.Skip(ids.Count - _blockSize).ToArray()
				: ids.ToArray();

			var logits = Forward(context);
			var lastRow = logits.Rows - 1;

			// argmax over the vocabulary for the final position.
			var best = 0;
			var bestScore = double.NegativeInfinity;
			for (var v = 0; v < logits.Cols; v++)
			{
				if (logits[lastRow, v] > bestScore)
				{
					bestScore = logits[lastRow, v];
					best = v;
				}
			}
			ids.Add(best);
		}
		return ids;
	}

	private static Matrix RandomMatrix(int rows, int cols, double std, Random rng)
	{
		var m = new Matrix(rows, cols);
		for (var i = 0; i < m.Data.Length; i++)
		{
			var u1 = 1.0 - rng.NextDouble();
			var u2 = 1.0 - rng.NextDouble();
			m.Data[i] = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2) * std;
		}
		return m;
	}

	private static Parameter OnesRow(int dim)
	{
		var m = new Matrix(1, dim);
		m.Fill(1.0);
		return new Parameter(m);
	}

	#endregion
}
