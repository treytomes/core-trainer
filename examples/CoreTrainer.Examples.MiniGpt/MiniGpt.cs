using Microsoft.Extensions.Logging;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace CoreTrainer.Examples.MiniGpt;

/// <summary>
/// A small GPT-2-style decoder-only transformer language model: token and positional embeddings, a
/// stack of causal transformer blocks, a final layer norm, and a linear head projecting to
/// per-token logits over the vocabulary.
/// </summary>
/// <remarks>
/// The architecture is the standard GPT recipe scaled down to run on CPU for a teaching example.
/// The forward pass consumes a batch of token-id sequences and produces a logit for every vocabulary
/// entry at every position; because the attention is causal, position <c>t</c>'s logits depend only
/// on tokens 0..t, which is exactly the next-token-prediction setup used both for training (compare
/// every position's prediction against the actual next token) and for generation (feed the model its
/// own output one step at a time).
/// </remarks>
public sealed class MiniGpt : nn.Module<Tensor, Tensor>
{
	#region Fields

	private readonly int _blockSize;
	private readonly nn.Module<Tensor, Tensor> _tokenEmbedding;
	private readonly nn.Module<Tensor, Tensor> _positionEmbedding;
	private readonly Sequential _blocks;
	private readonly LayerNorm _lnFinal;
	private readonly nn.Module<Tensor, Tensor> _head;

	#endregion

	#region Constructors

	/// <param name="vocabSize">Number of distinct tokens.</param>
	/// <param name="blockSize">Maximum context length (positions) the model can attend over.</param>
	/// <param name="embedDim">Embedding/model dimension.</param>
	/// <param name="heads">Attention heads per block.</param>
	/// <param name="layers">Number of stacked transformer blocks.</param>
	/// <param name="mlpMode">Feed-forward variant for every block: GELU (baseline) or a logic-gate network.</param>
	/// <param name="logicTemperature">Softmax/anneal temperature for the logic MLP variants.</param>
	/// <param name="useBitLinear">
	/// Use BitNet ternary-weight linears in the transformer <b>blocks</b> (attention + MLP/logic).
	/// </param>
	/// <param name="useBitLinearIO">
	/// Also ternarize the <b>I/O boundary</b> — the token/position embeddings and the LM head. BitNet
	/// deliberately leaves these full precision (they are the most precision-sensitive layers); setting
	/// this true is the aggressive "everything ternary" configuration, false is the standard BitNet recipe.
	/// </param>
	public MiniGpt(int vocabSize, int blockSize, int embedDim, int heads, int layers,
		MlpMode mlpMode = MlpMode.Gelu, double logicTemperature = 1.0, bool useBitLinear = false,
		bool useBitLinearIO = false)
		: base(nameof(MiniGpt))
	{
		_blockSize = blockSize;
		_tokenEmbedding = useBitLinearIO ? new BitEmbedding(vocabSize, embedDim) : nn.Embedding(vocabSize, embedDim);
		_positionEmbedding = useBitLinearIO ? new BitEmbedding(blockSize, embedDim) : nn.Embedding(blockSize, embedDim);

		var blockModules = new List<nn.Module<Tensor, Tensor>>();
		for (var i = 0; i < layers; i++)
			blockModules.Add(new TransformerBlock(embedDim, heads, mlpMode, logicTemperature, useBitLinear));
		_blocks = nn.Sequential(blockModules);

		_lnFinal = nn.LayerNorm(embedDim);
		_head = Linears.Make(embedDim, vocabSize, useBitLinearIO, hasBias: false);

		RegisterComponents();
	}

	#endregion

	#region Properties

	/// <summary>Maximum context length the model can attend over.</summary>
	public int BlockSize => _blockSize;

	#endregion

	#region Logic temperature

	/// <summary>
	/// Sets the softmax/anneal temperature on every block's logic feed-forward (no-op for GELU blocks).
	/// Called each epoch by the trainer to anneal the logic variants from hot to cold.
	/// </summary>
	public void SetLogicTemperature(double temperature)
	{
		foreach (var block in _blocks.children())
			if (block is TransformerBlock tb)
				tb.SetLogicTemperature(temperature);
	}

	/// <summary>Reports the cullability (gate mix) of every block's logic feed-forward (no-op for GELU).</summary>
	public void ReportLogicCullability(Microsoft.Extensions.Logging.ILogger log)
	{
		var layer = 0;
		foreach (var block in _blocks.children())
		{
			if (block is TransformerBlock tb)
				tb.ReportLogicCullability(log, $"block {layer}");
			layer++;
		}
	}

	#endregion

	#region Methods

	/// <summary>
	/// Maps a batch of token-id sequences (shape <c>[batch, time]</c>, dtype int64) to next-token
	/// logits (shape <c>[batch, time, vocabSize]</c>).
	/// </summary>
	public override Tensor forward(Tensor idx)
	{
		using var scope = NewDisposeScope();

		var time = idx.shape[1];
		if (time > _blockSize)
			throw new ArgumentException($"Sequence length {time} exceeds block size {_blockSize}.");

		var positions = arange(time, dtype: ScalarType.Int64, device: idx.device).unsqueeze(0);
		var x = _tokenEmbedding.forward(idx) + _positionEmbedding.forward(positions);
		x = _blocks.forward(x);
		x = _lnFinal.forward(x);
		return _head.forward(x).MoveToOuterDisposeScope();
	}

	/// <summary>
	/// Autoregressively generates <paramref name="maxNewTokens"/> tokens greedily (argmax) starting
	/// from a prompt of token ids (shape <c>[1, promptLen]</c>). Returns the full sequence including
	/// the prompt. The context is cropped to the block size as it grows.
	/// </summary>
	public Tensor Generate(Tensor prompt, int maxNewTokens)
	{
		using var scope = NewDisposeScope();
		eval();

		var idx = prompt;
		for (var step = 0; step < maxNewTokens; step++)
		{
			// Crop to the most recent blockSize tokens so positions stay in range.
			var cropped = idx.shape[1] > _blockSize
				? idx.narrow(1, idx.shape[1] - _blockSize, _blockSize)
				: idx;

			var logits = forward(cropped);
			// Logits for the final position only: [1, vocabSize].
			var lastStep = logits.narrow(1, logits.shape[1] - 1, 1).squeeze(1);
			var next = lastStep.argmax(dim: -1, keepdim: true); // greedy: most likely next token
			idx = cat([idx, next], dim: 1);
		}

		return idx.MoveToOuterDisposeScope();
	}

	#endregion
}
