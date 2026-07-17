using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace CoreTrainer.Examples.HeadTrack;

/// <summary>
/// A small GPT-2-style decoder-only transformer: token embeddings, a stack of causal blocks, a final
/// layer norm, and an LM head to per-token logits. Plain full-precision GELU model — the fixed substrate
/// for the context-as-tape experiment.
/// </summary>
/// <remarks>
/// <b>NoPE (no positional embeddings).</b> Deliberately there is no position embedding: the causal mask
/// already breaks positional symmetry, and for the length-generalization test it is essential — a learned
/// absolute position embedding would leave positions beyond the training length untrained, so held-out
/// longer tapes would fail for positional reasons rather than algorithmic ones. Decoder-only transformers
/// train fine without positions, and NoPE is the configuration that length-generalizes cleanest
/// (Kazemnejad et al., 2023), which is exactly what this experiment needs to measure.
/// </remarks>
public sealed class MiniGpt : nn.Module<Tensor, Tensor>
{
	private readonly int _blockSize;
	private readonly Embedding _tokenEmbedding;
	private readonly Sequential _blocks;
	private readonly LayerNorm _lnFinal;
	private readonly Linear _head;

	public MiniGpt(int vocabSize, int blockSize, int embedDim, int heads, int layers)
		: base(nameof(MiniGpt))
	{
		_blockSize = blockSize;
		_tokenEmbedding = nn.Embedding(vocabSize, embedDim);

		var blockModules = new List<nn.Module<Tensor, Tensor>>();
		for (var i = 0; i < layers; i++)
			blockModules.Add(new TransformerBlock(embedDim, heads));
		_blocks = nn.Sequential(blockModules);

		_lnFinal = nn.LayerNorm(embedDim);
		_head = nn.Linear(embedDim, vocabSize, hasBias: false);
		RegisterComponents();
	}

	/// <summary>Maximum context length.</summary>
	public int BlockSize => _blockSize;

	/// <summary>Maps token-id sequences <c>[batch, time]</c> to logits <c>[batch, time, vocab]</c>.</summary>
	public override Tensor forward(Tensor idx)
	{
		using var scope = NewDisposeScope();

		var time = idx.shape[1];
		if (time > _blockSize)
			throw new ArgumentException($"Sequence length {time} exceeds block size {_blockSize}.");

		// NoPE: token embeddings only, no positional term added (see class remarks).
		var x = _tokenEmbedding.forward(idx);
		x = _blocks.forward(x);
		x = _lnFinal.forward(x);
		return _head.forward(x).MoveToOuterDisposeScope();
	}

	/// <summary>
	/// Greedy autoregressive generation for a <b>batch</b> of equal-length prompts <c>[batch, promptLen]</c>:
	/// appends the argmax next token <paramref name="maxNewTokens"/> times for every row at once (one
	/// forward pass per step over the whole batch, not per sequence), returning <c>[batch, promptLen +
	/// maxNewTokens]</c>. A fixed token budget with no early stop — the tape is multi-line, so there is no
	/// single stop token; the caller parses the first <c>sum</c> line out of the generated text.
	/// </summary>
	public Tensor GenerateBatch(Tensor prompts, int maxNewTokens)
	{
		using var noGrad = no_grad();
		eval();

		var idx = prompts;
		for (var step = 0; step < maxNewTokens; step++)
		{
			using var stepScope = NewDisposeScope();
			var cropped = idx.shape[1] > _blockSize
				? idx.narrow(1, idx.shape[1] - _blockSize, _blockSize)
				: idx;

			var logits = forward(cropped);                                  // [batch, time, vocab]
			var lastStep = logits.narrow(1, logits.shape[1] - 1, 1).squeeze(1); // [batch, vocab]
			var next = lastStep.argmax(dim: -1, keepdim: true);             // [batch, 1]
			// Move the grown sequence out of this step's scope so it survives; intermediates are freed.
			idx = cat([idx, next], dim: 1).MoveToOuterDisposeScope();
		}

		return idx;
	}
}
