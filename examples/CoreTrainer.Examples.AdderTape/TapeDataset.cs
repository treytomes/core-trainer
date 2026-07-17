using static TorchSharp.torch;

namespace CoreTrainer.Examples.AdderTape;

/// <summary>
/// Builds batched training/eval tensors from adder tapes, with the loss masked to the <b>continuation</b>
/// only (the step-by-step work + sum), never the prompt (operands). Predicting the operands is trivial
/// copying; we want the model graded only on the derivation it must actually compute.
/// </summary>
/// <remarks>
/// The length-generalization split lives here: <see cref="Build"/> is called with disjoint bit-width sets
/// (e.g. train on 1–4 bits, eval on 6–8). Because the tape format is position-agnostic and NoPE removes
/// absolute positions, a longer tape is just "more of the same steps" — so a model that learned the step
/// function should extend to unseen lengths, and one that memorized bounded-depth patterns should not.
/// </remarks>
public sealed class TapeDataset
{
	/// <summary>One example: the full token ids and the index where the continuation (loss region) begins.</summary>
	public readonly record struct Example(long[] Ids, int PromptLen, int A, int B, int Bits);

	/// <summary>All generated examples (unpadded), for eval iteration.</summary>
	public IReadOnlyList<Example> Examples { get; }

	/// <summary>Padded input ids <c>[N, maxLen]</c> and shifted targets, for teacher-forced training.</summary>
	public Tensor Inputs { get; }
	public Tensor Targets { get; }

	/// <summary>The reserved pad id (== vocab size); used as CrossEntropy ignore_index.</summary>
	public long PadId { get; }

	/// <summary>Longest tape length across the examples (defines the padded width / min block size).</summary>
	public int MaxLen { get; }

	private TapeDataset(IReadOnlyList<Example> examples, Tensor inputs, Tensor targets, long padId, int maxLen)
	{
		Examples = examples;
		Inputs = inputs;
		Targets = targets;
		PadId = padId;
		MaxLen = maxLen;
	}

	/// <summary>
	/// Generates every (a, b) pair for each bit-width in <paramref name="bitWidths"/> (capped at
	/// <paramref name="maxPairsPerWidth"/> via a strided sample when the full set is too large), tokenizes
	/// each tape, and packs them into padded, loss-masked tensors.
	/// </summary>
	/// <param name="tokenizer">Vocabulary (shared across splits so ids line up).</param>
	/// <param name="bitWidths">The bit-widths to include (train and eval splits pass disjoint sets).</param>
	/// <param name="maxPairsPerWidth">Cap on (a,b) pairs per width; 0 = all 2^bits · 2^bits pairs.</param>
	/// <param name="padTo">Pad all sequences to this width (use the same value across splits so shapes match).</param>
	/// <param name="seed">Seed for the random pair sample, so splits are reproducible.</param>
	public static TapeDataset Build(CharTokenizer tokenizer, int[] bitWidths, int maxPairsPerWidth, int padTo, int seed)
	{
		var padId = (long)tokenizer.VocabSize; // reserved id beyond the real vocab
		var examples = new List<Example>();
		var rng = new Random(seed);

		foreach (var bits in bitWidths)
		{
			var range = 1 << bits;
			var totalPairs = (long)range * range;

			// Pick (a,b) pairs UNIFORMLY AT RANDOM, not strided. Striding over the a·range+b grid biased the
			// sample toward small a and b∈{0, range/2} — a low-carry, unrepresentative subset that distorted
			// held-out accuracy. When the full grid fits the cap, enumerate it; otherwise sample distinct
			// pairs uniformly so every carry pattern is fairly represented.
			IEnumerable<(int A, int B)> pairs;
			if (maxPairsPerWidth <= 0 || totalPairs <= maxPairsPerWidth)
			{
				pairs = Enumerable.Range(0, range).SelectMany(a => Enumerable.Range(0, range).Select(b => (a, b)));
			}
			else
			{
				var seen = new HashSet<long>();
				var picked = new List<(int, int)>(maxPairsPerWidth);
				while (picked.Count < maxPairsPerWidth)
				{
					var a = rng.Next(range);
					var b = rng.Next(range);
					if (seen.Add((long)a * range + b))
						picked.Add((a, b));
				}
				pairs = picked;
			}

			foreach (var (a, b) in pairs)
			{
				var tape = AdderTape.Render(a, b, bits);
				var promptText = AdderTape.RenderPrompt(a, b, bits);
				var ids = tokenizer.Encode(tape);
				examples.Add(new Example(ids, promptText.Length, a, b, bits));
			}
		}

		var maxLen = examples.Max(e => e.Ids.Length);
		if (padTo < maxLen)
			throw new ArgumentException($"padTo ({padTo}) is smaller than the longest tape ({maxLen}).");

		// Next-token setup: input[i] predicts target[i] = ids[i+1]. Target positions before the
		// continuation (the prompt) and all padding are set to padId, so CrossEntropy(ignore_index=padId)
		// trains ONLY on predicting the work/sum tokens.
		var n = examples.Count;
		var inputData = new long[n, padTo];
		var targetData = new long[n, padTo];
		for (var i = 0; i < n; i++)
		{
			var ids = examples[i].Ids;
			var promptLen = examples[i].PromptLen;
			for (var t = 0; t < padTo; t++)
			{
				inputData[i, t] = t < ids.Length ? ids[t] : padId;
				// target[t] = ids[t+1], but only inside the continuation region (t+1 >= promptLen) and
				// within the real sequence; everything else is ignored.
				var predictsContinuation = t + 1 < ids.Length && t + 1 >= promptLen;
				targetData[i, t] = predictsContinuation ? ids[t + 1] : padId;
			}
		}

		return new TapeDataset(examples, tensor(inputData), tensor(targetData), padId, maxLen);
	}
}
