using static TorchSharp.torch;

namespace CoreTrainer.Examples.CopyTape;

/// <summary>
/// Builds batched training/eval tensors from copy-machine tapes, loss-masked to the <b>output</b> only
/// (everything after the <c>&gt;</c> marker), never the prompt. The length-generalization split lives in
/// the disjoint bit-widths passed to <see cref="Build"/> (train short, test long).
/// </summary>
public sealed class CopyDataset
{
	/// <summary>One example: token ids, where the output (loss region) begins, and the source values.</summary>
	public readonly record struct Example(long[] Ids, int PromptLen, int Bits, int A, int B);

	public IReadOnlyList<Example> Examples { get; }
	public Tensor Inputs { get; }
	public Tensor Targets { get; }
	public long PadId { get; }
	public int MaxLen { get; }

	private CopyDataset(IReadOnlyList<Example> examples, Tensor inputs, Tensor targets, long padId, int maxLen)
	{
		Examples = examples;
		Inputs = inputs;
		Targets = targets;
		PadId = padId;
		MaxLen = maxLen;
	}

	/// <summary>
	/// Generates copy tapes for each bit-width. Plain/Reverse use one random value per example; Zip uses
	/// two. When the value space fits the cap it is enumerated; otherwise sampled uniformly at random
	/// (seeded, so splits are reproducible and train/held-out can be made independent).
	/// </summary>
	public static CopyDataset Build(CopyMode mode, CharTokenizer tok, int[] bitWidths, int maxPerWidth, int padTo, int seed)
	{
		var padId = (long)tok.VocabSize;
		var examples = new List<Example>();
		var rng = new Random(seed);

		foreach (var bits in bitWidths)
		{
			var range = 1 << bits;
			// Zip draws two independent values; plain/reverse only need one (b is unused but recorded).
			var space = mode == CopyMode.Zip ? (long)range * range : range;

			IEnumerable<(int A, int B)> pairs;
			if (maxPerWidth <= 0 || space <= maxPerWidth)
			{
				pairs = mode == CopyMode.Zip
					? Enumerable.Range(0, range).SelectMany(a => Enumerable.Range(0, range).Select(b => (a, b)))
					: Enumerable.Range(0, range).Select(a => (a, 0));
			}
			else
			{
				var seen = new HashSet<long>();
				var picked = new List<(int, int)>(maxPerWidth);
				while (picked.Count < maxPerWidth)
				{
					var a = rng.Next(range);
					var b = mode == CopyMode.Zip ? rng.Next(range) : 0;
					if (seen.Add((long)a * range + b))
						picked.Add((a, b));
				}
				pairs = picked;
			}

			foreach (var (a, b) in pairs)
			{
				var tape = CopyTask.Render(mode, bits, a, b);
				var prompt = CopyTask.RenderPrompt(mode, bits, a, b);
				examples.Add(new Example(tok.Encode(tape), prompt.Length, bits, a, b));
			}
		}

		var maxLen = examples.Max(e => e.Ids.Length);
		if (padTo < maxLen)
			throw new ArgumentException($"padTo ({padTo}) is smaller than the longest tape ({maxLen}).");

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
				// target[t] = ids[t+1] only inside the output region (t+1 >= promptLen); else ignored.
				var predictsOutput = t + 1 < ids.Length && t + 1 >= promptLen;
				targetData[i, t] = predictsOutput ? ids[t + 1] : padId;
			}
		}

		return new CopyDataset(examples, tensor(inputData), tensor(targetData), padId, maxLen);
	}
}
