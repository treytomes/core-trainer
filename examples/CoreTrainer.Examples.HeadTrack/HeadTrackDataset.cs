using static TorchSharp.torch;

namespace CoreTrainer.Examples.HeadTrack;

/// <summary>
/// Builds batched training/eval tensors from head-move probe examples, loss-masked to the trace only
/// (everything after the <c>&gt;</c> marker). The length dimension is <b>move count</b>: train on short
/// move sequences, test on longer unseen ones, so held-out accuracy measures how far the head can be
/// tracked (D) beyond the trained range.
/// </summary>
public sealed class HeadTrackDataset
{
	/// <summary>One example: token ids, where the trace (loss region) begins, and the move count.</summary>
	public readonly record struct Example(long[] Ids, int PromptLen, int Moves);

	public IReadOnlyList<Example> Examples { get; }
	public Tensor Inputs { get; }
	public Tensor Targets { get; }
	public long PadId { get; }
	public int MaxLen { get; }

	private HeadTrackDataset(IReadOnlyList<Example> ex, Tensor inputs, Tensor targets, long padId, int maxLen)
	{
		Examples = ex;
		Inputs = inputs;
		Targets = targets;
		PadId = padId;
		MaxLen = maxLen;
	}

	/// <summary>
	/// Generates <paramref name="perCount"/> random examples for each move count in <paramref name="moveCounts"/>,
	/// tokenizes, and packs into padded, trace-masked tensors. Seeded for reproducibility; train and
	/// held-out splits pass disjoint move-count sets (and typically different seeds).
	/// </summary>
	public static HeadTrackDataset Build(CharTokenizer tok, int[] moveCounts, int perCount, int padTo, int seed)
	{
		var padId = (long)tok.VocabSize;
		var rng = new Random(seed);
		var examples = new List<Example>();

		foreach (var moves in moveCounts)
		{
			// Dedup so we don't train on the same (tape, moves) twice at small move counts.
			var seen = new HashSet<string>();
			var target = perCount;
			// A tiny move count has few distinct examples; cap at the space size (2^(moves+1) tapes · 2^moves moves).
			var attempts = 0;
			while (examples.Count(e => e.Moves == moves) < target && attempts < target * 20)
			{
				attempts++;
				var (tape, mv) = HeadTrackTask.Generate(rng, moves);
				if (!seen.Add(tape + "|" + mv))
					continue;
				var text = HeadTrackTask.Render(tape, mv);
				var prompt = HeadTrackTask.RenderPrompt(tape, mv);
				examples.Add(new Example(tok.Encode(text), prompt.Length, moves));
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
				var predictsTrace = t + 1 < ids.Length && t + 1 >= promptLen;
				targetData[i, t] = predictsTrace ? ids[t + 1] : padId;
			}
		}

		return new HeadTrackDataset(examples, tensor(inputData), tensor(targetData), padId, maxLen);
	}
}
