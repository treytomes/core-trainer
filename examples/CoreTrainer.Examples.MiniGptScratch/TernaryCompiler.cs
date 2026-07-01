namespace CoreTrainer.Examples.MiniGptScratch;

/// <summary>
/// The "compile" step for a BitNet-style model: discards the full-precision latent weights kept for
/// training and exports the trained transformer blocks as packed ternary weights plus per-tensor
/// scales — the artifact you would actually deploy. Also measures what the compression buys.
/// </summary>
/// <remarks>
/// <para>
/// During training every <see cref="BitLinear"/> keeps a full-precision latent weight (what Adam
/// updates) and derives ternary values from it each forward pass. Those latents are only needed to
/// train; for inference the model runs purely on <c>γ · trit</c>. Compiling makes that explicit:
/// each weight becomes a trit in {-1, 0, +1} and each tensor keeps one full-precision scale γ.
/// </para>
/// <para>
/// Because the forward pass already computed on exactly these ternarized weights, the compiled model
/// is <b>bit-identical</b> to what was evaluated — compiling changes the storage, not the function.
/// The packing (5 trits per byte, since 3⁵ = 243 &lt; 256) is verified to round-trip losslessly.
/// </para>
/// </remarks>
public static class TernaryCompiler
{
	/// <summary>
	/// Bytes of one training-precision value, for the full-precision size baseline: 8 for double or
	/// the Fixed (long-backed) type, 4 for float. Uses the runtime size of the current <c>Real</c>.
	/// </summary>
	private static readonly int RealBytes = System.Runtime.CompilerServices.Unsafe.SizeOf<Real>();

	/// <summary>Result of compiling: the size comparison and the learned sparsity.</summary>
	public readonly record struct CompileReport(
		long WeightCount,
		long FullPrecisionBytes,
		long PackedBytes,
		long ZeroCount,
		int LayerCount)
	{
		/// <summary>Compression ratio of packed ternary vs. full-precision doubles.</summary>
		public double CompressionRatio => (double)FullPrecisionBytes / PackedBytes;

		/// <summary>Effective bits per weight after packing (packed bytes · 8 / weights).</summary>
		public double BitsPerWeight => PackedBytes * 8.0 / WeightCount;

		/// <summary>Fraction of weights that trained to exactly zero (pruned connections).</summary>
		public double Sparsity => (double)ZeroCount / WeightCount;
	}

	/// <summary>
	/// Compiles every ternary layer of the model: packs its trits, verifies the packing round-trips,
	/// and accumulates the size/sparsity totals. Throws if any packing fails to round-trip.
	/// </summary>
	public static CompileReport Compile(ScratchGpt model)
	{
		long weightCount = 0;
		long packedBytes = 0;
		long zeroCount = 0;
		var layerCount = 0;

		foreach (var layer in model.BlockLinearLayers())
		{
			if (layer is not BitLinear bit)
				continue; // full-precision layers are not compiled

			layerCount++;
			var (trits, _) = bit.Quantize();
			weightCount += trits.Length;
			foreach (var t in trits)
				if (t == 0)
					zeroCount++;

			var packed = Pack(trits);
			packedBytes += packed.Length + RealBytes; // packed trits + the one γ scale per tensor

			// Verify losslessness: unpacking must reproduce the trits exactly.
			var restored = Unpack(packed, trits.Length);
			for (var i = 0; i < trits.Length; i++)
				if (restored[i] != trits[i])
					throw new InvalidOperationException($"Ternary packing round-trip failed at index {i}.");
		}

		var fullPrecisionBytes = weightCount * RealBytes;
		return new CompileReport(weightCount, fullPrecisionBytes, packedBytes, zeroCount, layerCount);
	}

	/// <summary>
	/// Packs ternary values (each -1, 0, +1) five to a byte. Each trit is offset to {0,1,2} and the
	/// five are combined base-3 into a value in [0, 242], which fits one byte (3⁵ = 243).
	/// </summary>
	public static byte[] Pack(sbyte[] trits)
	{
		var byteCount = (trits.Length + 4) / 5;
		var packed = new byte[byteCount];
		for (var i = 0; i < trits.Length; i++)
		{
			var digit = trits[i] + 1;             // {-1,0,+1} -> {0,1,2}
			var slot = i / 5;
			var place = i % 5;
			packed[slot] += (byte)(digit * Pow3(place));
		}
		return packed;
	}

	/// <summary>Inverse of <see cref="Pack"/>: recovers <paramref name="count"/> trits.</summary>
	public static sbyte[] Unpack(byte[] packed, int count)
	{
		var trits = new sbyte[count];
		for (var i = 0; i < count; i++)
		{
			var slot = i / 5;
			var place = i % 5;
			var digit = packed[slot] / Pow3(place) % 3;   // extract base-3 digit
			trits[i] = (sbyte)(digit - 1);                // {0,1,2} -> {-1,0,+1}
		}
		return trits;
	}

	private static int Pow3(int place) => place switch
	{
		0 => 1,
		1 => 3,
		2 => 9,
		3 => 27,
		_ => 81,
	};
}
