using Microsoft.Extensions.Logging;
using static TorchSharp.torch;

namespace CoreTrainer.Examples.MiniGpt;

/// <summary>
/// The "compile" step for the BitNet-style model: discards the full-precision latent weights kept for
/// training and exports every ternary layer as packed trits plus a per-tensor scale — the artifact you
/// would actually deploy — and measures what the compression buys. Ported from the MiniGptScratch
/// example's compiler to the TorchSharp model.
/// </summary>
/// <remarks>
/// Each <see cref="BitLinear"/>/<see cref="BitEmbedding"/> keeps a full-precision latent weight for
/// training and derives ternary values from it each forward pass; for inference only <c>γ · trit</c> is
/// needed. Compiling makes that explicit: each weight becomes a trit in {-1, 0, +1}, each tensor keeps
/// one scale γ. Because the forward pass already computed on exactly these ternarized weights, the
/// compiled model is <b>bit-identical</b> to what was evaluated. Packing is 5 trits/byte (3⁵ = 243 &lt;
/// 256), verified to round-trip losslessly.
/// </remarks>
public static class TernaryCompiler
{
	private const int FloatBytes = 4; // the training-precision baseline (TorchSharp float32 weights)

	/// <summary>Result of compiling: the size comparison and the learned sparsity.</summary>
	public readonly record struct CompileReport(
		long WeightCount, long FullPrecisionBytes, long PackedBytes, long ZeroCount, int LayerCount)
	{
		public double CompressionRatio => PackedBytes == 0 ? 0 : (double)FullPrecisionBytes / PackedBytes;
		public double BitsPerWeight => WeightCount == 0 ? 0 : PackedBytes * 8.0 / WeightCount;
		public double Sparsity => WeightCount == 0 ? 0 : (double)ZeroCount / WeightCount;
	}

	/// <summary>
	/// Compiles every ternary layer of the model: packs its trits, verifies the packing round-trips,
	/// and accumulates the size/sparsity totals. Throws if any packing fails to round-trip. Returns null
	/// if the model has no ternary layers (full-precision build — nothing to compile).
	/// </summary>
	public static CompileReport? Compile(MiniGpt model)
	{
		long weightCount = 0, packedBytes = 0, zeroCount = 0;
		var layerCount = 0;

		foreach (var m in model.modules())
		{
			var trits = m switch
			{
				BitLinear bl => bl.ExtractTrits(),
				BitEmbedding be => be.ExtractTrits(),
				_ => null,
			};
			if (trits is null)
				continue;

			layerCount++;
			weightCount += trits.Length;
			foreach (var t in trits)
				if (t == 0)
					zeroCount++;

			var packed = Pack(trits);
			packedBytes += packed.Length + FloatBytes; // packed trits + the one γ scale per tensor

			var restored = Unpack(packed, trits.Length);
			for (var i = 0; i < trits.Length; i++)
				if (restored[i] != trits[i])
					throw new InvalidOperationException($"Ternary packing round-trip failed at index {i}.");
		}

		if (layerCount == 0)
			return null;
		return new CompileReport(weightCount, weightCount * FloatBytes, packedBytes, zeroCount, layerCount);
	}

	/// <summary>Compiles the model and logs the size/sparsity report (no-op for a full-precision model).</summary>
	public static void CompileAndReport(MiniGpt model, ILogger log)
	{
		var report = Compile(model);
		if (report is null)
		{
			log.LogInformation("  (no ternary layers — nothing to compile)");
			return;
		}

		var r = report.Value;
		log.LogInformation("=== Ternary compile ===");
		log.LogInformation("  Compiled {Layers} ternary layers, {Weights:N0} weights.", r.LayerCount, r.WeightCount);
		log.LogInformation("  Full precision (float32): {Bytes:N0} bytes ({KB:0.0} KB)", r.FullPrecisionBytes, r.FullPrecisionBytes / 1024.0);
		log.LogInformation("  Packed ternary:           {Bytes:N0} bytes ({KB:0.0} KB)  [5 trits/byte + γ per tensor]", r.PackedBytes, r.PackedBytes / 1024.0);
		log.LogInformation("  Compression: {Ratio:0.0}x smaller, {Bits:0.00} effective bits/weight (theoretical min 1.58).", r.CompressionRatio, r.BitsPerWeight);
		log.LogInformation("  Sparsity: {Zeros:N0}/{Total:N0} weights trained to exactly zero ({Pct:P1} pruned).", r.ZeroCount, r.WeightCount, r.Sparsity);
		log.LogInformation("  (Packing verified to round-trip losslessly; compiled model is bit-identical to inference.)");
	}

	/// <summary>Packs ternary values (each -1,0,+1) five to a byte, base-3 (3⁵ = 243 &lt; 256).</summary>
	public static byte[] Pack(sbyte[] trits)
	{
		var byteCount = (trits.Length + 4) / 5;
		var packed = new byte[byteCount];
		for (var i = 0; i < trits.Length; i++)
		{
			var digit = trits[i] + 1;             // {-1,0,+1} -> {0,1,2}
			packed[i / 5] += (byte)(digit * Pow3(i % 5));
		}
		return packed;
	}

	/// <summary>Inverse of <see cref="Pack"/>: recovers <paramref name="count"/> trits.</summary>
	public static sbyte[] Unpack(byte[] packed, int count)
	{
		var trits = new sbyte[count];
		for (var i = 0; i < count; i++)
		{
			var digit = packed[i / 5] / Pow3(i % 5) % 3; // extract base-3 digit
			trits[i] = (sbyte)(digit - 1);               // {0,1,2} -> {-1,0,+1}
		}
		return trits;
	}

	private static int Pow3(int place) => place switch { 0 => 1, 1 => 3, 2 => 9, 3 => 27, _ => 81 };
}
