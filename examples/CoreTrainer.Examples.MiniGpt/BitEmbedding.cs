using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace CoreTrainer.Examples.MiniGpt;

/// <summary>
/// A ternary-weight embedding table: a drop-in replacement for <see cref="Embedding"/> whose weight is
/// trained full-precision but ternarized to <c>{-1, 0, +1}</c> with a per-tensor scale for the lookup,
/// via the same straight-through estimator as <see cref="BitLinear"/>.
/// </summary>
/// <remarks>
/// BitNet deliberately leaves embeddings full precision, so ternarizing them is aggressive and expected
/// to be the most precision-sensitive part of an all-ternary model — included here as the upper-bound
/// "everything ternary" data point. A lookup is a row gather, so this just quantizes the table, then
/// indexes it.
/// </remarks>
public sealed class BitEmbedding : Module<Tensor, Tensor>
{
	#region Fields

	private readonly Parameter _weight; // latent full-precision table [numEmbeddings, embedDim]

	#endregion

	#region Constructors

	public BitEmbedding(long numEmbeddings, long embedDim) : base(nameof(BitEmbedding))
	{
		// Standard normal init, matching nn.Embedding's default.
		_weight = new Parameter(randn(numEmbeddings, embedDim));
		RegisterComponents();
	}

	#endregion

	#region Methods

	/// <summary>Looks up rows of the ternarized table for the given int64 indices (any shape).</summary>
	public override Tensor forward(Tensor indices)
	{
		using var scope = NewDisposeScope();
		var w = Ternary.QuantizeSte(_weight);
		return w.index_select(0, indices.reshape(-1))
			.reshape(indices.shape.Append(_weight.shape[1]).ToArray())
			.MoveToOuterDisposeScope();
	}

	/// <summary>The fraction of effective (ternary) table entries that are exactly zero.</summary>
	public double ZeroFraction()
	{
		using var noGrad = no_grad();
		var w = Ternary.QuantizeSte(_weight);
		return w.eq(0).to_type(ScalarType.Float32).mean().item<float>();
	}

	/// <summary>Total element count of the (ternary) table.</summary>
	public long WeightCount() => _weight.numel();

	/// <summary>The trained ternary table entries as trits in <c>{-1, 0, +1}</c>, for the compile step.</summary>
	public sbyte[] ExtractTrits()
	{
		using var noGrad = no_grad();
		var gamma = _weight.abs().mean().item<float>();
		var inv = 1.0 / (gamma + 1e-5);
		var q = (_weight * inv).clamp(-1.0, 1.0).round().to_type(ScalarType.Int32).cpu();
		var flat = q.reshape(-1).data<int>();
		var trits = new sbyte[flat.Count];
		for (var i = 0; i < trits.Length; i++)
			trits[i] = (sbyte)flat[i];
		return trits;
	}

	#endregion
}
