using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace CoreTrainer.Examples.MiniGpt;

/// <summary>
/// A BitNet-b1.58-style linear layer: a drop-in replacement for <see cref="Linear"/> whose weight is
/// trained full-precision (the "latent" weight the optimizer sees) but <b>ternarized to {-1, 0, +1}</b>
/// with a single per-tensor scale for the matmul, via a straight-through estimator. Activations stay
/// full precision (the weights-only form of BitNet). Bias, when present, stays full precision.
/// </summary>
/// <remarks>
/// The forward pass computes <c>y = x · Wᵀ_ternary (+ b)</c> where the ternary weight is derived fresh
/// each step from the latent weight (see <see cref="Ternary.QuantizeSte"/>). Many latent weights train
/// to values that round to exactly 0, so the effective weight matrix is sparse — the cullability the
/// <see cref="MiniGpt.ReportTernarySparsity"/> report measures.
/// </remarks>
public sealed class BitLinear : Module<Tensor, Tensor>
{
	#region Fields

	private readonly Parameter _weight;      // latent full-precision weight [outFeatures, inFeatures]
	private readonly Parameter? _bias;       // full-precision bias [outFeatures], or null

	#endregion

	#region Constructors

	/// <param name="inFeatures">Input dimension.</param>
	/// <param name="outFeatures">Output dimension.</param>
	/// <param name="hasBias">Whether to include a full-precision bias.</param>
	public BitLinear(long inFeatures, long outFeatures, bool hasBias = true) : base(nameof(BitLinear))
	{
		// Kaiming-uniform-ish init matching nn.Linear's default scale, so BitLinear and Linear start
		// from comparable weight magnitudes.
		var bound = 1.0 / Math.Sqrt(inFeatures);
		_weight = new Parameter(empty(outFeatures, inFeatures).uniform_(-bound, bound));
		if (hasBias)
			_bias = new Parameter(zeros(outFeatures));

		RegisterComponents();
	}

	#endregion

	#region Methods

	/// <summary>Forward pass: ternarize the latent weight (STE), then the standard affine transform.</summary>
	public override Tensor forward(Tensor x)
	{
		using var scope = NewDisposeScope();
		var w = Ternary.QuantizeSte(_weight);
		var y = functional.linear(x, w, _bias);
		return y.MoveToOuterDisposeScope();
	}

	/// <summary>The fraction of effective (ternary) weights that are exactly zero — the pruned share.</summary>
	public double ZeroFraction()
	{
		using var noGrad = no_grad();
		var w = Ternary.QuantizeSte(_weight);
		var zeros = w.eq(0).to_type(ScalarType.Float32).mean().item<float>();
		return zeros;
	}

	/// <summary>Total element count of the (ternary) weight matrix.</summary>
	public long WeightCount() => _weight.numel();

	/// <summary>
	/// The trained ternary weights as trits in <c>{-1, 0, +1}</c> (row-major), for the compile step.
	/// Derived from the latent weight exactly as the forward pass does, so the compiled circuit is
	/// bit-identical to what was evaluated.
	/// </summary>
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
