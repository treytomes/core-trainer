namespace CoreTrainer.Examples.MiniGptScratch;

/// <summary>
/// A BitNet-b1.58-style linear layer: the learnable weights are trained in full precision but
/// <b>ternarized to {-1, 0, +1}</b> (with a single per-tensor scale) for the actual matrix multiply.
/// A drop-in replacement for <see cref="Linear"/> — same constructor and forward/backward interface —
/// so the model can be made ternary one layer at a time without touching its wiring.
/// </summary>
/// <remarks>
/// <para>
/// The mechanism (from BitNet b1.58) is <b>quantization-aware training with latent weights</b>. The
/// optimizer never sees the ternary values; it trains the smooth, full-precision "latent" weights in
/// <see cref="Weight"/>. Each forward pass derives fresh ternary weights from them:
/// <code>
///   γ      = mean(|W_latent|)                         // per-tensor absmean scale (full precision)
///   W_tern = round(clip(W_latent / (γ+ε), -1, +1))    // each weight -> {-1, 0, +1}
///   y      = (x · W_tern) · γ  (+ b)
/// </code>
/// </para>
/// <para>
/// Backward uses the <b>straight-through estimator</b>: gradients flow to the latent weights as if
/// the round/clip/scale were the identity, so <c>dW_latent = Xᵀ·dY</c> and
/// <c>dX = dY·(γ·W_tern)ᵀ</c> — exactly the plain-linear rule but with the effective (scaled ternary)
/// weight in the input-gradient path. Adam then updates the latent weights; the ternarization is
/// re-applied next forward. Only weights are quantized here — activations stay full precision (the
/// gentlest, weights-only form of BitNet), which keeps the surrounding transformer's optimization
/// landscape smooth.
/// </para>
/// <para>
/// Note this is why the numerical gradient check does not apply to a BitLinear: perturbing a latent
/// weight by a tiny <c>h</c> almost never changes its rounded ternary value, so the finite-difference
/// gradient is ~0 while the STE gradient is deliberately nonzero. The correct check is that the STE
/// gradient equals the gradient of the <i>un-quantized</i> matmul (i.e. pass-through), not a
/// finite difference.
/// </para>
/// </remarks>
public sealed class BitLinear : ILinearLayer
{
	#region Fields

	private const double Epsilon = 1e-5;

	private readonly bool _hasBias;
	private Matrix? _lastInput;      // saved by forward for the weight gradient
	private Matrix? _effectiveWeight; // γ·W_tern from the last forward, for the input gradient

	#endregion

	#region Constructors

	/// <param name="inputDim">Input feature dimension.</param>
	/// <param name="outputDim">Output feature dimension.</param>
	/// <param name="init">Random source for weight initialization.</param>
	/// <param name="hasBias">Whether to include a learnable (full-precision) bias.</param>
	public BitLinear(int inputDim, int outputDim, Random init, bool hasBias = true)
	{
		// Same full-precision init as Linear — these latent weights are what Adam trains.
		var scale = Math.Sqrt(2.0 / inputDim);
		var w = new Matrix(inputDim, outputDim);
		for (var i = 0; i < w.Data.Length; i++)
			w.Data[i] = NextGaussian(init) * scale;
		Weight = new Parameter(w);

		_hasBias = hasBias;
		if (hasBias)
			Bias = new Parameter(new Matrix(1, outputDim));
	}

	#endregion

	#region Properties

	/// <summary>The full-precision latent weight parameter, shape <c>[inputDim, outputDim]</c>.</summary>
	public Parameter Weight { get; }

	/// <summary>The bias parameter, shape <c>[1, outputDim]</c>, or null when disabled.</summary>
	public Parameter? Bias { get; }

	#endregion

	#region Methods

	/// <summary>Collects this layer's learnable parameters (the latent weight and optional bias).</summary>
	public IEnumerable<Parameter> Parameters()
	{
		yield return Weight;
		if (Bias is not null)
			yield return Bias;
	}

	/// <summary>
	/// Forward pass. Ternarizes the latent weights, multiplies, and rescales:
	/// <c>y = (x · W_tern)·γ (+ b)</c>. Saves the input and the effective weight for backward.
	/// </summary>
	public Matrix Forward(Matrix x)
	{
		_lastInput = x;

		var (trits, gamma) = Quantize();

		// Effective weight = γ · trit; each entry is -γ, 0, or +γ.
		var effective = new Matrix(Weight.Value.Rows, Weight.Value.Cols);
		var eff = effective.Data;
		for (var i = 0; i < trits.Length; i++)
			eff[i] = trits[i] * gamma;
		_effectiveWeight = effective;

		var y = x.MatMul(effective);
		return _hasBias ? y.AddRowVector(Bias!.Value) : y;
	}

	/// <summary>
	/// Ternarizes the current latent weights, returning the per-element trits (each -1, 0, or +1) and
	/// the per-tensor scale γ. This is the exact quantization the forward pass uses, exposed so the
	/// "compile" step can export the same ternary weights the model was trained and evaluated on —
	/// making the exported model bit-identical to inference, not an approximation of it.
	/// </summary>
	public (sbyte[] Trits, double Gamma) Quantize()
	{
		var latent = Weight.Value.Data;
		var absSum = 0.0;
		for (var i = 0; i < latent.Length; i++)
			absSum += Math.Abs(latent[i]);
		var gamma = absSum / latent.Length;
		var inv = 1.0 / (gamma + Epsilon);

		var trits = new sbyte[latent.Length];
		for (var i = 0; i < latent.Length; i++)
		{
			var t = Math.Round(latent[i] * inv, MidpointRounding.AwayFromZero);
			trits[i] = t > 1.0 ? (sbyte)1 : t < -1.0 ? (sbyte)-1 : (sbyte)t;
		}
		return (trits, gamma);
	}

	/// <summary>
	/// Backward pass (straight-through estimator). Accumulates the latent-weight gradient as if the
	/// ternarization were the identity (<c>dW = Xᵀ·dY</c>), accumulates the bias gradient, and returns
	/// the input gradient through the effective (scaled ternary) weight (<c>dX = dY·W_effᵀ</c>).
	/// </summary>
	public Matrix Backward(Matrix dY)
	{
		if (_lastInput is null || _effectiveWeight is null)
			throw new InvalidOperationException("Backward called before Forward.");

		// STE: gradient to the latent weights ignores round/clip/scale — pass straight through.
		var dW = _lastInput.Transpose().MatMul(dY);
		Weight.Grad.AddInPlace(dW);

		if (_hasBias)
			Bias!.Grad.AddInPlace(dY.SumRows());

		// Input gradient uses the effective weight actually used in the forward matmul.
		return dY.MatMul(_effectiveWeight.Transpose());
	}

	/// <summary>Box–Muller standard normal sample.</summary>
	private static double NextGaussian(Random rng)
	{
		var u1 = 1.0 - rng.NextDouble();
		var u2 = 1.0 - rng.NextDouble();
		return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
	}

	#endregion
}
