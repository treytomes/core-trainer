namespace CoreTrainer.Examples.MiniGptScratch;

/// <summary>
/// The Adam optimizer, hand-rolled: per-parameter adaptive learning rates from running estimates of
/// the gradient's first moment (mean) and second moment (uncentered variance), with bias correction.
/// </summary>
/// <remarks>
/// This is the update rule a tensor library applies inside <c>optimizer.step()</c>. For each element
/// with gradient <c>g</c> at step <c>t</c>:
/// <code>
///   m = β1·m + (1−β1)·g              // first moment
///   v = β2·v + (1−β2)·g²             // second moment
///   m̂ = m / (1 − β1ᵗ),  v̂ = v / (1 − β2ᵗ)   // bias correction
///   param -= lr · m̂ / (√v̂ + ε)
/// </code>
/// The bias correction matters most early on, when <c>m</c> and <c>v</c> are still warming up from
/// their zero initialization.
/// </remarks>
public sealed class AdamOptimizer
{
	#region Fields

	private readonly IReadOnlyList<Parameter> _parameters;
	private readonly double _learningRate;
	private readonly double _beta1;
	private readonly double _beta2;
	private readonly double _epsilon;
	private int _step;

	// Optimizer moment estimates, held in DOUBLE regardless of the model's Real precision. This is
	// deliberate and load-bearing: the second moment v is a running mean of squared gradients, so it is
	// far smaller than the weights themselves (a 1e-3 gradient gives v ≈ 1e-6). If v were stored at the
	// model's grain — e.g. fixed-point F=16, grain 1.5e-5 — it would round to zero, and the next step's
	// mHat/(sqrt(v)+eps) would divide by ~eps and explode the weights (~1e8). Optimizer state is not a
	// model value; keeping it double is how real mixed-precision training avoids exactly this blowup.
	private readonly double[][] _m;
	private readonly double[][] _v;

	#endregion

	#region Constructors

	public AdamOptimizer(
		IReadOnlyList<Parameter> parameters,
		double learningRate,
		double beta1 = 0.9,
		double beta2 = 0.999,
		double epsilon = 1e-8)
	{
		_parameters = parameters;
		_learningRate = learningRate;
		_beta1 = beta1;
		_beta2 = beta2;
		_epsilon = epsilon;

		// Per-parameter double moment buffers, sized to each parameter's element count.
		_m = new double[parameters.Count][];
		_v = new double[parameters.Count][];
		for (var p = 0; p < parameters.Count; p++)
		{
			var count = parameters[p].Value.Data.Length;
			_m[p] = new double[count];
			_v[p] = new double[count];
		}
	}

	#endregion

	#region Methods

	/// <summary>Zeroes the accumulated gradient on every parameter.</summary>
	public void ZeroGrad()
	{
		foreach (var p in _parameters)
			p.ZeroGrad();
	}

	/// <summary>Applies one Adam update to every parameter using its current gradient.</summary>
	public void Step()
	{
		_step++;
		var biasCorrection1 = 1.0 - Math.Pow(_beta1, _step);
		var biasCorrection2 = 1.0 - Math.Pow(_beta2, _step);

		for (var pi = 0; pi < _parameters.Count; pi++)
		{
			var p = _parameters[pi];
			var value = p.Value.Data;
			var grad = p.Grad.Data;
			var m = _m[pi];  // double moment buffers (see field comment) — NOT the Real-typed p.M/p.V
			var v = _v[pi];

			for (var i = 0; i < value.Length; i++)
			{
				// Moments live in double so a tiny second moment (v ≈ g², often << the model grain)
				// cannot round to zero and blow up the mHat/(sqrt(v)+eps) division on the next step.
				double g = (double)grad[i];
				m[i] = _beta1 * m[i] + (1.0 - _beta1) * g;
				v[i] = _beta2 * v[i] + (1.0 - _beta2) * g * g;

				var mHat = m[i] / biasCorrection1;
				var vHat = v[i] / biasCorrection2;

				// Only the applied update is narrowed to Real — the weights genuinely live at model
				// precision, but the state that computed the update did not.
				value[i] -= (Real)(_learningRate * mHat / (Math.Sqrt(vHat) + _epsilon));
			}

			p.MarkUpdated(); // weights changed: invalidate anything derived from them (e.g. cached quantization)
		}
	}

	#endregion
}
