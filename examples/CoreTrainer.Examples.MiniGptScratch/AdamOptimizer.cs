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

		foreach (var p in _parameters)
		{
			var value = p.Value.Data;
			var grad = p.Grad.Data;
			var m = p.M.Data;
			var v = p.V.Data;

			for (var i = 0; i < value.Length; i++)
			{
				// The moment updates run in double for numerical stability (bias correction and the
				// running variance are sensitive), then narrow back to Real when written to the
				// parameter arrays — matching how BitNet keeps optimizer state higher-precision than
				// the weights it updates.
				double g = (double)grad[i];
				var mi = _beta1 * (double)m[i] + (1.0 - _beta1) * g;
				var vi = _beta2 * (double)v[i] + (1.0 - _beta2) * g * g;
				m[i] = (Real)mi;
				v[i] = (Real)vi;

				var mHat = mi / biasCorrection1;
				var vHat = vi / biasCorrection2;

				value[i] -= (Real)(_learningRate * mHat / (Math.Sqrt(vHat) + _epsilon));
			}
		}
	}

	#endregion
}
