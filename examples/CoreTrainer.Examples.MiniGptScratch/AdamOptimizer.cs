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
				var g = grad[i];
				m[i] = _beta1 * m[i] + (1.0 - _beta1) * g;
				v[i] = _beta2 * v[i] + (1.0 - _beta2) * g * g;

				var mHat = m[i] / biasCorrection1;
				var vHat = v[i] / biasCorrection2;

				value[i] -= _learningRate * mHat / (Math.Sqrt(vHat) + _epsilon);
			}
		}
	}

	#endregion
}
