namespace CoreTrainer.Examples.MiniGptScratch;

/// <summary>
/// A precomputed lookup table over a fixed domain, sampled at construction (in <c>double</c>, a
/// one-time setup cost) and evaluated at runtime with linear interpolation in <b>pure fixed-point</b>.
/// This is how real no-FPU hardware accelerates transcendentals: pay the floating-point cost once to
/// fill the table, then every runtime call is an index + a lerp — no series, no iteration, no float.
/// </summary>
/// <remarks>
/// The table stores <c>size + 1</c> samples of <c>fn</c> evenly across <c>[lo, hi]</c>. Evaluation
/// maps <c>x</c> to a fractional index, floors it, and linearly interpolates between the two nearest
/// samples; inputs outside the domain clamp to the endpoints. Building the samples with
/// <see cref="Math"/> at construction is the same "init-time double is allowed" rule the weight
/// initialization uses — nothing here touches floating point once training or inference begins.
/// </remarks>
public sealed class FixedLookupTable
{
	#region Fields

	private readonly Fixed[] _values; // fn sampled at lo + i·step, for i in [0, size]
	private readonly Fixed _lo;
	private readonly Fixed _invStep;  // 1/step, so (x-lo)·invStep is the fractional index
	private readonly int _maxIndex;

	#endregion

	#region Constructors

	/// <param name="fn">The function to tabulate (evaluated in double at construction only).</param>
	/// <param name="lo">Inclusive low end of the tabulated domain.</param>
	/// <param name="hi">Inclusive high end of the tabulated domain.</param>
	/// <param name="size">Number of intervals; the table holds <paramref name="size"/> + 1 samples.</param>
	public FixedLookupTable(Func<double, double> fn, double lo, double hi, int size)
	{
		if (size < 1)
			throw new ArgumentOutOfRangeException(nameof(size), size, "Table needs at least one interval.");
		if (hi <= lo)
			throw new ArgumentException($"hi ({hi}) must exceed lo ({lo}).");

		var step = (hi - lo) / size;
		_values = new Fixed[size + 1];
		for (var i = 0; i <= size; i++)
			_values[i] = (Fixed)fn(lo + i * step);

		_lo = (Fixed)lo;
		_invStep = (Fixed)(1.0 / step);
		_maxIndex = size;
	}

	#endregion

	#region Methods

	/// <summary>
	/// Evaluates the tabulated function at <paramref name="x"/> via clamped linear interpolation, in
	/// pure fixed-point arithmetic.
	/// </summary>
	public Fixed Evaluate(Fixed x)
	{
		// Fractional index t = (x - lo) / step, in [0, size] across the domain.
		var t = (x - _lo) * _invStep;
		if (t.Raw <= 0)
			return _values[0];

		var index = (int)(t.Raw >> Fixed.FractionalBits); // floor
		if (index >= _maxIndex)
			return _values[_maxIndex];

		// Interpolate: values[i] + frac·(values[i+1] − values[i]), frac in [0,1) the fractional index.
		var frac = Fixed.FromRaw(t.Raw - ((long)index << Fixed.FractionalBits));
		var a = _values[index];
		var b = _values[index + 1];
		return a + frac * (b - a);
	}

	#endregion
}
