namespace CoreTrainer.Examples.MiniGptScratch;

/// <summary>
/// The 32-bit counterpart of <see cref="FixedLookupTable"/>: a function tabulated over a fixed domain
/// (sampled once at construction in <c>double</c>) and evaluated at runtime with linear interpolation
/// in pure <see cref="Fixed32"/> arithmetic. Concrete duplication of the 64-bit table is deliberate —
/// this is a legibility-first example, and two straight-line concrete tables read better than a
/// generic-math abstraction that would still need a boxing bridge at the <see cref="RealMath"/> edge.
/// </summary>
public sealed class Fixed32LookupTable
{
	#region Fields

	private readonly Fixed32[] _values;
	private readonly Fixed32 _lo;
	private readonly Fixed32 _invStep;
	private readonly int _maxIndex;

	#endregion

	#region Constructors

	public Fixed32LookupTable(Func<double, double> fn, double lo, double hi, int size)
	{
		if (size < 1)
			throw new ArgumentOutOfRangeException(nameof(size), size, "Table needs at least one interval.");
		if (hi <= lo)
			throw new ArgumentException($"hi ({hi}) must exceed lo ({lo}).");

		var step = (hi - lo) / size;
		_values = new Fixed32[size + 1];
		for (var i = 0; i <= size; i++)
			_values[i] = (Fixed32)fn(lo + i * step);

		_lo = (Fixed32)lo;
		_invStep = (Fixed32)(1.0 / step);
		_maxIndex = size;
	}

	#endregion

	#region Methods

	/// <summary>Evaluates the tabulated function at <paramref name="x"/> via clamped linear interpolation.</summary>
	public Fixed32 Evaluate(Fixed32 x)
	{
		var t = (x - _lo) * _invStep;
		if (t.Raw <= 0)
			return _values[0];

		var index = t.Raw >> Fixed32.FractionalBits; // floor
		if (index >= _maxIndex)
			return _values[_maxIndex];

		var frac = Fixed32.FromRaw(t.Raw - (index << Fixed32.FractionalBits));
		var a = _values[index];
		var b = _values[index + 1];
		return a + frac * (b - a);
	}

	#endregion
}
