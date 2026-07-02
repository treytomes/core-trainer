namespace CoreTrainer.Examples.MiniGptScratch;

/// <summary>
/// A histogram of applied Adam updates bucketed by their magnitude in fixed-point <i>grains</i> (units
/// of the least-significant bit, <c>2^-F</c>). It exists to test one specific theory about why Fixed32
/// F=16 training thrashes instead of converging: near a minimum, Adam's normalized step shrinks toward
/// the grain floor, where a value can only round to <b>0 (frozen), ±1 grain (overshoot floor), or
/// something resolvable (≥2 grains)</b>. If the theory holds, the "frozen" and "±1" buckets swell and
/// the "resolvable" bucket collapses exactly as the loss starts bouncing — turning a hunch into a
/// measured fact, and pointing the fix at update <i>resolution</i> rather than Adam's own arithmetic.
/// </summary>
/// <remarks>
/// The grain count is read from the update's raw fixed-point integer, so it is the update <b>as
/// actually applied to the weight</b> (post-rounding), not a re-derivation from the double it was
/// computed from. For non-fixed <c>Real</c> types (double/float) the update is measured against a
/// reference F=16 grain so the same diagnostic is comparable across precisions. Reset between logging
/// intervals to get a per-interval snapshot.
/// </remarks>
public sealed class AdamUpdateStats
{
	// Reference grain for the double/float builds, so the diagnostic reads on the same F=16 scale as
	// the Fixed32 build it is designed to probe. For Fixed32/Fixed the true raw integer is used instead.
	private const double ReferenceGrain = 1.0 / (1 << 16);

	private long _frozen;      // |update| rounds to exactly 0 grains
	private long _oneGrain;    // |update| is exactly ±1 grain (the overshoot floor)
	private long _resolvable;  // |update| is ≥2 grains (a step Adam can actually modulate)

	/// <summary>Total updates recorded this interval.</summary>
	public long Total => _frozen + _oneGrain + _resolvable;

	/// <summary>Records one applied update, bucketed by its magnitude in grains.</summary>
	public void Record(Real update)
	{
		var grains = GrainMagnitude(update);
		if (grains == 0)
			_frozen++;
		else if (grains == 1)
			_oneGrain++;
		else
			_resolvable++;
	}

	/// <summary>Clears the histogram for the next interval.</summary>
	public void Reset()
	{
		_frozen = 0;
		_oneGrain = 0;
		_resolvable = 0;
	}

	/// <summary>A one-line summary of the current interval's update distribution, as percentages.</summary>
	public string Summary()
	{
		var total = Total;
		if (total == 0)
			return "no updates recorded";

		var pctFrozen = 100.0 * _frozen / total;
		var pctOne = 100.0 * _oneGrain / total;
		var pctResolvable = 100.0 * _resolvable / total;
		return $"updates: frozen(0) {pctFrozen:0.0}%  ±1grain {pctOne:0.0}%  resolvable(≥2) {pctResolvable:0.0}%  (n={total:N0})";
	}

	// |update| in grains. For the fixed-point types this is the absolute raw integer — the exact,
	// already-rounded step. For double/float it is measured against the reference grain (the typeof
	// checks fold at JIT time, so no boxing in the hot path).
	private static long GrainMagnitude(Real update)
	{
		if (typeof(Real) == typeof(Fixed32))
			return Math.Abs((long)((Fixed32)(object)update!).Raw);
		if (typeof(Real) == typeof(Fixed))
			return Math.Abs(((Fixed)(object)update!).Raw);
		return (long)Math.Floor(Math.Abs((double)update) / ReferenceGrain);
	}
}
