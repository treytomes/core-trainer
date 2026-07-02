namespace CoreTrainer.Examples.MiniGptScratch;

/// <summary>
/// A multiply-accumulate register that sums products at <b>double the fractional width</b> and narrows
/// to <c>Real</c> only once, at the end — the "compute low, accumulate high" rule that makes low-precision
/// training work. It exists because a dot product written as <c>Real acc = 0; acc += a*b;</c> rounds
/// <i>every</i> product to the model grain before adding it, so a K-term sum compounds K roundings. In
/// double that error is ~1e-16 and vanishes; at Fixed32 F=16 it is ~1.5e-5 per term and accumulates into
/// a gradient-<i>direction</i> error that a measured double-vs-Fixed32 control pinned as the cause of
/// Fixed32 training diverging near the loss basin.
/// </summary>
/// <remarks>
/// <para>
/// The accumulator type is chosen per <c>Real</c> at JIT time (the <c>typeof</c> checks fold to a constant,
/// so only one branch survives and there is no boxing in the hot loop):
/// </para>
/// <list type="bullet">
/// <item><b>Fixed32</b> (F=16): products of two int32 raws land at F=32 and sum in a <see cref="long"/>;
/// narrow by <c>&gt;&gt;16</c> once. The product is exact and, at the model's sizes (dot lengths ≤ a few
/// hundred, values O(1)), the F=32 sum stays far inside int64.</item>
/// <item><b>Fixed</b> (F=20): products land at F=40; summed in <see cref="Int128"/> so a 64×64 product
/// cannot overflow, then narrowed by <c>&gt;&gt;20</c>.</item>
/// <item><b>double / float</b>: a plain <see cref="double"/> sum — this is already what a naive loop does,
/// so the wide accumulator is inert (identical result) for the non-fixed builds.</item>
/// </list>
/// </remarks>
public struct Mac
{
	// Which fixed path is live, decided once at JIT from the Real alias (folds to a constant).
	private static readonly bool UseFixed32 = typeof(Real) == typeof(Fixed32);
	private static readonly bool UseFixed64 = typeof(Real) == typeof(Fixed);

	// Only the field for the live Real type is ever touched; the others are elided by the JIT branch.
	private double _d;   // double / float path
	private long _l;     // Fixed32 path: sum of int32×int32 products at F = 2·16 = 32
	private Int128 _i;   // Fixed path:   sum of int64×int64 products at F = 2·20 = 40

	/// <summary>Adds <c>a·b</c> to the running sum at full (doubled-fraction) precision — no per-term rounding.</summary>
	public void MultiplyAdd(Real a, Real b)
	{
		if (UseFixed32)
			_l += (long)To32(a).Raw * To32(b).Raw;          // exact int32×int32 → int64 at F=32
		else if (UseFixed64)
			_i += (Int128)To64(a).Raw * To64(b).Raw;        // exact int64×int64 → Int128 at F=40
		else
			_d += (double)a * (double)b;
	}

	/// <summary>The accumulated sum, narrowed to <c>Real</c> with round-to-nearest — the single rounding of the dot.</summary>
	public readonly Real Result
	{
		get
		{
			if (UseFixed32)
				return From32(Fixed32.FromRaw(NarrowFixed32(_l)));
			if (UseFixed64)
				return From64(Fixed.FromRaw(NarrowFixed(_i)));
			return (Real)_d;
		}
	}

	// Round-to-nearest narrow of an F=32 int64 sum down to an F=16 int32, matching Fixed32's own multiply
	// convention (add half a grain, then arithmetic-shift). Guards the int32 range like Fixed32.op_Multiply.
	private static int NarrowFixed32(long sumF32)
	{
		const int f = Fixed32.FractionalBits;             // 16
		var shifted = (sumF32 + (1L << (f - 1))) >> f;    // → F=16
		if (shifted is > int.MaxValue or < int.MinValue)
			throw new OverflowException(
				$"Mac narrow overflow at F={f}: accumulated dot {(double)sumF32 / (1L << (2 * f))} exceeds int32 range.");
		return (int)shifted;
	}

	// Round-to-nearest narrow of an F=40 Int128 sum down to an F=20 int64.
	private static long NarrowFixed(Int128 sumF40)
	{
		const int f = Fixed.FractionalBits;                       // 20
		var shifted = (sumF40 + ((Int128)1 << (f - 1))) >> f;     // → F=20
		if (shifted > long.MaxValue || shifted < long.MinValue)
			throw new OverflowException(
				$"Mac narrow overflow at F={f}: accumulated dot exceeds int64 range.");
		return (long)shifted;
	}

	// Bridge Real<->fixed via object; only reached on the matching fixed build, where the cast is exact.
	private static Fixed32 To32(Real x) => (Fixed32)(object)x!;
	private static Real From32(Fixed32 x) => (Real)(object)x;
	private static Fixed To64(Real x) => (Fixed)(object)x!;
	private static Real From64(Fixed x) => (Real)(object)x;
}
