namespace CoreTrainer.Examples.MiniGptScratch;

/// <summary>
/// The 32-bit counterpart of <see cref="FixedMath"/>: pure <see cref="Fixed32"/> implementations of
/// <c>exp</c>/<c>sqrt</c>/<c>tanh</c> so a 32-bit fixed-point build runs forward and backward with no
/// floating point on the runtime path. Same algorithms as the 64-bit version (base-2 range-reduced
/// exp, bit-by-bit integer sqrt, negative-exponent tanh), duplicated concretely for the narrower type.
/// </summary>
/// <remarks>
/// Duplication rather than generic math is intentional (see <see cref="Fixed32LookupTable"/>): two
/// concrete types do not yet justify an <c>IFixedPoint&lt;T&gt;</c> abstraction, and the
/// <see cref="RealMath"/> boundary would need an object-bridge regardless. If a third fixed-point type
/// appeared, generalizing would then earn its keep.
/// </remarks>
public static class Fixed32Math
{
	#region Constants

	private static readonly Fixed32 InvLn2 = (Fixed32)1.4426950408889634;
	private static readonly Fixed32 Ln2 = (Fixed32)0.6931471805599453;

	private static readonly Fixed32 C2 = (Fixed32)(1.0 / 2.0);
	private static readonly Fixed32 C3 = (Fixed32)(1.0 / 6.0);
	private static readonly Fixed32 C4 = (Fixed32)(1.0 / 24.0);
	private static readonly Fixed32 C5 = (Fixed32)(1.0 / 120.0);

	private static readonly Fixed32 TanhSaturation = (Fixed32)10.0;
	private static readonly Fixed32 ExpUnderflow = (Fixed32)(-30.0);

	private const double SqrtTableHi = 64.0;

	private static readonly Lazy<Fixed32LookupTable> ExpTable =
		new(() => new Fixed32LookupTable(Math.Exp, -30.0, 0.0, 4096));
	private static readonly Lazy<Fixed32LookupTable> SqrtTable =
		new(() => new Fixed32LookupTable(Math.Sqrt, 0.0, SqrtTableHi, 4096));
	private static readonly Lazy<Fixed32LookupTable> TanhTable =
		new(() => new Fixed32LookupTable(Math.Tanh, 0.0, 10.0, 4096));

	#endregion

	#region Methods

	/// <summary>Fixed-point square root of a non-negative value.</summary>
	public static Fixed32 Sqrt(Fixed32 x)
	{
		if (x.Raw <= 0)
			return Fixed32.Zero;
		if (Fixed32.UseLookupTables && x <= (Fixed32)SqrtTableHi)
			return SqrtTable.Value.Evaluate(x);
		return SqrtSeries(x);
	}

	/// <summary>Fixed-point exponential.</summary>
	public static Fixed32 Exp(Fixed32 x)
	{
		if (x.Raw == 0)
			return Fixed32.One;
		if (x <= ExpUnderflow)
			return Fixed32.Zero; // underflows to sub-grain; also guards range reduction from the -inf sentinel
		if (x.Raw > 0)
			return Fixed32.One / Exp(-x);
		return Fixed32.UseLookupTables ? ExpTable.Value.Evaluate(x) : ExpSeries(x);
	}

	/// <summary>Fixed-point hyperbolic tangent via odd symmetry, saturating to ±1 beyond the domain.</summary>
	public static Fixed32 Tanh(Fixed32 x)
	{
		if (x.Raw == 0)
			return Fixed32.Zero;

		var negative = x.Raw < 0;
		var ax = negative ? -x : x;
		if (ax > TanhSaturation)
			return negative ? -Fixed32.One : Fixed32.One;

		var result = Fixed32.UseLookupTables ? TanhTable.Value.Evaluate(ax) : TanhSeries(ax);
		return negative ? -result : result;
	}

	private static Fixed32 SqrtSeries(Fixed32 x)
	{
		// Cast to long BEFORE the shift: (int)Raw << F would overflow int32; the 64-bit intermediate holds it.
		var scaled = (ulong)((long)x.Raw << Fixed32.FractionalBits);
		return Fixed32.FromRaw((int)IntegerSqrt(scaled));
	}

	private static Fixed32 ExpSeries(Fixed32 x)
	{
		var u = x * InvLn2;
		var k = u.Raw >> Fixed32.FractionalBits;                 // floor, <= 0
		var fracRaw = u.Raw - (k << Fixed32.FractionalBits);     // u - k, in [0,1)
		var t = Fixed32.FromRaw(fracRaw) * Ln2;                  // natural units, in [0, ln2)

		var t2 = t * t;
		var t3 = t2 * t;
		var expT = Fixed32.One + t + C2 * t2 + C3 * t3 + C4 * (t2 * t2) + C5 * (t2 * t3);

		var shift = -k;
		if (shift >= 31)
			return Fixed32.Zero;
		return Fixed32.FromRaw(expT.Raw >> shift);
	}

	private static Fixed32 TanhSeries(Fixed32 ax)
	{
		var e = ExpSeries(-(ax + ax));           // e^(-2|x|), argument ≤ 0
		return (Fixed32.One - e) / (Fixed32.One + e);
	}

	#endregion

	#region Helpers

	/// <summary>Integer square root (floor), bit-by-bit — no floating point.</summary>
	private static ulong IntegerSqrt(ulong n)
	{
		ulong result = 0;
		var bit = 1UL << 62;
		while (bit > n)
			bit >>= 2;

		while (bit != 0)
		{
			if (n >= result + bit)
			{
				n -= result + bit;
				result = (result >> 1) + bit;
			}
			else
			{
				result >>= 1;
			}
			bit >>= 2;
		}
		return result;
	}

	#endregion
}
