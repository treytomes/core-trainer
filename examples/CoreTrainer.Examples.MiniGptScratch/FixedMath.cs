namespace CoreTrainer.Examples.MiniGptScratch;

/// <summary>
/// Pure fixed-point implementations of the transcendentals (<c>exp</c>, <c>sqrt</c>, <c>tanh</c>)
/// the transformer needs on its runtime path — so a <see cref="Fixed"/> build can run forward and
/// backward with <b>no floating point anywhere</b>, not even the "hybrid island" that
/// <see cref="RealMath"/> otherwise uses (convert to double, call <see cref="Math"/>, convert back).
/// </summary>
/// <remarks>
/// <para>
/// These are the DSP-style approximations that real no-FPU hardware would use: range reduction plus
/// a short polynomial (exp), a bit-by-bit integer square root (sqrt), and a division of exponentials
/// (tanh). Each is designed to stay within the representable range and avoid overflow on the inputs
/// the model actually produces — notably, exp is only ever evaluated on non-positive arguments in
/// softmax (row-max is subtracted first), and tanh routes through a <i>negative</i> exponent so its
/// intermediate never blows up.
/// </para>
/// <para>
/// The constants (1/ln2, ln2, Taylor coefficients) are built once from literals at static init. That
/// is a one-time setup conversion, not floating-point computation on data — analogous to how weight
/// initialization is allowed to use doubles. Nothing here touches <see cref="Math"/> on a per-value
/// basis during training or inference.
/// </para>
/// </remarks>
public static class FixedMath
{
	#region Constants

	private static readonly Fixed InvLn2 = (Fixed)1.4426950408889634;   // 1 / ln(2)
	private static readonly Fixed Ln2 = (Fixed)0.6931471805599453;      // ln(2)

	// Taylor coefficients for exp(t): 1 + t + t²/2 + t³/6 + t⁴/24 + t⁵/120, used via Horner.
	private static readonly Fixed C2 = (Fixed)(1.0 / 2.0);
	private static readonly Fixed C3 = (Fixed)(1.0 / 6.0);
	private static readonly Fixed C4 = (Fixed)(1.0 / 24.0);
	private static readonly Fixed C5 = (Fixed)(1.0 / 120.0);

	// Beyond this magnitude tanh is within one grain of ±1, so clamp rather than compute.
	private static readonly Fixed TanhSaturation = (Fixed)10.0;

	// exp(x) for x below this underflows to less than a grain, so it returns 0. This ALSO guards the
	// range-reduction multiply (x·InvLn2) from the softmax "-inf" sentinel (a huge negative value that
	// would overflow the Int64Only multiply) — masked attention cells legitimately want exp → 0.
	private static readonly Fixed ExpUnderflow = (Fixed)(-30.0);

	#endregion

	#region Methods

	/// <summary>
	/// Fixed-point square root of a non-negative value. <c>sqrt(raw/2^F) = isqrt(raw·2^F)/2^F</c>, so
	/// the raw result is the integer square root of the raw input shifted up by F.
	/// </summary>
	public static Fixed Sqrt(Fixed x)
	{
		if (x.Raw <= 0)
			return Fixed.Zero;

		// raw << F must not overflow int64; for the O(1)-scale values sqrt is used on (LayerNorm
		// variance) this holds comfortably at the F values in play.
		var scaled = (ulong)x.Raw << Fixed.FractionalBits;
		return Fixed.FromRaw((long)IntegerSqrt(scaled));
	}

	/// <summary>
	/// Fixed-point exponential. Base-2 range reduction: <c>exp(x) = 2^(x/ln2) = 2^k · exp(f·ln2)</c>
	/// where k is the integer part of <c>x/ln2</c> and f its fraction in [0,1); the fractional factor
	/// is a short Taylor series and <c>2^k</c> is a bit shift. For positive x it returns
	/// <c>1/exp(-x)</c> so the reduced argument stays bounded.
	/// </summary>
	public static Fixed Exp(Fixed x)
	{
		if (x.Raw == 0)
			return Fixed.One;
		if (x <= ExpUnderflow)
			return Fixed.Zero; // underflows to sub-grain; also guards range reduction from the -inf sentinel
		if (x.Raw > 0)
			return Fixed.One / Exp(-x);

		// x < 0 here. u = x/ln2 < 0; k = floor(u) (arithmetic shift floors toward -inf), frac in [0,1).
		var u = x * InvLn2;
		var k = u.Raw >> Fixed.FractionalBits;                 // floor, <= 0
		var fracRaw = u.Raw - (k << Fixed.FractionalBits);     // u - k, in [0,1)
		var t = Fixed.FromRaw(fracRaw) * Ln2;                  // back to natural units, in [0, ln2)

		// exp(t) for small t in [0, ln2) via Taylor: 1 + t + t²/2 + t³/6 + t⁴/24 + t⁵/120.
		var t2 = t * t;
		var t3 = t2 * t;
		var expT = Fixed.One + t + C2 * t2 + C3 * t3 + C4 * (t2 * t2) + C5 * (t2 * t3);

		// Multiply by 2^k. k <= 0, so shift right by -k; underflow to 0 past the representable range.
		var shift = (int)(-k);
		if (shift >= 63)
			return Fixed.Zero;
		return Fixed.FromRaw(expT.Raw >> shift);
	}

	/// <summary>
	/// Fixed-point hyperbolic tangent. Uses the odd symmetry and the negative-exponent form
	/// <c>tanh(|x|) = (1 − e^(−2|x|)) / (1 + e^(−2|x|))</c>, so the exponential argument is always
	/// ≤ 0 and cannot overflow; saturates to ±1 beyond where it is within a grain of the limit.
	/// </summary>
	public static Fixed Tanh(Fixed x)
	{
		if (x.Raw == 0)
			return Fixed.Zero;

		var negative = x.Raw < 0;
		var ax = negative ? -x : x;
		if (ax > TanhSaturation)
			return negative ? -Fixed.One : Fixed.One;

		var e = Exp(-(ax + ax));                 // e^(-2|x|), argument ≤ 0
		var result = (Fixed.One - e) / (Fixed.One + e);
		return negative ? -result : result;
	}

	#endregion

	#region Helpers

	/// <summary>Integer square root (floor) of a non-negative value, bit-by-bit — no floating point.</summary>
	private static ulong IntegerSqrt(ulong n)
	{
		ulong result = 0;
		// Highest power-of-four ≤ n.
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
