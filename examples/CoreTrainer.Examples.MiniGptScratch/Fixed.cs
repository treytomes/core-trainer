namespace CoreTrainer.Examples.MiniGptScratch;

/// <summary>
/// A fixed-point number: a signed integer <see cref="Raw"/> interpreted as <c>Raw / 2^F</c>, where
/// <c>F</c> = <see cref="FractionalBits"/>. This is the "signed integer with a scale" representation —
/// the same idea as BitNet's ternary weights, generalized to a full number type — used to run the
/// whole transformer with no floating point in its arithmetic.
/// </summary>
/// <remarks>
/// <para>
/// Unlike a float (mantissa × 2^exponent, whose grain grows with magnitude), a fixed-point value has
/// a <b>uniform grain</b> of <c>2^-F</c> everywhere. That is the deliberate trade this experiment
/// explores: give up float's magnitude-adaptive precision for a single, constant resolution.
/// </para>
/// <para>
/// The one subtlety is multiplication. Two values at scale <c>2^F</c> multiply to scale <c>2^2F</c>
/// (<c>aS·bS = ab·S²</c>), so every product is immediately rescaled back down by <c>F</c> bits. The
/// product is formed in <see cref="Int128"/> so the doubled-scale intermediate cannot overflow before
/// that shift — this is what lets <c>F</c> be large without wrecking range. Addition and subtraction
/// need no rescale (same scale in, same scale out).
/// </para>
/// <para>
/// <b>F is the one knob.</b> Larger F = finer grain but smaller representable range (overflow risk in
/// softmax/attention sums); smaller F = wider range but coarser grain (gradient underflow, weights
/// freeze). It must be a compile-time constant (it drives bit-shifts), so tune it here and rebuild.
/// </para>
/// </remarks>
public readonly struct Fixed : IEquatable<Fixed>
{
	#region Configuration

	/// <summary>
	/// Fractional bits: the value is <c>Raw / 2^FractionalBits</c>. Grain = <c>2^-F</c>; representable
	/// range ≈ <c>±2^(63-F)</c>. F=24 gives grain ≈ 6e-8 (finer than float near 1) and range ≈ ±5.5e11.
	/// This is the knob to tune the range/resolution trade — edit and rebuild.
	/// </summary>
	public const int FractionalBits = 24;

	private const long OneRaw = 1L << FractionalBits;
	private const long HalfRaw = 1L << (FractionalBits - 1); // for round-to-nearest on multiply

	#endregion

	#region Construction

	/// <summary>The underlying scaled integer.</summary>
	public long Raw { get; }

	private Fixed(long raw) => Raw = raw;

	/// <summary>Wraps a raw scaled integer directly (no conversion).</summary>
	public static Fixed FromRaw(long raw) => new(raw);

	/// <summary>Zero.</summary>
	public static readonly Fixed Zero = new(0);

	/// <summary>One.</summary>
	public static readonly Fixed One = new(OneRaw);

	/// <summary>
	/// A large-magnitude negative sentinel used as a softmax/argmax "−∞" (fixed-point has no true
	/// infinity). Deliberately <c>long.MinValue / 2</c>, not <c>long.MinValue</c>, so that subtracting
	/// a normal value from it (as row-max subtraction does) cannot overflow.
	/// </summary>
	public static Fixed NegativeInfinity => new(long.MinValue / 2);

	#endregion

	#region Conversions

	// int -> Fixed is implicit (exact and convenient: literals like 0 and 1, one-hot subtraction).
	public static implicit operator Fixed(int value) => new((long)value << FractionalBits);

	// double <-> Fixed is explicit: crossing the fixed/float boundary is a real, lossy step and should
	// be visible in the code (it happens only at init, the transcendental islands, and reporting).
	public static explicit operator Fixed(double value) => new((long)Math.Round(value * OneRaw));

	public static explicit operator double(Fixed value) => (double)value.Raw / OneRaw;

	#endregion

	#region Operators

	public static Fixed operator +(Fixed a, Fixed b) => new(a.Raw + b.Raw);
	public static Fixed operator -(Fixed a, Fixed b) => new(a.Raw - b.Raw);
	public static Fixed operator -(Fixed a) => new(-a.Raw);

	// Multiply: form the doubled-scale product in Int128, round to nearest, then rescale by F bits.
	public static Fixed operator *(Fixed a, Fixed b)
	{
		var product = (Int128)a.Raw * b.Raw;      // scale 2^2F
		product += product >= 0 ? HalfRaw : -HalfRaw; // round to nearest (symmetric)
		return new((long)(product >> FractionalBits));
	}

	// Divide: pre-scale the numerator by F bits so the quotient lands back at scale 2^F.
	public static Fixed operator /(Fixed a, Fixed b) =>
		new((long)(((Int128)a.Raw << FractionalBits) / b.Raw));

	public static bool operator >(Fixed a, Fixed b) => a.Raw > b.Raw;
	public static bool operator <(Fixed a, Fixed b) => a.Raw < b.Raw;
	public static bool operator >=(Fixed a, Fixed b) => a.Raw >= b.Raw;
	public static bool operator <=(Fixed a, Fixed b) => a.Raw <= b.Raw;
	public static bool operator ==(Fixed a, Fixed b) => a.Raw == b.Raw;
	public static bool operator !=(Fixed a, Fixed b) => a.Raw != b.Raw;

	#endregion

	#region Equality

	public bool Equals(Fixed other) => Raw == other.Raw;
	public override bool Equals(object? obj) => obj is Fixed f && Raw == f.Raw;
	public override int GetHashCode() => Raw.GetHashCode();
	public override string ToString() => ((double)this).ToString("0.######");

	#endregion
}
