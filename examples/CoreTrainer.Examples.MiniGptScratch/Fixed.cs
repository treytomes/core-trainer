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
	public const int FractionalBits = 20;

	private const long OneRaw = 1L << FractionalBits;
	private const long HalfRaw = 1L << (FractionalBits - 1); // for round-to-nearest on multiply
															 // In Int64Only divide, the numerator is pre-scaled by <<F; it must stay below this to fit in int64.
	private const long NumeratorShiftLimit = 1L << (63 - FractionalBits);

	/// <summary>
	/// When true, all <see cref="Fixed"/> arithmetic stays within 64-bit integers — no <see cref="Int128"/>
	/// intermediates anywhere (multiply and divide both use <c>long</c>). This models the ACTUAL
	/// constraint of the hardware fixed-point targets (microcontrollers, DSPs, FPGAs) that lack an FPU
	/// and have at most a 64-bit integer multiply; the Int128 path, while always safe, is a cheat those
	/// devices cannot do. It also runs faster (a 64-bit multiply is one instruction; Int128 is emulated).
	/// <para>
	/// The catch is that F now bounds the value RANGE, not just the grain: the doubled-scale product
	/// <c>a.Raw·b.Raw</c> (scale 2^2F) must fit in int64, so <c>|value|² &lt; 2^(63-2F)</c> — at F=24,
	/// <c>|value| &lt; ~181</c>. Too high an F overflows (softmax/attention sums); too low underflows
	/// gradients. The arithmetic is <c>checked</c>, so exceeding the range throws
	/// <see cref="OverflowException"/> loudly rather than wrapping silently — pair with
	/// <see cref="TrackRange"/> to find the safe F.
	/// </para>
	/// </summary>
	public static readonly bool Int64Only = true;

	/// <summary>
	/// When true, every constructed value records its magnitude into <see cref="MaxAbsRaw"/>, so a run
	/// can report the actual dynamic range the model uses (to decide whether <see cref="Int64Only"/>
	/// is safe). Adds an interlocked update to the hottest path, so it distorts timing — enable it for a
	/// measurement run, not a performance run. When false the JIT drops the tracking entirely.
	/// </summary>
	public static readonly bool TrackRange = true;

	/// <summary>
	/// When true, the transcendentals (exp/sqrt/tanh) used on the runtime path run in PURE fixed-point
	/// via <see cref="FixedMath"/> — no floating point anywhere in forward/backward. When false, they
	/// take the "hybrid island" route in <see cref="RealMath"/> (convert to double, use <see cref="Math"/>,
	/// convert back), which is the simpler default and the reference the fixed-point versions are checked
	/// against. Only meaningful when <c>Real</c> is <see cref="Fixed"/>; ignored for double/float.
	/// </summary>
	public static readonly bool UseFixedTranscendentals = true;

	/// <summary>
	/// When true (and <see cref="UseFixedTranscendentals"/> is on), the fixed-point transcendentals use
	/// precomputed <see cref="FixedLookupTable"/>s — each table is filled once at init (in double, a
	/// one-time setup cost like weight initialization), and every runtime call becomes an index plus a
	/// linear interpolation instead of a Taylor series / bit-by-bit iteration. This is how no-FPU
	/// hardware accelerates exp/sqrt/tanh. When false, the exact series/iterative versions run instead.
	/// </summary>
	public static readonly bool UseLookupTables = true;

	private static long _maxAbsRaw;

	#endregion

	#region Construction

	/// <summary>The underlying scaled integer.</summary>
	public long Raw { get; }

	private Fixed(long raw)
	{
		Raw = raw;
		if (TrackRange)
			RecordRange(raw);
	}

	// Magnitudes at/above the -inf sentinel (long.MinValue/2 ≈ 4.6e18) are the softmax/argmax "-inf"
	// marker, not values the arithmetic produced — recording them would peg MaxAbsRaw uselessly high
	// and hide the real dynamic range. Anything legitimately this large would already have overflowed
	// the Int64Only multiply, so excluding it loses no real signal.
	private static readonly long SentinelMagnitude = -(long.MinValue / 2);

	// Interlocked running max of |raw|, so the measured range is a trustworthy safety bound even under
	// the parallelized matmul (a racy max could under-report and make Int64Only wrongly look safe).
	private static void RecordRange(long raw)
	{
		var mag = raw < 0 ? -raw : raw;
		if (mag >= SentinelMagnitude)
			return; // the -inf sentinel, not a real value
		var prev = Interlocked.Read(ref _maxAbsRaw);
		while (mag > prev)
		{
			var seen = Interlocked.CompareExchange(ref _maxAbsRaw, mag, prev);
			if (seen == prev)
				break;
			prev = seen;
		}
	}

	/// <summary>The largest <c>|Raw|</c> observed since the last <see cref="ResetRange"/> (when
	/// <see cref="TrackRange"/> is on). Its value is <c>MaxAbsRaw / 2^F</c>; Int64Only multiply is safe
	/// while <c>MaxAbsRaw &lt; ~3.03e9</c>.</summary>
	public static long MaxAbsRaw => _maxAbsRaw;

	/// <summary>Resets the observed range to zero.</summary>
	public static void ResetRange() => _maxAbsRaw = 0;

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

	// Multiply: form the doubled-scale product, round to nearest, then rescale by F bits. Int128 is
	// always safe (the 2^2F intermediate cannot overflow it); the long path is faster but valid only
	// while max|Raw| stays within int64 range for the product — hence the checked multiply.
	public static Fixed operator *(Fixed a, Fixed b)
	{
		if (Int64Only)
		{
			// Single-instruction 64-bit multiply (what fixed-point hardware actually has). `checked`
			// turns a range violation into a loud OverflowException instead of silent wraparound. On
			// overflow, surface the operand VALUES (not just raw), so the crash log shows which factor
			// blew up — the diagnostic that pins the root cause instead of guessing.
			long lproduct;
			try
			{
				lproduct = checked(a.Raw * b.Raw);        // scale 2^2F, must fit in int64
			}
			catch (OverflowException ex)
			{
				throw new OverflowException(
					$"Fixed multiply overflow at F={FractionalBits}: {(double)a} × {(double)b} " +
					$"(raw {a.Raw} × {b.Raw}).", ex);
			}
			lproduct += lproduct >= 0 ? HalfRaw : -HalfRaw; // round to nearest (symmetric)
			return new(lproduct >> FractionalBits);
		}

		var product = (Int128)a.Raw * b.Raw;      // scale 2^2F
		product += product >= 0 ? HalfRaw : -HalfRaw; // round to nearest (symmetric)
		return new((long)(product >> FractionalBits));
	}

	// Divide: pre-scale the numerator by F bits so the quotient lands back at scale 2^F.
	public static Fixed operator /(Fixed a, Fixed b)
	{
		if (Int64Only)
		{
			// The pre-scaled numerator a.Raw<<F must fit in int64. A left shift does NOT throw on
			// overflow (it silently drops high bits), so guard explicitly: |a.Raw| < 2^(63-F).
			if (a.Raw >= NumeratorShiftLimit || a.Raw <= -NumeratorShiftLimit)
				throw new OverflowException(
					$"Fixed divide numerator {a.Raw} too large to pre-scale within int64 at F={FractionalBits}.");
			return new((a.Raw << FractionalBits) / b.Raw);
		}

		return new((long)(((Int128)a.Raw << FractionalBits) / b.Raw));
	}

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
