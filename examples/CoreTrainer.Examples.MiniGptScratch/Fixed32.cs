namespace CoreTrainer.Examples.MiniGptScratch;

/// <summary>
/// A 32-bit fixed-point number: a signed <see cref="int"/> <see cref="Raw"/> interpreted as
/// <c>Raw / 2^F</c>. The 32-bit sibling of <see cref="Fixed"/> — half the storage, and the multiply
/// maps onto a native SIMD instruction (signed 32×32→64: x86 <c>vpmuldq</c>, ARM <c>vmull_s32</c>),
/// which is the whole reason to drop from 64 to 32 bits.
/// </summary>
/// <remarks>
/// <para>
/// The multiply forms the doubled-scale product in a 64-bit <see cref="long"/> and shifts back down
/// by F. Because a 32×32 product always fits in 64 bits, there is <b>no Int128 path and no overflow
/// guard on the multiply</b> — unlike <see cref="Fixed"/>, where the 64×64 product needed either
/// Int128 or a checked int64 multiply. The narrower storage is what makes this clean.
/// </para>
/// <para>
/// The cost is a much tighter range/grain budget: only 32 bits total, so <c>F</c> fractional bits
/// leave <c>31−F</c> integer bits (value magnitude &lt; <c>2^(31−F)</c>). At F=20 that is
/// <c>|value| &lt; 2048</c> with grain <c>2^-20 ≈ 9.5e-7</c>. Whether the model's values fit is the
/// question the range-measurement run answers before this type is used.
/// </para>
/// </remarks>
public readonly struct Fixed32 : IEquatable<Fixed32>
{
	#region Configuration

	/// <summary>
	/// Fractional bits: the value is <c>Raw / 2^FractionalBits</c>. Grain = <c>2^-F</c>; value range
	/// ≈ <c>±2^(31-F)</c>. Compile-time constant (drives the shifts). Set to match the F the 64-bit
	/// range measurement showed the model's raw values fit within a signed int32.
	/// </summary>
	public const int FractionalBits = 18;

	private const int OneRaw = 1 << FractionalBits;
	private const long HalfRaw = 1L << (FractionalBits - 1); // round-to-nearest addend on the 64-bit product

	/// <summary>
	/// Whether the transcendentals (exp/sqrt/tanh) run in pure fixed-point via <see cref="Fixed32Math"/>
	/// (no floating point on the runtime path) rather than the double "hybrid island". Mirrors
	/// <see cref="Fixed.UseFixedTranscendentals"/> for the 32-bit type.
	/// </summary>
	public static readonly bool UseFixedTranscendentals = true;

	/// <summary>
	/// Whether the pure fixed-point transcendentals use precomputed lookup tables (index + lerp) rather
	/// than the exact series/iterative versions. Mirrors <see cref="Fixed.UseLookupTables"/>.
	/// </summary>
	public static readonly bool UseLookupTables = true;

	#endregion

	#region Construction

	/// <summary>The underlying scaled 32-bit integer.</summary>
	public int Raw { get; }

	private Fixed32(int raw) => Raw = raw;

	/// <summary>Wraps a raw scaled integer directly (no conversion).</summary>
	public static Fixed32 FromRaw(int raw) => new(raw);

	/// <summary>Zero.</summary>
	public static readonly Fixed32 Zero = new(0);

	/// <summary>One.</summary>
	public static readonly Fixed32 One = new(OneRaw);

	/// <summary>
	/// A large-magnitude negative sentinel used as a softmax/argmax "−∞". <c>int.MinValue / 2</c>, not
	/// <c>int.MinValue</c>, so subtracting a normal value from it (row-max subtraction) cannot overflow.
	/// </summary>
	public static Fixed32 NegativeInfinity => new(int.MinValue / 2);

	#endregion

	#region Conversions

	// int -> Fixed32 is implicit (exact for the small literals the model uses: 0, 1, one-hot subtraction).
	public static implicit operator Fixed32(int value) => new(value << FractionalBits);

	// double <-> Fixed32 is explicit: the fixed/float boundary is a real, lossy step (init, transcendental
	// setup, reporting) and should be visible.
	public static explicit operator Fixed32(double value) => new((int)Math.Round(value * OneRaw));

	public static explicit operator double(Fixed32 value) => (double)value.Raw / OneRaw;

	#endregion

	#region Operators

	// Add/sub/narrowing are `checked`: a 32-bit fixed-point value has little range headroom, so an
	// overflow is a real risk while finding the right F — and int arithmetic wraps SILENTLY by default,
	// which would show up as garbage training rather than a fault. `checked` makes an int32 overflow a
	// loud OverflowException that pinpoints the problem (as the 64-bit `Fixed` multiply did). This has a
	// small cost; once an F is validated it could be dropped for speed.
	public static Fixed32 operator +(Fixed32 a, Fixed32 b)
	{
		var sum = (long)a.Raw + b.Raw;
		if (sum > int.MaxValue || sum < int.MinValue)
			throw new OverflowException($"Fixed32 add overflow at F={FractionalBits}: {(double)a} + {(double)b} (exceeds int32 range).");
		return new((int)sum);
	}

	public static Fixed32 operator -(Fixed32 a, Fixed32 b)
	{
		var diff = (long)a.Raw - b.Raw;
		if (diff > int.MaxValue || diff < int.MinValue)
			throw new OverflowException($"Fixed32 subtract overflow at F={FractionalBits}: {(double)a} - {(double)b} (exceeds int32 range).");
		return new((int)diff);
	}

	public static Fixed32 operator -(Fixed32 a) => new(checked(-a.Raw));

	// Multiply: 32×32→64 product (always fits int64 — the intermediate needs no guard), round to
	// nearest, shift back by F, then narrow to int32 (checked: this is where a too-large result
	// overflows). This is the operation that vectorizes natively (vpmuldq / vmull_s32).
	public static Fixed32 operator *(Fixed32 a, Fixed32 b)
	{
		var product = (long)a.Raw * b.Raw;              // scale 2^2F, fits int64 for any int32 operands
		product += product >= 0 ? HalfRaw : -HalfRaw;   // round to nearest (symmetric)
		var shifted = product >> FractionalBits;
		if (shifted > int.MaxValue || shifted < int.MinValue)
			throw new OverflowException(
				$"Fixed32 multiply overflow at F={FractionalBits}: {(double)a} × {(double)b} = {(double)shifted / OneRaw} (exceeds int32 range).");
		return new((int)shifted);
	}

	// Divide: pre-scale the numerator by F in a 64-bit intermediate so the quotient lands at scale 2^F.
	// (int)a.Raw << F always fits int64 (|Raw| < 2^31, F <= ~30), so no guard is needed.
	public static Fixed32 operator /(Fixed32 a, Fixed32 b) =>
		new((int)(((long)a.Raw << FractionalBits) / b.Raw));

	public static bool operator >(Fixed32 a, Fixed32 b) => a.Raw > b.Raw;
	public static bool operator <(Fixed32 a, Fixed32 b) => a.Raw < b.Raw;
	public static bool operator >=(Fixed32 a, Fixed32 b) => a.Raw >= b.Raw;
	public static bool operator <=(Fixed32 a, Fixed32 b) => a.Raw <= b.Raw;
	public static bool operator ==(Fixed32 a, Fixed32 b) => a.Raw == b.Raw;
	public static bool operator !=(Fixed32 a, Fixed32 b) => a.Raw != b.Raw;

	#endregion

	#region Equality

	public bool Equals(Fixed32 other) => Raw == other.Raw;
	public override bool Equals(object? obj) => obj is Fixed32 f && Raw == f.Raw;
	public override int GetHashCode() => Raw.GetHashCode();
	public override string ToString() => ((double)this).ToString("0.######");

	#endregion
}
