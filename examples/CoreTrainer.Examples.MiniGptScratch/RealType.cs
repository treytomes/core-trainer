// The numeric precision of the entire from-scratch model is controlled by this one alias. Flip it
// to System.Single and rebuild to run the whole transformer — weights, activations, gradients, Adam
// state — in 32-bit float instead of 64-bit double; flip back to compare. Everything downstream
// refers to `Real` (never `double`/`float` directly) and does its math through `RealMath`, so this
// is the single lever for precision.
global using Real = CoreTrainer.Examples.MiniGptScratch.Fixed32;
// global using Real = CoreTrainer.Examples.MiniGptScratch.Fixed;
// global using Real = double;

namespace CoreTrainer.Examples.MiniGptScratch;

/// <summary>
/// Math functions typed for <c>Real</c>, so the numeric core can switch between double and float by
/// flipping the <c>Real</c> alias without touching call sites. Each computes in double (the BCL
/// <see cref="Math"/> overloads) and narrows to <c>Real</c> on return — a no-op when Real is double,
/// and a compute-in-double-store-in-float step when Real is float.
/// </summary>
/// <remarks>
/// This exists because C# numeric literals are <c>double</c> and mixed <c>float·double</c> arithmetic
/// promotes to <c>double</c>: without Real-typed wrappers (and a few <c>(Real)</c> casts on literal
/// expressions) a float build would not compile. Routing math through here keeps the model code
/// precision-agnostic.
/// </remarks>
public static class RealMath
{
	// Which pure fixed-point path (if any) the transcendentals take, decided once at JIT time from the
	// Real alias. Only one is ever true; for double/float both are false and the double route is used.
	private static readonly bool UseFixed64 = typeof(Real) == typeof(Fixed) && Fixed.UseFixedTranscendentals;
	private static readonly bool UseFixed32 = typeof(Real) == typeof(Fixed32) && Fixed32.UseFixedTranscendentals;

	// Transcendentals route through double by default: convert the Real to double, use the BCL Math
	// overload, convert back. For double this is an identity; for float it computes in double and
	// narrows; for a fixed type it is the "hybrid transcendental island". When the matching UseFixed
	// flag is set, exp/sqrt/tanh instead run in PURE fixed-point via FixedMath / Fixed32Math — no
	// floating point at all on the runtime path.
	public static Real Sqrt(Real x) =>
		UseFixed64 ? From64(FixedMath.Sqrt(To64(x))) :
		UseFixed32 ? From32(Fixed32Math.Sqrt(To32(x))) :
		(Real)Math.Sqrt((double)x);
	public static Real Exp(Real x) =>
		UseFixed64 ? From64(FixedMath.Exp(To64(x))) :
		UseFixed32 ? From32(Fixed32Math.Exp(To32(x))) :
		(Real)Math.Exp((double)x);
	public static Real Tanh(Real x) =>
		UseFixed64 ? From64(FixedMath.Tanh(To64(x))) :
		UseFixed32 ? From32(Fixed32Math.Tanh(To32(x))) :
		(Real)Math.Tanh((double)x);
	// Log and Cos are not on the runtime path (Log = loss metric only; Cos = init only), so they keep
	// the double route — no pure fixed-point version is needed.
	public static Real Log(Real x) => (Real)Math.Log((double)x);
	public static Real Cos(Real x) => (Real)Math.Cos((double)x);

	// Bridge Real<->fixed via object; only reached on the matching fixed build, where the cast is exact.
	private static Fixed To64(Real x) => (Fixed)(object)x!;
	private static Real From64(Fixed x) => (Real)(object)x;
	private static Fixed32 To32(Real x) => (Fixed32)(object)x!;
	private static Real From32(Fixed32 x) => (Real)(object)x;

	// Abs/Max/Min/Round are exact for every Real type and stay in-type (no float round-trip).
	public static Real Abs(Real x) => x < (Real)0 ? -x : x;
	public static Real Max(Real a, Real b) => a > b ? a : b;
	public static Real Min(Real a, Real b) => a < b ? a : b;
	public static Real Round(Real x, MidpointRounding mode) => (Real)Math.Round((double)x, mode);

	/// <summary>
	/// A "smaller than any real value" sentinel for softmax row-max init, argmax init, and the causal
	/// attention mask (a masked score must exp() to ~0). Not every <c>Real</c> has a native −∞:
	/// double/float do; the integer types (<c>Fixed</c>, raw <c>long</c>) use half their minimum, so
	/// that subtracting a normal value from the sentinel (as row-max subtraction does) cannot overflow.
	/// Computed once (the <c>typeof</c> checks fold at JIT time) to avoid boxing in the hot loops.
	/// </summary>
	public static readonly Real NegativeInfinity = MakeNegativeInfinity();

	private static Real MakeNegativeInfinity()
	{
		if (typeof(Real) == typeof(Fixed))
			return (Real)(object)Fixed.NegativeInfinity;
		if (typeof(Real) == typeof(Fixed32))
			return (Real)(object)Fixed32.NegativeInfinity;
		if (typeof(Real) == typeof(long))
			return (Real)(object)(long.MinValue / 2);
		// double / float. A non-const local defeats the compile-time constant conversion that would
		// otherwise reject (Real)double.NegativeInfinity when the Real alias is an integer type.
		var negInf = double.NegativeInfinity;
		return (Real)negInf;
	}
}
