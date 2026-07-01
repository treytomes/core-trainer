// The numeric precision of the entire from-scratch model is controlled by this one alias. Flip it
// to System.Single and rebuild to run the whole transformer — weights, activations, gradients, Adam
// state — in 32-bit float instead of 64-bit double; flip back to compare. Everything downstream
// refers to `Real` (never `double`/`float` directly) and does its math through `RealMath`, so this
// is the single lever for precision.
global using Real = CoreTrainer.Examples.MiniGptScratch.Fixed;
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
	// Transcendentals route through double: convert the Real to double, use the BCL Math overload,
	// convert back. For double this is an identity; for float it computes in double and narrows; for
	// Fixed it is the "hybrid transcendental island" — the only place fixed-point arithmetic borrows
	// floating point, isolated to exp/sqrt/tanh/log inside softmax, LayerNorm, and GELU. Every Real
	// type supports the explicit Real<->double conversions this relies on.
	public static Real Sqrt(Real x) => (Real)Math.Sqrt((double)x);
	public static Real Exp(Real x) => (Real)Math.Exp((double)x);
	public static Real Log(Real x) => (Real)Math.Log((double)x);
	public static Real Tanh(Real x) => (Real)Math.Tanh((double)x);
	public static Real Cos(Real x) => (Real)Math.Cos((double)x);

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
		if (typeof(Real) == typeof(long))
			return (Real)(object)(long.MinValue / 2);
		// double / float. A non-const local defeats the compile-time constant conversion that would
		// otherwise reject (Real)double.NegativeInfinity when the Real alias is an integer type.
		var negInf = double.NegativeInfinity;
		return (Real)negInf;
	}
}
