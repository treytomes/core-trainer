namespace CoreTrainer.Examples.CopyTape;

/// <summary>Which head-tracking probe the copy machine runs.</summary>
public enum CopyMode
{
	/// <summary>Emit the input verbatim: pure forward head advance (+1). May be solvable by an induction
	/// head without a real position sense — the "too easy" baseline.</summary>
	Plain,

	/// <summary>Emit the input reversed: the head must locate the end and walk backward. Not a simple
	/// forward induction match, so it needs genuine position tracking.</summary>
	Reverse,

	/// <summary>Interleave two equal-length inputs (a0 b0 a1 b1 ...): two heads advanced in alternation.
	/// The most clearly-positional probe — requires tracking two positions at once.</summary>
	Zip,
}

/// <summary>
/// Generates copy-machine tapes that isolate <b>head-tracking</b> — "which cell am I on, advance the
/// head" — from any computation. This is rung 0 of the revised ladder: the binary adder conflated
/// head-tracking with arithmetic + carry, and both adder formats cliffed on held-out length with failures
/// that were structural (dropped slots, lost record alignment), not arithmetic. Copy strips away the
/// computation so a length-generalization cliff can ONLY be head-tracking.
/// </summary>
/// <remarks>
/// <para>
/// Format: a prompt of bit characters, a single-char control token, then the output and a <c>.</c>
/// terminator. Control tokens are single characters (<c>|</c>, <c>&gt;</c>, <c>.</c>) so the char-level
/// tokenizer doesn't fragment them into multi-token control words, and there are no spaces between bits,
/// keeping tapes short so autoregressive reliability decay doesn't muddy the head-tracking signal.
/// </para>
/// <list type="bullet">
/// <item><b>Plain</b>: <c>0110&gt;</c> → <c>0110.</c></item>
/// <item><b>Reverse</b>: <c>0110&gt;</c> → <c>0110.</c> (here symmetric; e.g. <c>10110&gt;</c> → <c>01101.</c>)</item>
/// <item><b>Zip</b>: <c>0110|1001&gt;</c> → <c>01101001.</c> (a0 b0 a1 b1 …)</item>
/// </list>
/// <para>
/// The three variants form a discriminating sub-ladder: if <b>Plain</b> length-generalizes but
/// <b>Reverse</b>/<b>Zip</b> cliff, the model was faking head-tracking with an induction head on the easy
/// case and lacks a real position register — exactly the structural insight the probe is meant to surface.
/// </para>
/// </remarks>
public static class CopyTask
{
	/// <summary>Renders the full tape (prompt + completion) for the given input length in bits.</summary>
	public static string Render(CopyMode mode, int bits, int a, int b) =>
		RenderPrompt(mode, bits, a, b) + Output(mode, bits, a, b) + ".\n";

	/// <summary>The prompt prefix the model is given before it must generate (ends with <c>&gt;</c>).</summary>
	public static string RenderPrompt(CopyMode mode, int bits, int a, int b)
	{
		var aStr = Bits(a, bits);
		return mode == CopyMode.Zip
			? $"{aStr}|{Bits(b, bits)}>"
			: $"{aStr}>";
	}

	/// <summary>The expected output body (without the terminator), for building targets and scoring.</summary>
	public static string Output(CopyMode mode, int bits, int a, int b)
	{
		var aStr = Bits(a, bits);
		return mode switch
		{
			CopyMode.Plain => aStr,
			CopyMode.Reverse => new string(aStr.Reverse().ToArray()),
			CopyMode.Zip => Interleave(aStr, Bits(b, bits)),
			_ => throw new ArgumentOutOfRangeException(nameof(mode)),
		};
	}

	// LSB-first bit string of `value` over `bits` positions (matches the AdderTape convention).
	private static string Bits(int value, int bits)
	{
		var chars = new char[bits];
		for (var k = 0; k < bits; k++)
			chars[k] = (char)('0' + ((value >> k) & 1));
		return new string(chars);
	}

	// Interleaves two equal-length strings: a0 b0 a1 b1 ...
	private static string Interleave(string a, string b)
	{
		var chars = new char[a.Length + b.Length];
		for (var i = 0; i < a.Length; i++)
		{
			chars[2 * i] = a[i];
			chars[2 * i + 1] = b[i];
		}
		return new string(chars);
	}
}
