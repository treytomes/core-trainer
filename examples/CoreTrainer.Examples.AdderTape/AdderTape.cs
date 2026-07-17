using System.Text;

namespace CoreTrainer.Examples.AdderTape;

/// <summary>
/// Generates <b>streaming tape</b> training examples for binary addition: the operands are presented as a
/// queue consumed one pair per step, each step echoes the pair it read and the running carry, and the
/// answer is read directly from the per-step sum digits. There is no indexed operand header, no separate
/// sum line, and no length count — every operation is a fixed local offset on the tape.
/// </summary>
/// <remarks>
/// <para>
/// This is the tape-faithful redesign of the original format, driven by a measured failure. The first
/// format wrote operands in a header block and required the model, at step k, to (a) <b>random-access</b>
/// bit k of the header, (b) <b>count</b> to the bit-width to know when to stop, and (c) <b>scatter-gather</b>
/// the sum digits into a positional sum line. None of those are things a Turing head does — a head reads
/// what is <i>adjacent</i> and consumes the tape <i>in order</i> — and all three broke on held-out lengths
/// (5-bit generalized at 55%, then a cliff to 0% by 8-bit; failures were wrong-bit reads, wrong step
/// counts, and garbage sum gathers). This format removes each:
/// </para>
/// <list type="bullet">
/// <item><b>Queue-consumed operands</b> — the prompt lists one <c>a b</c> pair per line (LSB-first); each
/// work record echoes the pair it consumed, so "which bits to add" is the pair just read, not bit k of a
/// header. No random access.</item>
/// <item><b>Run-out termination</b> — the pairs end, then a single <c>end &lt;carry&gt;</c> record emits the
/// final carry-out, then <c>.</c>. Stopping is "the input ran out", not "I counted to W". No counting.</item>
/// <item><b>Answer is the s-digits</b> — the sum is read straight from the <c>s</c> of each work record in
/// order, plus the <c>end</c> carry, LSB-first. No separate sum line, nothing to gather by position.</item>
/// </list>
/// <para>
/// Carry stays a fixed local offset: each record's carry-in equals the previous record's carry-out
/// (offset −1). Signals are the small fixed alphabet; NoPE (no positional embeddings) is retained, since
/// the format now expresses all needed structure as local/relative tape adjacency rather than absolute
/// index. The length-held-out test is unchanged: train short, test longer.
/// </para>
/// </remarks>
public static class AdderTape
{
	/// <summary>
	/// Renders one addition example as a full streaming tape (prompt + completion). See remarks for the
	/// format. Example (a=3, b=1, 3-bit), LSB-first:
	/// <code>
	/// 1 1
	/// 1 0
	/// 0 0
	/// go
	/// 1 1 c0 s0 k1
	/// 1 0 c1 s0 k1
	/// 0 0 c1 s1 k0
	/// end 0
	/// .
	/// </code>
	/// </summary>
	public static string Render(int a, int b, int bits)
	{
		var sb = new StringBuilder();
		AppendPromptBody(sb, a, b, bits);

		var carry = 0;
		for (var k = 0; k < bits; k++)
		{
			var ak = (a >> k) & 1;
			var bk = (b >> k) & 1;
			var full = ak + bk + carry;
			var s = full & 1;
			var cout = full >> 1;

			// Echo the consumed pair, the carry-in, the sum digit, and the carry-out — all local.
			sb.Append(ak).Append(' ').Append(bk)
				.Append(" c").Append(carry)
				.Append(" s").Append(s)
				.Append(" k").Append(cout)
				.Append('\n');

			carry = cout;
		}

		// Final carry-out as its own record (the high sum bit), then the terminator.
		sb.Append("end ").Append(carry).Append('\n');
		sb.Append(".\n");
		return sb.ToString();
	}

	/// <summary>The prompt prefix: the operand-pair queue and the <c>go</c> marker (everything the model
	/// is given before it must generate).</summary>
	public static string RenderPrompt(int a, int b, int bits)
	{
		var sb = new StringBuilder();
		AppendPromptBody(sb, a, b, bits);
		return sb.ToString();
	}

	// The operand queue, one "a b" pair per line LSB-first, terminated by "go\n".
	private static void AppendPromptBody(StringBuilder sb, int a, int b, int bits)
	{
		for (var k = 0; k < bits; k++)
			sb.Append((a >> k) & 1).Append(' ').Append((b >> k) & 1).Append('\n');
		sb.Append("go\n");
	}
}
