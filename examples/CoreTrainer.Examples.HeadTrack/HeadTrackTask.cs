using System.Text;

namespace CoreTrainer.Examples.HeadTrack;

/// <summary>
/// Generates the <b>head-move primitive</b> probe: track a read head through a sequence of left/right
/// moves over a windowed tape, emitting the symbol under the head after <i>each</i> move. This isolates
/// the one operation the scrollable-tape Turing-machine design rests on — "follow relative moves, know
/// what's under the head" — with all computation stripped away.
/// </summary>
/// <remarks>
/// <para>
/// Why this task, and why now: the copy experiment showed a transformer's reliable head-tracking distance
/// (D) is capped and does not, at small size, reach held-out lengths — and that NoPE does not remove the
/// limit, it rebuilds it in the attention dynamics. The proposed way forward is a Turing machine whose
/// <b>control</b> is the transformer and whose <b>tape</b> is an external, scrollable buffer with a window
/// of size ≤ D. That design needs exactly one primitive to work: within a window, track a head through
/// L/R moves and read the cell it lands on. This probe verifies that primitive and <b>measures D</b>.
/// </para>
/// <para>
/// The head moves one cell per step (L/R), never random-access "jump to cell k" — which matters twice:
/// it is the Turing-machine head primitive, and it needs only <i>local/relative</i> position tracking (the
/// thing the copy work suggests is learnable) rather than the global indexing that broke the adder. Left
/// of cell 0 and right of the last cell clamp (defined bounded-tape edge behavior).
/// </para>
/// <para>
/// Format (single-char control tokens so the char tokenizer doesn't fragment them; no spaces, short
/// tapes): <c>&lt;tape&gt;|&lt;moves&gt;&gt;&lt;trace&gt;.</c> — e.g. tape <c>01101</c>, moves <c>RRLR</c>:
/// </para>
/// <code>
/// 01101|RRLR>1111.
/// </code>
/// The trace is the symbol under the head after each move (head starts at cell 0). Emitting the symbol
/// after <i>every</i> move (not just the last) gives a per-step signal: the length of the correct prefix
/// in a failed generation is exactly the number of moves the head was tracked before drifting — i.e. a
/// direct read of D in moves. The tape has <c>moves + 1</c> cells so an all-right walk stays in range.
/// </remarks>
public static class HeadTrackTask
{
	/// <summary>Renders the full probe tape (prompt + completion) for the given tape bits and move string.</summary>
	public static string Render(string tape, string moves) =>
		RenderPrompt(tape, moves) + Trace(tape, moves) + ".\n";

	/// <summary>The prompt prefix the model is given before generating (ends with <c>&gt;</c>).</summary>
	public static string RenderPrompt(string tape, string moves) => $"{tape}|{moves}>";

	/// <summary>The expected trace: symbol under the head after each move (head starts at cell 0, edges clamp).</summary>
	public static string Trace(string tape, string moves)
	{
		var sb = new StringBuilder(moves.Length);
		var pos = 0;
		var n = tape.Length;
		foreach (var m in moves)
		{
			pos = Math.Max(0, Math.Min(n - 1, pos + (m == 'R' ? 1 : -1)));
			sb.Append(tape[pos]);
		}
		return sb.ToString();
	}

	/// <summary>
	/// Generates one random example with <paramref name="moveCount"/> moves: a random 0/1 tape of
	/// <c>moveCount + 1</c> cells and a random L/R move string. The <paramref name="rng"/> makes it seeded.
	/// </summary>
	public static (string Tape, string Moves) Generate(Random rng, int moveCount)
	{
		var tape = new char[moveCount + 1];
		for (var i = 0; i < tape.Length; i++)
			tape[i] = (char)('0' + rng.Next(2));
		var moves = new char[moveCount];
		for (var i = 0; i < moves.Length; i++)
			moves[i] = rng.Next(2) == 0 ? 'L' : 'R';
		return (new string(tape), new string(moves));
	}
}
