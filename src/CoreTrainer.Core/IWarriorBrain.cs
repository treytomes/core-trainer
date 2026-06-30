namespace CoreTrainer.Core;

/// <summary>
/// A decision maker that synthesizes the instruction a warrior executes each cycle,
/// rather than running a fixed program from memory.
/// </summary>
/// <remarks>
/// <para>
/// When a warrior has a brain, MARS asks it — each time that warrior is about to
/// execute — for the instruction to place at the current program counter. The
/// instruction is written to the PC (attributed to the warrior) and then executed
/// normally. Warriors without a brain run their in-memory program in the classic way,
/// so brain-driven and standard warriors coexist in the same battle.
/// </para>
/// <para>
/// The brain receives only a read-only observation (the window of cells around the PC
/// plus the warrior's own state). It cannot reach into or advance the simulation, so it
/// is decoupled from the engine and testable with hand-built data.
/// </para>
/// </remarks>
public interface IWarriorBrain
{
	/// <summary>
	/// Number of cells on each side of the program counter the brain wants to observe.
	/// MARS assembles a window of <c>2 * WindowRadius + 1</c> cells to pass to <see cref="Decide"/>.
	/// </summary>
	int WindowRadius { get; }

	/// <summary>
	/// Chooses the instruction to execute at the current program counter.
	/// </summary>
	/// <param name="window">
	/// The cells around the PC, ordered from offset <c>-WindowRadius</c> to
	/// <c>+WindowRadius</c> (length <c>2 * WindowRadius + 1</c>); the center element is the
	/// cell at the PC.
	/// </param>
	/// <param name="programCounter">The warrior's current program counter.</param>
	/// <param name="processCount">The warrior's current number of live processes.</param>
	/// <returns>The instruction to write to the PC and execute.</returns>
	Instruction Decide(IReadOnlyList<Instruction> window, int programCounter, int processCount);
}
