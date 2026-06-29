namespace CoreTrainer.Core;

/// <summary>
/// Represents the result of a simulation step or complete battle.
/// </summary>
public enum ExecutionResult
{
	/// <summary>
	/// Battle is still in progress.
	/// </summary>
	Continue,

	/// <summary>
	/// Exactly one warrior remains alive and has won. The winning warrior's index is
	/// available from <see cref="Mars.Winner"/>.
	/// </summary>
	Win,

	/// <summary>
	/// All warriors died in the same cycle (draw).
	/// </summary>
	Draw,

	/// <summary>
	/// Maximum cycle limit reached without a clear winner.
	/// </summary>
	MaxCyclesReached
}
