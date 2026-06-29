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
	/// Warrior 0 has won the battle.
	/// </summary>
	Warrior0Wins,

	/// <summary>
	/// Warrior 1 has won the battle.
	/// </summary>
	Warrior1Wins,

	/// <summary>
	/// Both warriors are dead (draw).
	/// </summary>
	Draw,

	/// <summary>
	/// Maximum cycle limit reached without a clear winner.
	/// </summary>
	MaxCyclesReached
}
