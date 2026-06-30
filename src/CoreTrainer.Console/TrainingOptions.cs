namespace CoreTrainer.Console;

/// <summary>
/// Configuration for a NAND-warrior training run (the <c>--train</c> mode).
/// </summary>
public class TrainingOptions
{
	/// <summary>
	/// Number of cells on each side of the program counter the warrior observes.
	/// </summary>
	public int WindowRadius { get; set; } = 1;

	/// <summary>
	/// Number of NAND nodes in each candidate network.
	/// </summary>
	public int NodeCount { get; set; } = 200;

	/// <summary>
	/// Number of generations to evolve.
	/// </summary>
	public int Generations { get; set; } = 200;

	/// <summary>
	/// Offspring per generation (the lambda in the 1+lambda strategy).
	/// </summary>
	public int Lambda { get; set; } = 4;

	/// <summary>
	/// Per-gene point-mutation probability.
	/// </summary>
	public double MutationRate { get; set; } = 0.03;

	/// <summary>
	/// Battles fought per opponent when scoring a genome (averaged to damp placement noise).
	/// </summary>
	public int BattlesPerOpponent { get; set; } = 3;

	/// <summary>
	/// Maximum cycles per training battle. Smaller than a showcase battle keeps training fast.
	/// </summary>
	public int MaxCycles { get; set; } = 8000;
}
