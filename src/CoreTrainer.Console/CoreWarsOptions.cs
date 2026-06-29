namespace CoreTrainer.Console;

/// <summary>
/// Configuration options for Core Wars simulator.
/// </summary>
public class CoreWarsOptions
{
	/// <summary>
	/// Size of the core memory arena.
	/// </summary>
	public int CoreSize { get; set; } = 8000;

	/// <summary>
	/// Maximum number of execution cycles before declaring a draw.
	/// </summary>
	public int MaxCycles { get; set; } = 80000;

	/// <summary>
	/// Maximum number of processes per warrior.
	/// </summary>
	public int MaxProcesses { get; set; } = 8000;

	/// <summary>
	/// Minimum circular distance between warrior starting positions.
	/// </summary>
	public int MinWarriorSeparation { get; set; } = 100;
}
