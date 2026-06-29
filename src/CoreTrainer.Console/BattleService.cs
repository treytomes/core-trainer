using CoreTrainer.Core;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace CoreTrainer.Console;

/// <summary>
/// Service that orchestrates Core Wars battles between warriors.
/// </summary>
public class BattleService
{
	#region Fields

	private readonly ILogger<BattleService> _logger;
	private readonly CoreWarsOptions _options;
	private readonly Random _random;

	#endregion

	public BattleService(ILogger<BattleService> logger, IOptions<CoreWarsOptions> options)
	{
		_logger = logger;
		_options = options.Value;
		_random = new Random();
	}

	#region Methods

	/// <summary>
	/// Runs a battle between two warriors.
	/// </summary>
	/// <param name="warrior1">First warrior.</param>
	/// <param name="warrior2">Second warrior.</param>
	/// <returns>Index of winning warrior (0 or 1), or null for draw.</returns>
	public int? RunBattle(Warrior warrior1, Warrior warrior2)
	{
		_logger.LogInformation("Initializing battle between {Warrior1} and {Warrior2}",
			warrior1.Name, warrior2.Name);

		// Create MARS
		var mars = new Mars(
			coreSize: _options.CoreSize,
			maxCycles: _options.MaxCycles,
			maxProcesses: _options.MaxProcesses);

		// Calculate starting positions with proper circular separation
		var warrior1Pos = _random.Next(mars.Core.Size);
		var warrior2Pos = WarriorPlacement.GetSeparatedPosition(
			mars.Core.Size, warrior1Pos, _options.MinWarriorSeparation, _random);

		_logger.LogInformation("Loading {Warrior1} at position {Position1}", warrior1.Name, warrior1Pos);
		mars.LoadWarrior(0, warrior1, warrior1Pos);

		_logger.LogInformation("Loading {Warrior2} at position {Position2}", warrior2.Name, warrior2Pos);
		mars.LoadWarrior(1, warrior2, warrior2Pos);

		// Calculate circular distance for logging
		var linearDistance = Math.Abs(warrior1Pos - warrior2Pos);
		var circularDistance = Math.Min(linearDistance, mars.Core.Size - linearDistance);
		_logger.LogDebug("Warrior separation: {CircularDistance} cells (linear: {LinearDistance})",
			circularDistance, linearDistance);

		// Run the battle
		_logger.LogInformation("Starting battle (max {MaxCycles} cycles, max {MaxProcesses} processes per warrior)",
			_options.MaxCycles, _options.MaxProcesses);

		var winner = mars.Run();

		_logger.LogInformation("Battle finished after {Cycles} cycles", mars.CurrentCycle);

		if (winner.HasValue)
		{
			var winnerName = winner.Value == 0 ? warrior1.Name : warrior2.Name;
			_logger.LogInformation("Winner: {Winner}", winnerName);
		}
		else
		{
			_logger.LogInformation("Result: Draw or time limit reached");
		}

		return winner;
	}

	#endregion
}
