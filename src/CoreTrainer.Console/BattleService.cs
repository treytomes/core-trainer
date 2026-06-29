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
	/// Runs a battle between any number of warriors.
	/// </summary>
	/// <param name="warriors">The warriors to load, in order.</param>
	/// <returns>Index of the winning warrior, or null for a draw / time limit.</returns>
	public int? RunBattle(IReadOnlyList<Warrior> warriors)
	{
		ArgumentNullException.ThrowIfNull(warriors);
		if (warriors.Count == 0)
			throw new ArgumentException("At least one warrior is required.", nameof(warriors));

		_logger.LogInformation("Initializing battle between {WarriorCount} warriors: {Names}",
			warriors.Count, string.Join(", ", warriors.Select(w => w.Name)));

		var mars = new Mars(
			coreSize: _options.CoreSize,
			maxCycles: _options.MaxCycles,
			maxProcesses: _options.MaxProcesses,
			warriorCount: warriors.Count);

		// Choose separated starting positions for all warriors, then load each.
		var positions = WarriorPlacement.GetSeparatedPositions(
			mars.Core.Size, warriors.Count, _options.MinWarriorSeparation, _random);

		for (var i = 0; i < warriors.Count; i++)
		{
			_logger.LogInformation("Loading {Warrior} at position {Position}", warriors[i].Name, positions[i]);
			mars.LoadWarrior(i, warriors[i], positions[i]);
		}

		_logger.LogInformation("Starting battle (max {MaxCycles} cycles, max {MaxProcesses} processes per warrior)",
			_options.MaxCycles, _options.MaxProcesses);

		var winner = mars.Run();

		_logger.LogInformation("Battle finished after {Cycles} cycles", mars.CurrentCycle);

		if (winner.HasValue)
		{
			_logger.LogInformation("Winner: {Winner}", warriors[winner.Value].Name);
		}
		else
		{
			_logger.LogInformation("Result: Draw or time limit reached");
		}

		return winner;
	}

	#endregion
}
