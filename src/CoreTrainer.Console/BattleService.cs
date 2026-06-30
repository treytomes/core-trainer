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
	/// Runs a battle between any number of combatants.
	/// </summary>
	/// <param name="entries">The combatants to load, in order (warrior plus optional brain).</param>
	/// <returns>Index of the winning combatant, or null for a draw / time limit.</returns>
	public int? RunBattle(IReadOnlyList<BattleEntry> entries)
	{
		ArgumentNullException.ThrowIfNull(entries);
		if (entries.Count == 0)
			throw new ArgumentException("At least one combatant is required.", nameof(entries));

		_logger.LogInformation("Initializing battle between {WarriorCount} warriors: {Names}",
			entries.Count, string.Join(", ", entries.Select(e => e.Warrior.Name)));

		var mars = new Mars(
			coreSize: _options.CoreSize,
			maxCycles: _options.MaxCycles,
			maxProcesses: _options.MaxProcesses,
			warriorCount: entries.Count);

		// Choose separated starting positions for all combatants, then load each.
		var positions = WarriorPlacement.GetSeparatedPositions(
			mars.Core.Size, entries.Count, _options.MinWarriorSeparation, _random);

		for (var i = 0; i < entries.Count; i++)
		{
			_logger.LogInformation("Loading {Warrior} at position {Position}", entries[i].Warrior.Name, positions[i]);
			mars.LoadWarrior(i, entries[i].Warrior, positions[i], entries[i].Brain);
		}

		_logger.LogInformation("Starting battle (max {MaxCycles} cycles, max {MaxProcesses} processes per warrior)",
			_options.MaxCycles, _options.MaxProcesses);

		var winner = mars.Run();

		_logger.LogInformation("Battle finished after {Cycles} cycles", mars.CurrentCycle);

		if (winner.HasValue)
		{
			_logger.LogInformation("Winner: {Winner}", entries[winner.Value].Warrior.Name);
		}
		else
		{
			_logger.LogInformation("Result: Draw or time limit reached");
		}

		return winner;
	}

	#endregion
}
