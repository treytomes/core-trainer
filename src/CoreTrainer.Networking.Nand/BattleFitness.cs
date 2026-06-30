using CoreTrainer.Core;

namespace CoreTrainer.Networking.Nand;

/// <summary>
/// Scores a NAND genome by running it as a brain-driven warrior against a fixed set of
/// deterministic opponents and combining a shaped fitness signal across the battles.
/// </summary>
/// <remarks>
/// <para>
/// Fitness is dominated by the terminal outcome (win &gt; draw &gt; loss) but shaped by
/// survival time and territory so that even genomes which never win still receive a
/// gradient to climb: a warrior that survives longer or paints more of the core scores
/// higher than one that dies immediately. Without shaping, early random genomes all lose
/// and the search has nothing to follow.
/// </para>
/// <para>
/// Opponents are fixed and deterministic, so the only variation between evaluations comes
/// from starting positions. A configurable number of battles per opponent (with averaged
/// scores) damps that positional noise.
/// </para>
/// </remarks>
public sealed class BattleFitness
{
	#region Constants

	// Outcome base rewards. Win dominates; draw is mildly positive; loss is zero base
	// (shaping still differentiates losers by how well they did).
	private const double WinReward = 1000.0;
	private const double DrawReward = 250.0;
	private const double LossReward = 0.0;

	#endregion

	#region Fields

	private readonly IReadOnlyList<Warrior> _opponents;
	private readonly WarriorObservationCodec _codec;
	private readonly int _coreSize;
	private readonly int _maxCycles;
	private readonly int _maxProcesses;
	private readonly int _battlesPerOpponent;
	private readonly Random _random;

	#endregion

	#region Constructors

	/// <summary>
	/// Creates a battle-based fitness evaluator.
	/// </summary>
	/// <param name="opponents">Fixed opponents each genome is scored against (at least one).</param>
	/// <param name="codec">Observation codec; its widths must match evaluated genomes.</param>
	/// <param name="random">Random source for warrior placement.</param>
	/// <param name="coreSize">Core size for battles.</param>
	/// <param name="maxCycles">Cycle limit per battle.</param>
	/// <param name="maxProcesses">Process limit per warrior.</param>
	/// <param name="battlesPerOpponent">Battles per opponent, averaged to damp positional noise.</param>
	public BattleFitness(
		IReadOnlyList<Warrior> opponents,
		WarriorObservationCodec codec,
		Random random,
		int coreSize = 8000,
		int maxCycles = 80000,
		int maxProcesses = 8000,
		int battlesPerOpponent = 3)
	{
		ArgumentNullException.ThrowIfNull(opponents);
		if (opponents.Count == 0)
			throw new ArgumentException("At least one opponent is required.", nameof(opponents));
		ArgumentNullException.ThrowIfNull(codec);
		ArgumentNullException.ThrowIfNull(random);
		if (battlesPerOpponent < 1)
			throw new ArgumentOutOfRangeException(nameof(battlesPerOpponent), battlesPerOpponent, "Battles per opponent must be at least 1.");

		_opponents = opponents;
		_codec = codec;
		_random = random;
		_coreSize = coreSize;
		_maxCycles = maxCycles;
		_maxProcesses = maxProcesses;
		_battlesPerOpponent = battlesPerOpponent;
	}

	#endregion

	#region Methods

	/// <summary>
	/// Computes the average shaped fitness of a genome across all opponents and battles.
	/// Suitable as the fitness function for <see cref="CgpEvolution"/>.
	/// </summary>
	/// <param name="genome">The genome to evaluate.</param>
	/// <returns>Average fitness (higher is better).</returns>
	public double Evaluate(NandGenome genome)
	{
		ArgumentNullException.ThrowIfNull(genome);

		// The network is deterministic, so it can be built once and reused across battles.
		var network = new NandNetwork(genome);

		var total = 0.0;
		var battles = 0;

		foreach (var opponent in _opponents)
		{
			for (var b = 0; b < _battlesPerOpponent; b++)
			{
				// A fresh brain per battle keeps evaluation free of any cross-battle state.
				var brain = new NandBrain(network, _codec);
				total += ScoreBattle(brain, opponent);
				battles++;
			}
		}

		return total / battles;
	}

	#endregion

	#region Private Methods

	/// <summary>
	/// Runs a single battle of the brain warrior (index 0) against an opponent (index 1)
	/// and returns its shaped score.
	/// </summary>
	private double ScoreBattle(NandBrain brain, Warrior opponent)
	{
		var mars = new Mars(_coreSize, _maxCycles, _maxProcesses, warriorCount: 2);

		var positions = WarriorPlacement.GetSeparatedPositions(_coreSize, 2, _coreSize / 4, _random);

		// The brain warrior is seeded with a single NOP; the brain overwrites it each step.
		var brainWarrior = new Warrior("Nand", [new Instruction(OpCode.NOP, AddressingMode.Direct, 0, AddressingMode.Direct, 0)]);
		mars.LoadWarrior(0, brainWarrior, positions[0], brain);
		mars.LoadWarrior(1, opponent, positions[1]);

		var winner = mars.Run();

		// Base reward from the terminal outcome.
		var score = winner switch
		{
			0 => WinReward,
			1 => LossReward,
			_ => DrawReward // null = draw or cycle-limit
		};

		// Shaping term 1: survival — fraction of the cycle budget that elapsed. A warrior
		// that lasts longer (or drags the battle out) scored more "time alive".
		score += _maxCycles == 0 ? 0.0 : 100.0 * mars.CurrentCycle / _maxCycles;

		// Shaping term 2: territory — share of the core the brain warrior owns at the end.
		score += 100.0 * OwnedFraction(mars, warriorIndex: 0);

		return score;
	}

	/// <summary>
	/// Fraction of core cells currently owned by the given warrior.
	/// </summary>
	private double OwnedFraction(Mars mars, int warriorIndex)
	{
		var owned = 0;
		for (var address = 0; address < mars.Core.Size; address++)
		{
			if (mars.GetCellOwner(address) == warriorIndex)
				owned++;
		}
		return (double)owned / mars.Core.Size;
	}

	#endregion
}
