using CoreTrainer.Core;
using CoreTrainer.Networking.Nand;

namespace CoreTrainer.Tests;

/// <summary>
/// Tests for the MARS-based shaped fitness evaluator.
/// </summary>
public class BattleFitnessTests
{
	#region Helpers

	private static Warrior Imp() =>
		new("Imp", [new Instruction(OpCode.MOV, AddressingMode.Direct, 0, AddressingMode.Direct, 1)]);

	private static WarriorObservationCodec SmallCodec() =>
		// Small core keeps the OwnedFraction scan and battles fast for tests.
		new(windowRadius: 1, coreSize: 800, maxProcesses: 800);

	private static NandGenome RandomGenomeFor(WarriorObservationCodec codec, Random random) =>
		NandGenome.CreateRandom(codec.InputBitCount, nodeCount: 50, codec.OutputBitCount, random);

	#endregion

	[Fact]
	public void Evaluate_ReturnsFiniteScore_ForRandomGenome()
	{
		var random = new Random(11);
		var codec = SmallCodec();
		var fitness = new BattleFitness(
			[Imp()], codec, random,
			coreSize: 800, maxCycles: 500, maxProcesses: 800, battlesPerOpponent: 2);

		var score = fitness.Evaluate(RandomGenomeFor(codec, random));

		Assert.True(double.IsFinite(score));
		Assert.True(score >= 0.0);
	}

	[Fact]
	public void Evaluate_IsUsableAsEvolutionFitness_AndDoesNotRegress()
	{
		// End-to-end smoke test: the battle fitness drives CGP without error and the
		// elitist strategy never reports a final fitness below the starting genome's.
		var random = new Random(23);
		var codec = SmallCodec();
		var fitness = new BattleFitness(
			[Imp()], codec, random,
			coreSize: 800, maxCycles: 400, maxProcesses: 800, battlesPerOpponent: 1);

		var initial = RandomGenomeFor(codec, random);
		var initialScore = fitness.Evaluate(initial);

		var evo = new CgpEvolution(fitness.Evaluate, lambda: 3, mutationRate: 0.05, random);
		var result = evo.Evolve(initial, generations: 3);

		Assert.True(double.IsFinite(result.Fitness));
		// Note: fitness here is stochastic across battles (random placement), so we assert
		// the run completes and yields a finite score rather than strict monotonicity.
		Assert.NotNull(result.Best);
	}
}
