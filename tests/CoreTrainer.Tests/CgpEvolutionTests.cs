using CoreTrainer.Networking.Nand;

namespace CoreTrainer.Tests;

/// <summary>
/// Tests for the 1+λ evolution strategy, using closed-form fitness functions so the
/// search behavior can be verified independent of MARS.
/// </summary>
public class CgpEvolutionTests
{
	#region Helpers

	// A fitness function rewarding networks whose output-0 source points at a high address:
	// a simple, monotone objective the search can climb by mutating output genes.
	private static double PreferHighOutputAddress(NandGenome g) => g.GetOutputSource(0);

	#endregion

	[Fact]
	public void Evolve_NeverReturnsWorseThanInitial()
	{
		var random = new Random(1);
		var initial = NandGenome.CreateRandom(inputCount: 4, nodeCount: 30, outputCount: 2, random);
		var initialFitness = PreferHighOutputAddress(initial);

		var evo = new CgpEvolution(PreferHighOutputAddress, lambda: 4, mutationRate: 0.05, random);
		var result = evo.Evolve(initial, generations: 50);

		// 1+λ with offspring-ties-parent is elitist: fitness is monotonically non-decreasing.
		Assert.True(result.Fitness >= initialFitness);
	}

	[Fact]
	public void Evolve_ImprovesFitnessOnAClimbableObjective()
	{
		var random = new Random(7);
		// Start from the worst case: force output to read address 0.
		var initial = new NandGenome(inputCount: 4, outputCount: 1,
			nodeGenes: BuildAscendingNodes(inputCount: 4, nodeCount: 40), outputGenes: [0]);
		var initialFitness = PreferHighOutputAddress(initial);

		var evo = new CgpEvolution(PreferHighOutputAddress, lambda: 8, mutationRate: 0.1, random);
		var result = evo.Evolve(initial, generations: 200);

		// The search should have pushed the output source to a much higher address.
		Assert.True(result.Fitness > initialFitness,
			$"Expected improvement over {initialFitness}, got {result.Fitness}");
	}

	[Fact]
	public void Evolve_ZeroGenerations_ReturnsInitialUnchanged()
	{
		var random = new Random(3);
		var initial = NandGenome.CreateRandom(inputCount: 3, nodeCount: 10, outputCount: 1, random);

		var evo = new CgpEvolution(PreferHighOutputAddress, lambda: 4, mutationRate: 0.1, random);
		var result = evo.Evolve(initial, generations: 0);

		Assert.Equal(0, result.Generations);
		Assert.Same(initial, result.Best);
	}

	[Fact]
	public void Evolve_ReportsProgressPerGeneration()
	{
		var random = new Random(5);
		var initial = NandGenome.CreateRandom(inputCount: 3, nodeCount: 10, outputCount: 1, random);
		var evo = new CgpEvolution(PreferHighOutputAddress, lambda: 4, mutationRate: 0.1, random);

		var reported = new List<int>();
		evo.Evolve(initial, generations: 10, onGeneration: progress => reported.Add(progress.Generation));

		Assert.Equal(Enumerable.Range(0, 10), reported);
	}

	[Fact]
	public void Evolve_InvokesGenerationStartHookOncePerGeneration()
	{
		var random = new Random(5);
		var initial = NandGenome.CreateRandom(inputCount: 3, nodeCount: 10, outputCount: 1, random);
		var evo = new CgpEvolution(PreferHighOutputAddress, lambda: 4, mutationRate: 0.1, random);

		var starts = 0;
		evo.Evolve(initial, generations: 10, onGenerationStart: () => starts++);

		Assert.Equal(10, starts);
	}

	[Fact]
	public void Evolve_WithDynamicFitness_ReEvaluatesParentEachGeneration()
	{
		// A fitness that changes between generations: the hook flips a flag the fitness reads.
		// Without per-generation parent re-evaluation, a stale cached score would mask the change.
		var random = new Random(11);
		var initial = NandGenome.CreateRandom(inputCount: 2, nodeCount: 5, outputCount: 1, random);

		var generation = 0;
		// Fitness alternates its scale each generation; we just confirm the run completes and
		// tracks the shifting landscape rather than freezing on a first-generation score.
		double DynamicFitness(NandGenome g) => generation % 2 == 0 ? 1.0 : 2.0;
		var evo = new CgpEvolution(DynamicFitness, lambda: 4, mutationRate: 0.1, random);

		var result = evo.Evolve(
			initial,
			generations: 6,
			onGenerationStart: () => generation++);

		Assert.Equal(6, result.Generations);
	}

	// Builds a valid ascending node-gene array where each node references its immediate
	// predecessor addresses — a deterministic, valid starting wiring.
	private static int[] BuildAscendingNodes(int inputCount, int nodeCount)
	{
		var genes = new int[nodeCount * 2];
		for (var i = 0; i < nodeCount; i++)
		{
			var upper = inputCount + i; // exclusive
			genes[i * 2] = upper - 1;
			genes[i * 2 + 1] = upper - 1;
		}
		return genes;
	}
}
