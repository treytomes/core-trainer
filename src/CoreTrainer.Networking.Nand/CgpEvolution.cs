namespace CoreTrainer.Networking.Nand;

/// <summary>
/// The classic CGP (1+λ) evolutionary strategy: keep one parent, produce λ mutated
/// offspring each generation, and promote the best — with the crucial rule that an
/// offspring which merely <i>ties</i> the parent is preferred over it.
/// </summary>
/// <remarks>
/// <para>
/// The offspring-ties-parent rule is what enables <b>neutral drift</b>: when an offspring
/// is no worse, the search moves to it even though fitness did not improve. Combined with
/// the genome's inactive nodes — whose genes mutate without affecting fitness until they
/// happen to be wired in — this neutral exploration is empirically a large part of why CGP
/// escapes local optima. Without the rule the search stalls on the first plateau.
/// </para>
/// <para>
/// This type is deliberately ignorant of how fitness is computed: it takes a fitness
/// function and maximizes it. That keeps the search loop pure and testable against a
/// closed-form objective, independent of MARS battles.
/// </para>
/// </remarks>
public sealed class CgpEvolution
{
	#region Fields

	private readonly Func<NandGenome, double> _fitness;
	private readonly int _lambda;
	private readonly double _mutationRate;
	private readonly Random _random;

	#endregion

	#region Constructors

	/// <summary>
	/// Creates a 1+λ evolution strategy.
	/// </summary>
	/// <param name="fitness">Fitness function to maximize (higher is better).</param>
	/// <param name="lambda">Number of offspring per generation (at least 1; classic CGP uses 4).</param>
	/// <param name="mutationRate">Per-gene point-mutation probability in [0, 1].</param>
	/// <param name="random">Random source.</param>
	public CgpEvolution(Func<NandGenome, double> fitness, int lambda, double mutationRate, Random random)
	{
		ArgumentNullException.ThrowIfNull(fitness);
		if (lambda < 1)
			throw new ArgumentOutOfRangeException(nameof(lambda), lambda, "Lambda must be at least 1.");
		ArgumentNullException.ThrowIfNull(random);

		_fitness = fitness;
		_lambda = lambda;
		_mutationRate = Math.Clamp(mutationRate, 0.0, 1.0);
		_random = random;
	}

	#endregion

	#region Methods

	/// <summary>
	/// Evolves from a starting genome for a fixed number of generations.
	/// </summary>
	/// <param name="initial">The starting parent genome.</param>
	/// <param name="generations">Number of generations to run (at least 0).</param>
	/// <param name="onGeneration">
	/// Optional callback invoked after each generation with (generationIndex, bestFitness),
	/// for progress reporting.
	/// </param>
	/// <returns>The best genome found and its fitness.</returns>
	public EvolutionResult Evolve(NandGenome initial, int generations, Action<int, double>? onGeneration = null)
	{
		ArgumentNullException.ThrowIfNull(initial);
		if (generations < 0)
			throw new ArgumentOutOfRangeException(nameof(generations), generations, "Generations cannot be negative.");

		var parent = initial;
		var parentFitness = _fitness(parent);

		for (var generation = 0; generation < generations; generation++)
		{
			// Produce λ offspring and find the best of them.
			var bestChild = parent.Mutate(_mutationRate, _random);
			var bestChildFitness = _fitness(bestChild);

			for (var i = 1; i < _lambda; i++)
			{
				var child = parent.Mutate(_mutationRate, _random);
				var childFitness = _fitness(child);
				if (childFitness > bestChildFitness)
				{
					bestChild = child;
					bestChildFitness = childFitness;
				}
			}

			// Promote the best offspring when it ties or beats the parent. The tie case is
			// the neutral-drift rule: equal fitness still moves the search forward.
			if (bestChildFitness >= parentFitness)
			{
				parent = bestChild;
				parentFitness = bestChildFitness;
			}

			onGeneration?.Invoke(generation, parentFitness);
		}

		return new EvolutionResult(parent, parentFitness, generations);
	}

	#endregion
}
