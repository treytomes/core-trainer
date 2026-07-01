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
	private readonly AdaptiveMutation? _adaptiveMutation;
	private readonly bool _parallel;
	private readonly Random _random;

	#endregion

	#region Constructors

	/// <summary>
	/// Creates a 1+λ evolution strategy.
	/// </summary>
	/// <param name="fitness">Fitness function to maximize (higher is better).</param>
	/// <param name="lambda">Number of offspring per generation (at least 1; classic CGP uses 4).</param>
	/// <param name="mutationRate">Base per-gene point-mutation probability in [0, 1].</param>
	/// <param name="random">Random source.</param>
	/// <param name="adaptiveMutation">
	/// Optional stagnation-triggered mutation scaling. When supplied, the mutation rate ramps
	/// up while the best fitness is stuck and snaps back to the base rate on any improvement —
	/// a way to escape elitist plateaus. When null, the base rate is used throughout.
	/// </param>
	/// <param name="parallel">
	/// When true, the λ offspring of each generation are mutated and evaluated in parallel.
	/// This requires the <paramref name="fitness"/> function to be thread-safe (callable
	/// concurrently for different genomes). Runs stay reproducible for a given
	/// <paramref name="random"/> seed because each offspring is mutated from its own
	/// pre-drawn seed rather than a shared stream. Leave false (the default) when the fitness
	/// function is not thread-safe.
	/// </param>
	public CgpEvolution(
		Func<NandGenome, double> fitness,
		int lambda,
		double mutationRate,
		Random random,
		AdaptiveMutation? adaptiveMutation = null,
		bool parallel = false)
	{
		ArgumentNullException.ThrowIfNull(fitness);
		if (lambda < 1)
			throw new ArgumentOutOfRangeException(nameof(lambda), lambda, "Lambda must be at least 1.");
		ArgumentNullException.ThrowIfNull(random);

		_fitness = fitness;
		_lambda = lambda;
		_mutationRate = Math.Clamp(mutationRate, 0.0, 1.0);
		_adaptiveMutation = adaptiveMutation;
		_parallel = parallel;
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
	/// Optional callback invoked after each generation with a <see cref="GenerationProgress"/>
	/// snapshot, for progress reporting.
	/// </param>
	/// <param name="targetFitness">
	/// Optional early-stop threshold. When the best fitness reaches or exceeds this value,
	/// evolution stops immediately and returns. Use it to advance a curriculum stage as
	/// soon as the current objective is solved, rather than always running every generation.
	/// </param>
	/// <param name="onGenerationStart">
	/// Optional hook invoked at the start of each generation, before any evaluation. Use it for
	/// a <i>dynamic</i> (non-stationary) fitness landscape — e.g. resampling the test cases each
	/// generation so the population can never settle on a fixed target. When supplied, the parent
	/// is re-evaluated each generation after this runs, so parent and offspring are always
	/// compared on the same freshly-changed conditions (without this, a parent that drew a lucky
	/// easy sample would keep a stale high score that nothing could beat).
	/// </param>
	/// <param name="confirm">
	/// Optional two-tier early-stop oracle. When <paramref name="targetFitness"/> is reached on
	/// the (cheap, possibly sampled) selection fitness, this expensive check is consulted and the
	/// run stops only if it also returns true. This is the minibatch/validation pattern: the cheap
	/// signal gates the loop every generation, the expensive oracle confirms the decision rarely.
	/// When null, crossing <paramref name="targetFitness"/> stops the run directly.
	/// </param>
	/// <returns>
	/// The best genome found, its fitness, and the number of generations actually run
	/// (which is less than <paramref name="generations"/> when the target is reached early).
	/// </returns>
	public EvolutionResult Evolve(
		NandGenome initial,
		int generations,
		Action<GenerationProgress>? onGeneration = null,
		double? targetFitness = null,
		Action? onGenerationStart = null,
		Func<NandGenome, bool>? confirm = null)
	{
		ArgumentNullException.ThrowIfNull(initial);
		if (generations < 0)
			throw new ArgumentOutOfRangeException(nameof(generations), generations, "Generations cannot be negative.");

		var parent = initial;
		var parentFitness = _fitness(parent);

		// An initial genome may already meet the target (e.g. a later curriculum stage
		// starting from a strong parent); honor that before running any generations.
		// Skip this shortcut under a dynamic landscape, where a single high score is not
		// trustworthy evidence the objective is truly met.
		if (onGenerationStart is null && targetFitness.HasValue && parentFitness >= targetFitness.Value
			&& (confirm is null || confirm(parent)))
			return new EvolutionResult(parent, parentFitness, 0);

		// CUSUM stagnation trigger: smooths the (possibly noisy) fitness and accumulates how far its
		// velocity falls short of a real-climb rate. Symmetric resampling jitter cancels in the sum
		// rather than resetting a counter, and — crucially — the sum self-drains when progress
		// resumes, so the ramp can never latch on the way a level high-water mark did. Only set up
		// when that trigger is configured.
		var stallTrigger = _adaptiveMutation?.Stall;
		var tracker = stallTrigger?.CreateTracker();
		var stallSum = 0.0;

		var generationsRun = 0;
		var generationsSinceImprovement = 0;
		var stallRampDuration = 0;
		for (var generation = 0; generation < generations; generation++)
		{
			// Dynamic fitness: let the caller change the landscape (e.g. resample test cases),
			// then re-score the parent so it is judged on the same new conditions as its
			// offspring this generation. This is what keeps the population "off balance".
			if (onGenerationStart is not null)
			{
				onGenerationStart();
				parentFitness = _fitness(parent);
			}

			// Stagnation duration drives the mutation ramp. The fixed trigger counts consecutive
			// non-improving generations; the CUSUM trigger counts how long it has been armed (its
			// accumulated velocity shortfall has stayed over threshold), which the sawtooth uses as
			// its ramp position. The duration drops back to zero the moment the CUSUM drains.
			var stagnationDuration = stallTrigger is not null ? stallRampDuration : generationsSinceImprovement;

			// While stuck, adaptive mutation ramps the rate to push offspring out of the
			// parent's neighborhood; it stays at the base rate when not configured.
			var rate = _adaptiveMutation?.EffectiveRate(_mutationRate, stagnationDuration)
				?? _mutationRate;

			// Produce λ offspring and find the best of them (parallelized when enabled).
			var (bestChild, bestChildFitness) = ProduceBestOffspring(parent, rate);

			// Promote the best offspring when it ties or beats the parent. The tie case is
			// the neutral-drift rule: equal fitness still moves the search forward.
			if (bestChildFitness >= parentFitness)
			{
				// A strict improvement resets the stagnation ramp; an equal-fitness (neutral)
				// move does not, since flat fitness is exactly the stall we are escaping.
				if (bestChildFitness > parentFitness)
					generationsSinceImprovement = 0;
				else
					generationsSinceImprovement++;

				parent = bestChild;
				parentFitness = bestChildFitness;
			}
			else
			{
				generationsSinceImprovement++;
			}

			// Feed the smoothed tracker, advance the CUSUM, and grow the ramp duration while the sum
			// stays over threshold. When progress resumes the sum drains back below threshold and the
			// duration resets to zero, releasing the ramp — the self-draining behavior that keeps the
			// search from latching in a permanent stall.
			if (stallTrigger is not null)
			{
				tracker!.Update(parentFitness);
				stallSum = stallTrigger.Accumulate(stallSum, tracker.Velocity);
				stallRampDuration = stallTrigger.IsStalling(stallSum) ? stallRampDuration + 1 : 0;
			}

			generationsRun = generation + 1;
			onGeneration?.Invoke(new GenerationProgress(
				generation, parentFitness, parent, rate,
				tracker?.Level, tracker?.Velocity,
				stallTrigger is not null ? stallSum : null));

			// Early stop once the objective is met — gated by the expensive oracle when present.
			if (targetFitness.HasValue && parentFitness >= targetFitness.Value
				&& (confirm is null || confirm(parent)))
				break;
		}

		return new EvolutionResult(parent, parentFitness, generationsRun);
	}

	/// <summary>
	/// Produces the λ offspring of <paramref name="parent"/> at the given mutation rate and
	/// returns the fittest, breaking ties toward the lowest offspring index. Each offspring is
	/// mutated from its own pre-drawn seed, so the result is identical whether evaluation runs
	/// serially or in parallel — only the wall-clock cost differs.
	/// </summary>
	private (NandGenome Genome, double Fitness) ProduceBestOffspring(NandGenome parent, double rate)
	{
		// Draw one seed per offspring from the shared stream. Doing this sequentially keeps the
		// seed sequence (and thus the whole run) reproducible regardless of parallelism.
		var seeds = new int[_lambda];
		for (var i = 0; i < _lambda; i++)
			seeds[i] = _random.Next();

		var children = new NandGenome[_lambda];
		var fitnesses = new double[_lambda];

		if (_parallel && _lambda > 1)
		{
			Parallel.For(0, _lambda, i =>
			{
				var child = parent.Mutate(rate, new Random(seeds[i]));
				children[i] = child;
				fitnesses[i] = _fitness(child);
			});
		}
		else
		{
			for (var i = 0; i < _lambda; i++)
			{
				var child = parent.Mutate(rate, new Random(seeds[i]));
				children[i] = child;
				fitnesses[i] = _fitness(child);
			}
		}

		// Reduce to the best, preferring the lowest index on ties (matches serial behavior).
		var bestIndex = 0;
		for (var i = 1; i < _lambda; i++)
		{
			if (fitnesses[i] > fitnesses[bestIndex])
				bestIndex = i;
		}

		return (children[bestIndex], fitnesses[bestIndex]);
	}

	#endregion
}
