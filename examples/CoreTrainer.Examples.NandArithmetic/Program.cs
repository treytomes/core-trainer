using CoreTrainer.Networking.Nand;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using NetEscapades.Configuration.Yaml;
using Serilog;
using Serilog.Extensions.Logging;
// Disambiguate ILogger (both MS and Serilog define one); the example logs via the MS abstraction.
using ILogger = Microsoft.Extensions.Logging.ILogger;

// ---------------------------------------------------------------------------------------
// NAND Arithmetic — a standalone, isolated demonstration of the CGP training system.
//
// This example evolves a network of NAND gates to perform 8-bit addition and subtraction,
// with no Core Wars involved. It exists to show, in the simplest possible setting, how the
// pieces fit together:
//
//   NandGenome    — the evolvable description of a NAND-gate circuit
//   NandNetwork   — decodes a genome and evaluates it for a set of input bits
//   CgpEvolution  — the 1+lambda evolution strategy that searches for a good genome
//
// The only thing specific to "arithmetic" here is the FITNESS FUNCTION. Swap that out and
// the very same evolution machinery learns something else — which is exactly how the Core
// Wars NAND warriors work (their fitness is "win battles" instead of "add correctly").
// ---------------------------------------------------------------------------------------

// === Configuration ===================================================================
// Try lowering BitWidth to 4 to watch the circuit approach 100% accuracy quickly; 8-bit
// add/sub is a genuinely hard structure to discover from random wiring, so expect steady
// improvement rather than perfection within a short run.

/// <summary>
/// How the network is trained.
/// </summary>
public enum TrainingStrategy
{
	/// <summary>
	/// Score all output bits at once for the whole run. Simple, but the carry chain makes
	/// this a deceptive landscape — accuracy climbs the low bits fast, then plateaus.
	/// </summary>
	Flat,

	/// <summary>
	/// A curriculum: train output bit 0 to a threshold, then bits 0-1, then 0-2, and so on,
	/// expanding the scored bits one at a time on a single continuous genome. This matches
	/// the carry-chain dependency order (bit k needs the carry from the bits below it), so
	/// each stage adds a climbable step instead of asking for the whole chain at once.
	/// Earlier bits keep being scored each stage, so selection preserves them — a "soft
	/// freeze" through the objective rather than locking part of the circuit.
	/// </summary>
	StagedCurriculum
}

/// <summary>
/// How candidate circuits are scored.
/// </summary>
public enum FitnessMode
{
	/// <summary>
	/// Score against a fixed random sample of cases. Fast, but "100%" means only "correct
	/// on the sample" — a large network could memorize the sample and still be wrong
	/// elsewhere (the generalization gap familiar from neural networks).
	/// </summary>
	Sampled,

	/// <summary>
	/// Score against every possible input combination. Slower per generation, but "100%"
	/// is a proof of a fully correct circuit — there is no unseen input to be wrong on.
	/// </summary>
	Exhaustive
}

public static class Program
{
	#region Constants

	/// <summary>
	/// Swap this to compare flat training vs. the staged curriculum (see <see cref="TrainingStrategy"/>).
	/// </summary>
	private const TrainingStrategy TRAINING_STRATEGY = TrainingStrategy.StagedCurriculum;

	/// <summary>
	/// Swap this to compare sampled vs. exhaustive scoring (see <see cref="FitnessMode"/>).
	/// </summary>
	private const FitnessMode FITNESS_MODE = FitnessMode.Sampled;

	private const int BIT_WIDTH = 4;

	/// <summary>
	/// Accuracy a curriculum stage must reach on its scored bits before advancing. A viable
	/// circuit needs every scored bit correct on every case, so this is a true 100%. Because
	/// accuracy is an exact rational (correct bits / total bits), 1.0 is exactly reachable.
	/// Each stage runs with no generation cap until it hits this target — so a run can take a
	/// long time, and a stage that cannot reach 100% will not advance.
	/// </summary>
	private const double STAGE_TARGET = 1.0;

	/// <summary>
	/// NAND gates available to wire together
	/// </summary>
	private const int NODE_COUNT = 600;
	private const int GENERATIONS = 30000;

	/// <summary>
	/// Offspring per generation. Raised above the CGP-classic 4 because the carry chain needs
	/// a coordinated multi-gene change to cross each plateau; more offspring per generation is
	/// more independent lottery tickets at that rare change, at a modest mutation rate — which
	/// beats one ticket at a destructive high rate.
	/// </summary>
	private const int LAMBDA = 16;
	private const double MUTATION_RATE = 0.03;

	/// <summary>
	/// Adaptive mutation uses a CUSUM trigger on the smoothed fitness velocity. The raw fitness is
	/// noisy under dynamic sampling (it bobbles as the test set shifts), so a per-generation
	/// improvement counter is fooled into thinking progress is happening, and the sign of the
	/// smoothed velocity bobs around zero on a plateau (defeating a consecutive-stall counter). A
	/// level high-water mark detects the plateau but latches the ramp on if the level ever regresses.
	/// The CUSUM accumulates how far velocity falls short of <see cref="MIN_VELOCITY"/> each
	/// generation: symmetric noise cancels in the sum, it arms once the sum crosses
	/// <see cref="STALL_THRESHOLD"/>, and it self-drains (releasing the ramp) the moment real
	/// progress resumes — so it cannot stay stuck in a stalled state.
	/// </summary>
	private const double MIN_VELOCITY = 0.0002;

	/// <summary>
	/// Accumulated velocity shortfall (CUSUM limit) at which the search is declared stalled. Roughly
	/// <c>STALL_THRESHOLD / MIN_VELOCITY</c> stalled generations to engage — here ~300.
	/// </summary>
	private const double STALL_THRESHOLD = 0.06;

	/// <summary>
	/// EMA smoothing for the fitness level (smaller = smoother, slower to react).
	/// </summary>
	private const double LEVEL_SMOOTHING = 0.05;

	/// <summary>
	/// EMA smoothing for the velocity of the level (the signal the CUSUM integrates).
	/// </summary>
	private const double VELOCITY_SMOOTHING = 0.05;

	/// <summary>
	/// How much the mutation rate grows per stagnant generation once the ramp engages.
	/// </summary>
	private const double MUTATION_GROWTH = 0.0005;

	/// <summary>
	/// Ceiling for the scaled mutation rate. Kept low: at high rates almost every gene is
	/// randomized, so offspring are near-restarts that can never beat an established parent
	/// under elitist selection — the rate then pins at max and fitness freezes. A modest cap
	/// perturbs enough to cross a valley while keeping the working circuit mostly intact.
	/// </summary>
	private const double MUTATION_MAX = 0.08;

	/// <summary>
	/// When true (and FitnessMode is Sampled), the test cases are redrawn every generation so
	/// the network faces a constantly shifting target — a "Red Queen" pressure that prevents
	/// settling on a good-enough circuit that only happens to beat one fixed sample. Inspired
	/// by Hillis's co-evolved sorting networks. Has no effect under Exhaustive scoring, where
	/// every case is already tested each generation.
	/// </summary>
	private const bool DYNAMIC_SAMPLING = true;

	/// <summary>
	/// When true (Sampled mode only), part of each generation's sample is reserved for cases the
	/// current-best circuit gets WRONG on the exhaustive check — mining its own failures and folding
	/// them back into the sample so selection is pulled directly toward fixing them. This is the
	/// Hillis "parasite" idea taken to its conclusion: rather than a purely random shifting target,
	/// the target actively hunts the circuit's current weaknesses. The remaining fraction stays
	/// fresh-random to preserve general coverage and the Red Queen pressure. Costs one exhaustive
	/// evaluation of the best genome per generation.
	/// </summary>
	private const bool HARD_CASE_MINING = true;

	/// <summary>
	/// The first curriculum stage (by scored-bit count) at which hard-case mining engages; earlier
	/// stages use plain dynamic sampling. Measured rationale: the low-bit stages already converge
	/// fast on their own and have no stubborn failure set worth mining, so mining only taxes them —
	/// stage 1 went 13→27 gens and stage 2 went 4,580→12,000+ gens once mining was on. Mining was
	/// built for the carry-chain plateau that appears from the middle bits onward (stage 3+), where
	/// nothing else cracked the stall; gating it there isolates the one place it might earn its keep.
	/// Set to 1 to mine every stage, or above <c>outputCount</c> to disable mining via the gate.
	/// </summary>
	private const int MINING_FROM_STAGE = 99;

	/// <summary>
	/// How many mined hard cases to APPEND to each sample, as a fraction of <see cref="TEST_CASES"/>
	/// (e.g. 0.2 adds up to ~51 failing cases on top of the full 256-case random sample). The hard
	/// cases are added, not substituted: the complete fresh-random sample is preserved so coverage of
	/// the cases the circuit already gets right is never reduced — earlier, replacing part of the
	/// sample with hard cases drove catastrophic forgetting (at 0.5 even trivial stage 1 blew up from
	/// 13 to ~1000 generations). Kept modest so the failures bias selection without swamping the
	/// random coverage. Capped by how many failing cases actually exist, so the appended portion
	/// shrinks toward the few remaining failures as the circuit approaches correctness.
	/// </summary>
	private const double HARD_CASE_FRACTION = 0.2;

	/// <summary>
	/// Sampled (A, B, op) cases used to score a genome.
	/// </summary>
	private const int TEST_CASES = 256;

	/// <summary>
	/// Generations between progress lines.
	/// </summary>
	private const int PROGRESS_EVERY = 100;

	/// <summary>
	/// Generations between exhaustive-accuracy lines. The per-generation <c>bit-acc</c> is measured
	/// on the (possibly mining-augmented) sample, which is deliberately stacked with cases the circuit
	/// fails — so it understates true progress. The exhaustive accuracy over every input combination
	/// is the truthful signal, but expensive, so it is logged only this often (10x the progress
	/// cadence) to keep its cost amortized.
	/// </summary>
	private const int EXHAUSTIVE_EVERY = 1000;

	/// <summary>
	/// Fixed so the test set (and fitness) is stable.
	/// </summary>
	private const int TEST_SEED = 12345;

	/// <summary>
	/// Fixed so runs are reproducible.
	/// </summary>
	private const int EVOLUTION_SEED = 1;

	#endregion

	#region Methods

	public static void Main(params string[] args)
	{
		// Configure logging the same way CoreTrainer.Console does: Serilog driven by the
		// Serilog section of appsettings.yaml, bridged to Microsoft.Extensions.Logging so
		// output and the rolling log file match the main app's format.
		using var loggerFactory = CreateLoggerFactory();
		var log = loggerFactory.CreateLogger("NandArithmetic");

		// Input layout:  [A bits (BitWidth)] [B bits (BitWidth)] [opSelect (1)]
		//   opSelect = false -> A + B,  true -> A - B
		//
		// Output layout: the sum bits and the INTERNAL ripple carries, INTERLEAVED in dependency
		// order so the curriculum learns each carry right before the sum bit that consumes it:
		//
		//   index:  0      1     2      3     4      5     6      7     8
		//   bit:    sum0   c1    sum1   c2    sum2   c3    sum3   c4    sum4
		//
		// where c_k is the carry OUT of bit position k-1 (the carry INTO bit k). Exposing the
		// carries as scored outputs turns the hidden shared structure of the carry chain — the thing
		// that walls the sum-only curriculum at the middle bits — into an explicit reward at each
		// step. sum4 (the MSB) is the addition overflow / subtraction sign bit. Total outputs:
		// (BitWidth+1) sum bits + BitWidth carries = 2*BitWidth + 1.
		var inputCount = 2 * BIT_WIDTH + 1;
		var outputCount = 2 * BIT_WIDTH + 1;

		// Mask covering just the (BitWidth+1) result bits, for decoding the human-readable A op B
		// value from the sum outputs in the report.
		var resultMask = (1 << (BIT_WIDTH + 1)) - 1;

		// Build the set of (A, B, op) cases the fitness function scores against. The only
		// difference between sampled and exhaustive fitness is WHICH cases this returns —
		// the scoring logic below is identical either way.
		var cases = BuildCases(FITNESS_MODE, BIT_WIDTH);
		var scoringDescription = FITNESS_MODE == FitnessMode.Exhaustive
			? $"bit accuracy over ALL {cases.Length} input combinations (exhaustive)"
			: $"bit accuracy over {cases.Length} sampled cases";

		var searchBudget = TRAINING_STRATEGY == TrainingStrategy.StagedCurriculum
			? "each stage runs to 100% (uncapped)"
			: $"{GENERATIONS} generations";

		log.LogInformation("=== NAND Arithmetic Trainer ===");
		log.LogInformation("Goal:     evolve a {BitWidth}-bit adder/subtractor from NAND gates", BIT_WIDTH);
		log.LogInformation("Network:  {Inputs} inputs, {Nodes} nodes, {Outputs} outputs ({Sums} sum + {Carries} carry, interleaved)",
			inputCount, NODE_COUNT, outputCount, BIT_WIDTH + 1, BIT_WIDTH);
		log.LogInformation("Search:   1+{Lambda}, {Budget}, mutation {Rate:P1}", LAMBDA, searchBudget, MUTATION_RATE);
		log.LogInformation("Scoring:  {Scoring}", scoringDescription);
		log.LogInformation("Strategy: {Strategy}", TRAINING_STRATEGY);

		// Accuracy over the lowest `scoredBits` output bits: the fraction of those bits the
		// network gets right across all cases. Scoring a subset is what lets the curriculum
		// focus on one bit at a time; scoring all bits (scoredBits == outputCount) is the
		// flat strategy. Bit-level scoring gives a smooth gradient — a circuit that gets most
		// scored bits right outscores one that doesn't, even before any answer is complete.
		double Accuracy(NandGenome genome, int scoredBits)
		{
			var network = new NandNetwork(genome);
			var inputs = new bool[inputCount];
			var correctBits = 0;

			foreach (var (a, b, subtract) in cases)
			{
				EncodeInputs(inputs, a, b, subtract);
				var outputs = network.Evaluate(inputs);

				var expected = Expected(a, b, subtract);
				for (var bit = 0; bit < scoredBits; bit++)
				{
					var expectedBit = ((expected >> bit) & 1) == 1;
					if (outputs[bit] == expectedBit)
						correctBits++;
				}
			}

			return (double)correctBits / (cases.Length * scoredBits);
		}

		// The exhaustive (truthful) accuracy over EVERY input combination, independent of the
		// shifting sample. This is the expensive oracle: used to confirm stage advancement and
		// to report the real final accuracy — never in the per-generation selection loop.
		var allCases = BuildExhaustiveCases(BIT_WIDTH);
		double ExhaustiveAccuracy(NandGenome genome, int scoredBits)
		{
			var network = new NandNetwork(genome);
			var inputs = new bool[inputCount];
			var correctBits = 0;

			foreach (var (a, b, subtract) in allCases)
			{
				EncodeInputs(inputs, a, b, subtract);
				var outputs = network.Evaluate(inputs);

				var expected = Expected(a, b, subtract);
				for (var bit = 0; bit < scoredBits; bit++)
				{
					var expectedBit = ((expected >> bit) & 1) == 1;
					if (outputs[bit] == expectedBit)
						correctBits++;
				}
			}

			return (double)correctBits / (allCases.Length * scoredBits);
		}

		// Returns the cases the given network gets WRONG on the lowest `scoredBits` bits — the
		// circuit's current failures, harvested from the full input space. Hard-case mining folds a
		// subset of these into the next sample so selection is pulled straight at the weaknesses.
		List<(int A, int B, bool Subtract)> FindFailures(NandNetwork network, int scoredBits)
		{
			var failures = new List<(int, int, bool)>();
			var inputs = new bool[inputCount];
			foreach (var (a, b, subtract) in allCases)
			{
				EncodeInputs(inputs, a, b, subtract);
				var outputs = network.Evaluate(inputs);
				var expected = Expected(a, b, subtract);
				for (var bit = 0; bit < scoredBits; bit++)
				{
					if (outputs[bit] != (((expected >> bit) & 1) == 1))
					{
						failures.Add((a, b, subtract));
						break; // one wrong bit is enough to call the whole case a failure
					}
				}
			}
			return failures;
		}

		// The per-generation resampler (Sampled mode only). It rebuilds `cases` from a single
		// advancing RNG so the target keeps shifting (Red Queen pressure) — and, when hard-case
		// mining is on, reserves part of the sample for cases the current-best circuit actually
		// fails on, hunting its weaknesses instead of sampling blind. Null when neither dynamic
		// sampling nor mining applies, leaving the landscape stationary.
		//
		// Takes the stage's scored-bit count and the current-best genome (mining needs both to know
		// what "wrong" means right now). Costs one exhaustive evaluation of the best genome per
		// generation when mining — the price of aiming the sample.
		var range = 1 << BIT_WIDTH;
		var samplingSource = new Random(TEST_SEED);
		var useDynamic = (DYNAMIC_SAMPLING || HARD_CASE_MINING) && FITNESS_MODE == FitnessMode.Sampled;

		Action<int, NandGenome>? resample = null;
		if (useDynamic && HARD_CASE_MINING)
		{
			// Mine the current failures, then fill the sample: a capped hard portion drawn from those
			// failures, the remainder fresh-random. As the circuit improves there are fewer failures,
			// so the hard portion shrinks toward the last stubborn cases on its own.
			resample = (scoredBits, best) =>
			{
				// Mining engages only from MINING_FROM_STAGE onward (scoredBits == the stage number).
				// Earlier stages converge fine on plain dynamic sampling, where mining only taxes them.
				if (scoredBits < MINING_FROM_STAGE)
				{
					cases = Sample(range, samplingSource);
					return;
				}

				var failures = FindFailures(new NandNetwork(best), scoredBits);
				var hardCount = Math.Min((int)(TEST_CASES * HARD_CASE_FRACTION), failures.Count);
				var fresh = Sample(range, samplingSource);

				// Draw the hard cases WITHOUT replacement (a partial Fisher–Yates shuffle takes
				// `hardCount` distinct failures). Drawing with replacement would stamp the same few
				// cases into many slots once only a handful fail — massively over-weighting a tiny set
				// and amplifying the forgetting that mining already risks.
				for (var i = 0; i < hardCount; i++)
				{
					var j = i + samplingSource.Next(failures.Count - i);
					(failures[i], failures[j]) = (failures[j], failures[i]);
				}

				// APPEND the hard cases on top of the full fresh sample rather than replacing part of
				// it. Replacing shrank the random portion, sacrificing coverage of the cases the circuit
				// already gets right — which is itself a driver of catastrophic forgetting. Appending
				// keeps the complete random sample (so "what it already handles" is never under-sampled)
				// and adds the failures as pure extra pressure. Accuracy normalizes by cases.Length, so
				// a variable-length sample scores correctly.
				var mixed = new (int A, int B, bool Subtract)[TEST_CASES + hardCount];
				Array.Copy(fresh, 0, mixed, 0, TEST_CASES);
				for (var i = 0; i < hardCount; i++)
					mixed[TEST_CASES + i] = failures[i];
				cases = mixed;
			};
		}
		else if (useDynamic)
		{
			// Plain dynamic sampling: a fresh random draw each generation, no mining.
			resample = (_, _) => cases = Sample(range, samplingSource);
		}

		// Two-tier gate (Sampled mode only): selection runs on the cheap shifting sample, but a
		// stage advances only when the exhaustive oracle confirms its bits are perfect on every
		// input. Under Exhaustive selection the sample already IS the truth, so no separate
		// confirmation is needed (null gate = stop directly on target).
		var useTwoTier = FITNESS_MODE == FitnessMode.Sampled;

		var random = new Random(EVOLUTION_SEED);
		var initial = NandGenome.CreateRandom(inputCount, NODE_COUNT, outputCount, random);

		var best = TRAINING_STRATEGY == TrainingStrategy.StagedCurriculum
			? RunStaged(log, initial, random, Accuracy, ExhaustiveAccuracy, outputCount, resample, useTwoTier)
			: RunFlat(log, initial, random, Accuracy, ExhaustiveAccuracy, outputCount, resample);

		// === Report =========================================================================
		// Report both signals: the exhaustive truth (what the circuit really does) and, in
		// Sampled mode, the noisy sample the search actually optimized against.
		var exhaustiveFinal = ExhaustiveAccuracy(best, outputCount);
		log.LogInformation("Final accuracy (exhaustive, all {Outputs} bits): {Accuracy:P2}  (active gates: {Active}/{Total})",
			outputCount, exhaustiveFinal, new NandNetwork(best).ActiveNodeCount, NODE_COUNT);
		if (FITNESS_MODE == FitnessMode.Sampled)
		{
			log.LogInformation("Final accuracy (current sample): {Accuracy:P2}", Accuracy(best, outputCount));
		}

		// Show a handful of concrete evaluations so the result is tangible.
		log.LogInformation("Sample evaluations (after training):");
		var bestNetwork = new NandNetwork(best);
		var sampleInputs = new bool[inputCount];
		for (var i = 0; i < 8; i++)
		{
			var (a, b, subtract) = cases[i];
			EncodeInputs(sampleInputs, a, b, subtract);
			var got = DecodeOutputs(bestNetwork.Evaluate(sampleInputs));
			// Human-readable result value (just the sum bits), to compare against the decoded sum.
			var expected = (subtract ? a - b : a + b) & resultMask;
			var op = subtract ? "-" : "+";
			var mark = got == expected ? "OK " : "xx ";
			log.LogInformation("  {Mark} {A,3} {Op} {B,3}  expected {Expected,4}  got {Got,4}", mark, a, op, b, expected, got);
		}
	}

	/// <summary>
	/// Builds an <see cref="ILoggerFactory"/> backed by Serilog, configured from the Serilog
	/// section of appsettings.yaml — the same logging stack and format CoreTrainer.Console uses.
	/// </summary>
	private static ILoggerFactory CreateLoggerFactory()
	{
		var configuration = new ConfigurationBuilder()
			.SetBasePath(AppContext.BaseDirectory)
			.AddYamlFile("appsettings.yaml", optional: false, reloadOnChange: false)
			.Build();

		var serilogLogger = new Serilog.LoggerConfiguration()
			.ReadFrom.Configuration(configuration)
			.CreateLogger();

		// dispose: true lets the factory flush/close Serilog (and its file sink) on dispose.
		return new SerilogLoggerFactory(serilogLogger, dispose: true);
	}

	/// <summary>
	/// Flat strategy: one run scoring all output bits for the whole generation budget.
	/// </summary>
	private static NandGenome RunFlat(
		ILogger log,
		NandGenome initial,
		Random random,
		Func<NandGenome, int, double> accuracy,
		Func<NandGenome, int, double> exhaustiveAccuracy,
		int outputCount,
		Action<int, NandGenome>? resample)
	{
		// Flat fitness scores every output bit at once. Accuracy is thread-safe (each call
		// builds its own network and buffers over read-only cases), so offspring evaluate in parallel.
		var evolution = new CgpEvolution(g => accuracy(g, outputCount), LAMBDA, MUTATION_RATE, random, BuildAdaptiveMutation(), parallel: true);

		log.LogInformation("Starting accuracy: {Accuracy:P2}", accuracy(initial, outputCount));

		// Peak mutation rate since the last log line — see RunStaged for why the instantaneous rate
		// aliases against the log cadence and a running peak is reported instead.
		var peakMut = 0.0;

		// The resampler (when mining) needs the current-best genome to harvest its failures. The
		// per-generation hook runs before evaluation, so it sees the previous generation's parent —
		// which is exactly the current best. Seed it with the starting genome and keep it current.
		var currentBest = initial;
		Action? onGenerationStart = resample is null ? null : () => resample(outputCount, currentBest);

		var result = evolution.Evolve(initial, GENERATIONS, onGeneration: progress =>
		{
			currentBest = progress.Best;
			peakMut = Math.Max(peakMut, progress.MutationRate);
			if (progress.Generation == 0 || (progress.Generation + 1) % PROGRESS_EVERY == 0)
			{
				var activeGates = new NandNetwork(progress.Best).ActiveNodeCount;
				log.LogInformation(
					"  gen {Gen,5}/{Total}   accuracy {Accuracy:P2}   smooth {Smooth:P2}   vel {Vel:+0.0000;-0.0000; 0.0000}   cusum {Cusum:0.0000}/{Threshold:0.0000}   mut peak {Mut:P2}   active gates {Active,4}/{Nodes}",
					progress.Generation + 1, GENERATIONS, progress.Fitness, progress.SmoothedFitness, progress.FitnessVelocity, progress.StallSum, STALL_THRESHOLD, peakMut, activeGates, NODE_COUNT);
				peakMut = 0.0;
			}

			// Exhaustive accuracy is the truthful signal (the accuracy above is on the mining-
			// augmented sample, which is stacked with known failures and so understates progress).
			// It is expensive, so log it only every EXHAUSTIVE_EVERY generations.
			if (progress.Generation == 0 || (progress.Generation + 1) % EXHAUSTIVE_EVERY == 0)
			{
				log.LogInformation(
					"  gen {Gen,5}/{Total}   exhaustive {Exhaustive:P2}",
					progress.Generation + 1, GENERATIONS, exhaustiveAccuracy(progress.Best, outputCount));
			}
		}, onGenerationStart: onGenerationStart);

		return result.Best;
	}

	/// <summary>
	/// Staged curriculum: train output bit 0 to <see cref="STAGE_TARGET"/>, then bits 0-1,
	/// then 0-2, and so on — expanding the scored bits one at a time on a single genome that
	/// carries forward between stages. Each stage gets its own evolution loop with a fitness
	/// function that scores exactly the bits unlocked so far, and stops early once the stage
	/// target is met. Because every stage keeps scoring all previously-won bits, selection
	/// preserves them ("soft freeze") even as new bits are learned.
	/// </summary>
	private static NandGenome RunStaged(
		ILogger log,
		NandGenome initial,
		Random random,
		Func<NandGenome, int, double> accuracy,
		Func<NandGenome, int, double> exhaustiveAccuracy,
		int outputCount,
		Action<int, NandGenome>? resample,
		bool useTwoTier)
	{
		var genome = initial;

		for (var scoredBits = 1; scoredBits <= outputCount; scoredBits++)
		{
			// This stage's fitness scores only the lowest `scoredBits` output bits. Accuracy is
			// thread-safe, so the λ offspring evaluate in parallel.
			var bits = scoredBits;
			var evolution = new CgpEvolution(g => accuracy(g, bits), LAMBDA, MUTATION_RATE, random, BuildAdaptiveMutation(), parallel: true);

			log.LogInformation("-- Stage {Stage}/{Total}: scoring bits 0..{HighBit} (start {Start:P2}) --",
				scoredBits, outputCount, scoredBits - 1, accuracy(genome, bits));

			// Two-tier gate: when selection runs on the cheap shifting sample, a sampled 100%
			// only proposes advancement — the exhaustive oracle must confirm the stage's bits are
			// perfect on every input before we move on.
			Func<NandGenome, bool>? confirm = useTwoTier
				? g => exhaustiveAccuracy(g, bits) >= 1.0
				: null;

			// Track the peak mutation rate since the last log line. The sawtooth ramp can cycle many
			// times between two logged generations, and sampling it only at the logged generation
			// aliases (if the log cadence and the sawtooth period share a factor, every sample lands
			// on the same phase — making a live ramp look frozen at the base rate). The running peak
			// shows the ramp's true reach regardless of where the tooth happens to sit when we log.
			var peakMut = 0.0;

			// The resampler (when mining) harvests the current-best genome's failures on THIS stage's
			// bits. Bridge the no-arg per-generation hook to it by tracking the current best; seed it
			// with the genome this stage starts from. Reset per stage so mining always targets the
			// bits in play now.
			var currentBest = genome;
			Action? onGenerationStart = resample is null ? null : () => resample(bits, currentBest);

			// No generation cap: the stage runs until it reaches (and confirms) a perfect score.
			var result = evolution.Evolve(
				genome,
				generations: int.MaxValue,
				onGeneration: progress =>
				{
					currentBest = progress.Best;
					peakMut = Math.Max(peakMut, progress.MutationRate);
					if (progress.Generation == 0 || (progress.Generation + 1) % PROGRESS_EVERY == 0)
					{
						var activeGates = new NandNetwork(progress.Best).ActiveNodeCount;
						log.LogInformation(
							"  stage {Stage}/{Stages}   gen {Gen,7}   bit-acc {Accuracy:P2}   smooth {Smooth:P2}   vel {Vel:+0.0000;-0.0000; 0.0000}   cusum {Cusum:0.0000}/{Threshold:0.0000}   mut peak {Mut:P2}   active gates {Active,4}/{Nodes}",
							bits, outputCount, progress.Generation + 1, progress.Fitness, progress.SmoothedFitness, progress.FitnessVelocity, progress.StallSum, STALL_THRESHOLD, peakMut, activeGates, NODE_COUNT);
						peakMut = 0.0;
					}

					// Exhaustive accuracy is the truthful signal (the bit-acc above is on the mining-
					// augmented sample, which is stacked with known failures and so understates progress).
					// It is expensive, so log it only every EXHAUSTIVE_EVERY generations.
					if (progress.Generation == 0 || (progress.Generation + 1) % EXHAUSTIVE_EVERY == 0)
					{
						log.LogInformation(
							"  stage {Stage}/{Stages}   gen {Gen,7}   exhaustive {Exhaustive:P2}",
							bits, outputCount, progress.Generation + 1, exhaustiveAccuracy(progress.Best, bits));
					}
				},
				targetFitness: STAGE_TARGET,
				onGenerationStart: onGenerationStart,
				confirm: confirm);

			genome = result.Best;
			log.LogInformation("   stage solved after {Gens} gens, sample bit-acc {Sample:P2}, exhaustive {Exhaustive:P2}",
				result.Generations, result.Fitness, exhaustiveAccuracy(genome, bits));
		}

		return genome;
	}

	/// <summary>
	/// Builds the stagnation-triggered mutation scaling shared by both strategies: a CUSUM velocity
	/// trigger detects when the fitness has stopped climbing (surviving the noise of dynamic
	/// sampling), the rate then sawtooth-ramps to escape the plateau, and it snaps back to base —
	/// because the CUSUM self-drains — once real progress resumes. Patience is 0 here: the CUSUM
	/// threshold already governs how long a stall must persist before the ramp engages.
	/// </summary>
	private static AdaptiveMutation BuildAdaptiveMutation() =>
		new(
			Patience: 0,
			MUTATION_GROWTH,
			MUTATION_MAX,
			Sawtooth: true,
			Stall: new StallTrigger(LEVEL_SMOOTHING, VELOCITY_SMOOTHING, MIN_VELOCITY, STALL_THRESHOLD));

	/// <summary>
	/// Builds the (A, B, op) cases the fitness function scores against.
	/// <list type="bullet">
	/// <item><see cref="FitnessMode.Sampled"/> — a fixed random subset, scored quickly.</item>
	/// <item><see cref="FitnessMode.Exhaustive"/> — every combination of A, B, and op, so a
	/// perfect score proves a correct circuit.</item>
	/// </list>
	/// </summary>
	private static (int A, int B, bool Subtract)[] BuildCases(FitnessMode mode, int bitWidth)
	{
		if (mode == FitnessMode.Exhaustive)
			return BuildExhaustiveCases(bitWidth);

		// Sampled: the fixed baseline subset, drawn from a fresh seeded RNG. Dynamic resampling
		// (when enabled) replaces this each generation via Sample() with an advancing source.
		return Sample(1 << bitWidth, new Random(TEST_SEED));
	}

	/// <summary>
	/// Builds every (A, B, op) combination — the full input space, used as the truthful oracle
	/// for confirmation and final reporting.
	/// </summary>
	private static (int A, int B, bool Subtract)[] BuildExhaustiveCases(int bitWidth)
	{
		var range = 1 << bitWidth;
		var all = new (int, int, bool)[range * range * 2];
		var index = 0;
		for (var a = 0; a < range; a++)
		{
			for (var b = 0; b < range; b++)
			{
				all[index++] = (a, b, false);
				all[index++] = (a, b, true);
			}
		}
		return all;
	}

	/// <summary>
	/// Draws <see cref="TEST_CASES"/> random (A, B, op) cases from the given source. Reusing one
	/// advancing <paramref name="source"/> across calls yields a different assortment each time.
	/// </summary>
	private static (int A, int B, bool Subtract)[] Sample(int range, Random source)
	{
		var sampled = new (int, int, bool)[TEST_CASES];
		for (var i = 0; i < TEST_CASES; i++)
		{
			var a = source.Next(range);
			var b = source.Next(range);
			var subtract = source.Next(2) == 1;
			sampled[i] = (a, b, subtract);
		}
		return sampled;
	}

	/// <summary>
	/// Writes A, B (LSB-first) and the op-select bit into the input buffer.
	/// </summary>
	private static void EncodeInputs(bool[] buffer, int a, int b, bool subtract)
	{
		for (var bit = 0; bit < BIT_WIDTH; bit++)
		{
			buffer[bit] = ((a >> bit) & 1) == 1;
			buffer[BIT_WIDTH + bit] = ((b >> bit) & 1) == 1;
		}
		buffer[2 * BIT_WIDTH] = subtract;
	}

	/// <summary>
	/// Reads the network's SUM outputs (the even indices of the interleaved sum/carry layout) back
	/// into the (BitWidth+1)-bit result value, ignoring the interleaved carry outputs. Used only for
	/// the human-readable report; scoring reads the packed bits directly.
	/// </summary>
	private static int DecodeOutputs(bool[] outputs)
	{
		var value = 0;
		for (var k = 0; k <= BIT_WIDTH; k++)
		{
			if (outputs[2 * k])
				value |= 1 << k;
		}
		return value;
	}

	/// <summary>
	/// The correct output vector, packed LSB-first into an int with one bit per network output, in
	/// the interleaved sum/carry layout (sum0, c1, sum1, c2, …, sum_{W-1}, c_W, sum_W). Modelled as a
	/// ripple-carry adder/subtractor: subtraction is two's complement (a + ~b + 1), so the op-select
	/// bit both inverts b and supplies the carry-in. Exposing the internal carries c1..c_W as scored
	/// bits — not just the sum — is the whole point of this layout: it rewards the shared carry-chain
	/// structure directly instead of leaving the network to rediscover it with no gradient.
	/// </summary>
	private static int Expected(int a, int b, bool subtract)
	{
		var packed = 0;
		var carry = subtract ? 1 : 0; // two's-complement subtraction: a + ~b + 1
		for (var k = 0; k < BIT_WIDTH; k++)
		{
			var ak = (a >> k) & 1;
			var bk = (b >> k) & 1;
			var op2 = subtract ? bk ^ 1 : bk; // invert b's bits when subtracting
			var full = ak + op2 + carry;

			packed |= (full & 1) << (2 * k);       // sum_k at index 2k
			carry = full >> 1;                      // carry out of position k = c_{k+1}
			packed |= carry << (2 * k + 1);         // c_{k+1} at index 2k+1
		}

		// MSB sum bit (index 2*BitWidth): the two's-complement sign/overflow bit, which equals the
		// final ripple carry XORed with the op-select (carry-out for add, its complement for sub).
		packed |= ((subtract ? 1 : 0) ^ carry) << (2 * BIT_WIDTH);
		return packed;
	}

	#endregion
}
