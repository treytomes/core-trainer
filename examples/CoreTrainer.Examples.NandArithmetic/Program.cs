using CoreTrainer.Networking.Nand;

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
	/// Swap this to compare sampled vs. exhaustive scoring (see <see cref="FitnessMode"/>).
	/// </summary>
	private const FitnessMode FITNESS_MODE = FitnessMode.Sampled;

	private const int BIT_WIDTH = 4;

	/// <summary>
	/// NAND gates available to wire together
	/// </summary>
	private const int NODE_COUNT = 600;
	private const int GENERATIONS = 30000;

	/// <summary>
	/// Offspring per generation.
	/// </summary>
	private const int LAMBDA = 4;
	private const double MUTATION_RATE = 0.03;

	/// <summary>
	/// Sampled (A, B, op) cases used to score a genome.
	/// </summary>
	private const int TEST_CASES = 256;

	/// <summary>
	/// Generations between progress lines.
	/// </summary>
	private const int PROGRESS_EVERY = 100;

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
		// Input layout:  [A bits (BitWidth)] [B bits (BitWidth)] [opSelect (1)]
		//   opSelect = false -> A + B,  true -> A - B
		// Output layout: BitWidth + 1 bits (result plus the carry/borrow bit), so addition's
		// overflow and subtraction's two's-complement borrow both fit losslessly.
		var inputCount = 2 * BIT_WIDTH + 1;
		var outputCount = BIT_WIDTH + 1;
		var outputMask = (1 << outputCount) - 1;

		// Build the set of (A, B, op) cases the fitness function scores against. The only
		// difference between sampled and exhaustive fitness is WHICH cases this returns —
		// the scoring logic below is identical either way.
		var cases = BuildCases(FITNESS_MODE, BIT_WIDTH);
		var scoringDescription = FITNESS_MODE == FitnessMode.Exhaustive
			? $"bit accuracy over ALL {cases.Length} input combinations (exhaustive)"
			: $"bit accuracy over {cases.Length} sampled cases";

		Console.WriteLine("=== NAND Arithmetic Trainer ===");
		Console.WriteLine($"Goal:    evolve a {BIT_WIDTH}-bit adder/subtractor from NAND gates");
		Console.WriteLine($"Network: {inputCount} inputs, {NODE_COUNT} nodes, {outputCount} outputs");
		Console.WriteLine($"Search:  1+{LAMBDA}, {GENERATIONS} generations, mutation {MUTATION_RATE:P1}");
		Console.WriteLine($"Scoring: {scoringDescription}\n");

		// === Fitness: fraction of output bits the network gets right ==========================
		// Bit-level (rather than whole-answer) scoring gives a smooth gradient: a genome that
		// gets 7 of 9 bits right scores higher than one that gets 3, so the search has something
		// to climb even before any answer is fully correct. A purely random circuit hovers near
		// 0.5 (each bit is a coin flip).
		double Fitness(NandGenome genome)
		{
			var network = new NandNetwork(genome);
			var inputs = new bool[inputCount];
			var correctBits = 0;

			foreach (var (a, b, subtract) in cases)
			{
				EncodeInputs(inputs, a, b, subtract);
				var outputs = network.Evaluate(inputs);

				var expected = Expected(a, b, subtract) & outputMask;
				for (var bit = 0; bit < outputCount; bit++)
				{
					var actualBit = outputs[bit];
					var expectedBit = ((expected >> bit) & 1) == 1;
					if (actualBit == expectedBit)
						correctBits++;
				}
			}

			return (double)correctBits / (cases.Length * outputCount);
		}

		// === Run the evolution ===============================================================
		var random = new Random(EVOLUTION_SEED);
		var initial = NandGenome.CreateRandom(inputCount, NODE_COUNT, outputCount, random);
		var evolution = new CgpEvolution(Fitness, LAMBDA, MUTATION_RATE, random);

		Console.WriteLine($"Starting accuracy: {Fitness(initial):P2}\n");

		var result = evolution.Evolve(initial, GENERATIONS, onGeneration: progress =>
		{
			// progress.Generation is zero-based; report on the first and every ProgressEvery-th.
			if (progress.Generation == 0 || (progress.Generation + 1) % PROGRESS_EVERY == 0)
			{
				// Active-gate count shows how much of the NODE_COUNT pool the solution
				// actually recruits — watch it grow as the circuit takes shape.
				var activeGates = new NandNetwork(progress.Best).ActiveNodeCount;
				Console.WriteLine(
					$"  gen {progress.Generation + 1,5}/{GENERATIONS}   accuracy {progress.Fitness:P2}   active gates {activeGates,4}/{NODE_COUNT}");
			}
		});

		// === Report =========================================================================
		Console.WriteLine($"\nFinal accuracy: {result.Fitness:P2}  (active gates: {new NandNetwork(result.Best).ActiveNodeCount}/{NODE_COUNT})\n");

		// Show a handful of concrete evaluations so the result is tangible.
		Console.WriteLine("Sample evaluations (after training):");
		var best = new NandNetwork(result.Best);
		var sampleInputs = new bool[inputCount];
		for (var i = 0; i < 8; i++)
		{
			var (a, b, subtract) = cases[i];
			EncodeInputs(sampleInputs, a, b, subtract);
			var got = DecodeOutputs(best.Evaluate(sampleInputs));
			var expected = Expected(a, b, subtract) & outputMask;
			var op = subtract ? "-" : "+";
			var mark = got == expected ? "OK " : "xx ";
			Console.WriteLine($"  {mark} {a,3} {op} {b,3}  expected {expected,4}  got {got,4}");
		}

	}

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
		var range = 1 << bitWidth;

		if (mode == FitnessMode.Exhaustive)
		{
			// Every A, every B, both operations: 2^inputCount total combinations.
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

		// Sampled: a fixed random subset. The fixed seed keeps the set (and thus fitness)
		// stable across genomes and runs.
		var testRandom = new Random(TEST_SEED);
		var sampled = new (int, int, bool)[TEST_CASES];
		for (var i = 0; i < TEST_CASES; i++)
		{
			var a = testRandom.Next(range);
			var b = testRandom.Next(range);
			var subtract = testRandom.Next(2) == 1;
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
	/// Reads the network's output bits (LSB-first) back into an integer.
	/// </summary>
	private static int DecodeOutputs(bool[] outputs)
	{
		var value = 0;
		for (var bit = 0; bit < outputs.Length; bit++)
		{
			if (outputs[bit])
				value |= 1 << bit;
		}
		return value;
	}

	/// <summary>
	/// The correct answer, as a (BitWidth+1)-bit pattern. Subtraction uses two's complement so
	/// every result — including "negative" ones — maps onto a valid bit pattern.
	/// </summary>
	private static int Expected(int a, int b, bool subtract) => subtract ? a - b : a + b;

	#endregion
}
