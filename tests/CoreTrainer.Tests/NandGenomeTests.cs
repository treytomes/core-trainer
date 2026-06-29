using CoreTrainer.Networking.Nand;

namespace CoreTrainer.Tests;

/// <summary>
/// Tests for the CGP genome — especially the backwards-only invariant that makes
/// cyclic or out-of-range wiring unrepresentable.
/// </summary>
public class NandGenomeTests
{
	#region Construction and accessors

	[Fact]
	public void Constructor_ExposesShapeAndGenes()
	{
		// 2 inputs (addresses 0,1), 2 nodes (addresses 2,3), 1 output.
		var nodeGenes = new[] { 0, 1, /* node0 */ 1, 2 /* node1 */ };
		var outputGenes = new[] { 3 };

		var genome = new NandGenome(inputCount: 2, outputCount: 1, nodeGenes, outputGenes);

		Assert.Equal(2, genome.InputCount);
		Assert.Equal(1, genome.OutputCount);
		Assert.Equal(2, genome.NodeCount);
		Assert.Equal(4, genome.AddressableCount); // 2 inputs + 2 nodes
		Assert.Equal((0, 1), genome.GetNode(0));
		Assert.Equal((1, 2), genome.GetNode(1));
		Assert.Equal(3, genome.GetOutputSource(0));
	}

	#endregion

	#region Backwards-only invariant

	[Fact]
	public void Constructor_RejectsNodeReferencingItself()
	{
		// Node 0 lives at address 2 and references address 2 — a self-loop (cycle).
		var nodeGenes = new[] { 2, 0 };
		var outputGenes = new[] { 2 };

		Assert.Throws<ArgumentException>(
			() => new NandGenome(inputCount: 2, outputCount: 1, nodeGenes, outputGenes));
	}

	[Fact]
	public void Constructor_RejectsNodeReferencingLaterNode()
	{
		// Node 0 (address 2) references node 1 (address 3) — a forward edge (cycle risk).
		var nodeGenes = new[] { 3, 0, /* node0 */ 0, 1 /* node1 */ };
		var outputGenes = new[] { 3 };

		Assert.Throws<ArgumentException>(
			() => new NandGenome(inputCount: 2, outputCount: 1, nodeGenes, outputGenes));
	}

	[Fact]
	public void Constructor_RejectsOutputAddressOutOfRange()
	{
		var nodeGenes = new[] { 0, 1 };
		var outputGenes = new[] { 99 }; // beyond AddressableCount (3)

		Assert.Throws<ArgumentException>(
			() => new NandGenome(inputCount: 2, outputCount: 1, nodeGenes, outputGenes));
	}

	[Fact]
	public void Constructor_RejectsOddNodeGeneArray()
	{
		Assert.Throws<ArgumentException>(
			() => new NandGenome(inputCount: 2, outputCount: 1, [0, 1, 2], [2]));
	}

	#endregion

	#region Random construction and mutation

	[Fact]
	public void CreateRandom_ProducesValidGenomeOfRequestedShape()
	{
		var random = new Random(123);

		var genome = NandGenome.CreateRandom(inputCount: 4, nodeCount: 20, outputCount: 8, random);

		Assert.Equal(4, genome.InputCount);
		Assert.Equal(20, genome.NodeCount);
		Assert.Equal(8, genome.OutputCount);
		// Construction validates wiring, so reaching here proves the invariant holds.
	}

	[Fact]
	public void CreateRandom_ManyGenomes_AllSatisfyBackwardsOnlyInvariant()
	{
		var random = new Random(7);

		for (var trial = 0; trial < 200; trial++)
		{
			var genome = NandGenome.CreateRandom(inputCount: 3, nodeCount: 30, outputCount: 5, random);

			for (var i = 0; i < genome.NodeCount; i++)
			{
				var (a, b) = genome.GetNode(i);
				var upperBound = genome.InputCount + i;
				Assert.InRange(a, 0, upperBound - 1);
				Assert.InRange(b, 0, upperBound - 1);
			}
		}
	}

	[Fact]
	public void Mutate_AlwaysProducesValidGenome()
	{
		var random = new Random(42);
		var genome = NandGenome.CreateRandom(inputCount: 4, nodeCount: 25, outputCount: 6, random);

		// High mutation rate stresses the range logic; every result must still validate.
		for (var i = 0; i < 200; i++)
		{
			genome = genome.Mutate(mutationRate: 0.5, random);
			Assert.Equal(25, genome.NodeCount);
			Assert.Equal(6, genome.OutputCount);
		}
	}

	[Fact]
	public void Mutate_DoesNotModifyOriginal()
	{
		var random = new Random(99);
		var original = NandGenome.CreateRandom(inputCount: 3, nodeCount: 10, outputCount: 3, random);
		var originalNodes = original.NodeGenes.ToArray();

		_ = original.Mutate(mutationRate: 1.0, random);

		Assert.Equal(originalNodes, original.NodeGenes.ToArray());
	}

	#endregion
}
