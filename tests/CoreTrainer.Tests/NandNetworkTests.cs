using CoreTrainer.Networking.Nand;

namespace CoreTrainer.Tests;

/// <summary>
/// Tests for decoding and evaluating a NAND network. Validated by hand-wiring classic
/// boolean functions from NAND gates and checking their full truth tables.
/// </summary>
public class NandNetworkTests
{
	#region Single-gate behavior

	[Fact]
	public void Evaluate_SingleNandNode_MatchesTruthTable()
	{
		// 2 inputs (addr 0,1), 1 node = NAND(0,1) at addr 2, output reads addr 2.
		var genome = new NandGenome(inputCount: 2, outputCount: 1,
			nodeGenes: [0, 1], outputGenes: [2]);
		var net = new NandNetwork(genome);

		Assert.True(net.Evaluate([false, false])[0]);
		Assert.True(net.Evaluate([false, true])[0]);
		Assert.True(net.Evaluate([true, false])[0]);
		Assert.False(net.Evaluate([true, true])[0]);
	}

	[Fact]
	public void Evaluate_OutputReadingInputDirectly_PassesInputThrough()
	{
		// No nodes; the single output reads input address 0 directly.
		var genome = new NandGenome(inputCount: 2, outputCount: 1,
			nodeGenes: [], outputGenes: [0]);
		var net = new NandNetwork(genome);

		Assert.False(net.Evaluate([false, true])[0]);
		Assert.True(net.Evaluate([true, false])[0]);
	}

	#endregion

	#region Composed circuits

	[Fact]
	public void Evaluate_NotFromNand_InvertsInput()
	{
		// NOT(x) = NAND(x, x). 1 input, node0 = NAND(0,0) at addr 1, output reads addr 1.
		var genome = new NandGenome(inputCount: 1, outputCount: 1,
			nodeGenes: [0, 0], outputGenes: [1]);
		var net = new NandNetwork(genome);

		Assert.True(net.Evaluate([false])[0]);
		Assert.False(net.Evaluate([true])[0]);
	}

	[Fact]
	public void Evaluate_AndFromNand_MatchesTruthTable()
	{
		// AND(a,b) = NOT(NAND(a,b)).
		// node0 = NAND(0,1) @2 ; node1 = NAND(2,2) @3 ; output reads 3.
		var genome = new NandGenome(inputCount: 2, outputCount: 1,
			nodeGenes: [0, 1, 2, 2], outputGenes: [3]);
		var net = new NandNetwork(genome);

		Assert.False(net.Evaluate([false, false])[0]);
		Assert.False(net.Evaluate([false, true])[0]);
		Assert.False(net.Evaluate([true, false])[0]);
		Assert.True(net.Evaluate([true, true])[0]);
	}

	[Fact]
	public void Evaluate_XorFromNand_MatchesTruthTable()
	{
		// XOR from four NANDs:
		// node0 = NAND(0,1)      @2
		// node1 = NAND(0,node0)  @3  = NAND(0,2)
		// node2 = NAND(1,node0)  @4  = NAND(1,2)
		// node3 = NAND(node1,node2) @5 = NAND(3,4) = XOR
		var genome = new NandGenome(inputCount: 2, outputCount: 1,
			nodeGenes: [0, 1, 0, 2, 1, 2, 3, 4], outputGenes: [5]);
		var net = new NandNetwork(genome);

		Assert.False(net.Evaluate([false, false])[0]);
		Assert.True(net.Evaluate([false, true])[0]);
		Assert.True(net.Evaluate([true, false])[0]);
		Assert.False(net.Evaluate([true, true])[0]);
	}

	#endregion

	#region Active-node analysis

	[Fact]
	public void ActiveNodeCount_IgnoresNodesNotReachableFromOutput()
	{
		// node0 = NAND(0,1) @2 feeds the output; node1 = NAND(0,1) @3 is dangling.
		var genome = new NandGenome(inputCount: 2, outputCount: 1,
			nodeGenes: [0, 1, 0, 1], outputGenes: [2]);
		var net = new NandNetwork(genome);

		Assert.Equal(2, genome.NodeCount);
		Assert.Equal(1, net.ActiveNodeCount); // only node0 is active
	}

	[Fact]
	public void Evaluate_InactiveNodesDoNotAffectOutput()
	{
		// Same as the single-NAND case but with an extra dangling node; output must match.
		var withDangling = new NandNetwork(new NandGenome(inputCount: 2, outputCount: 1,
			nodeGenes: [0, 1, 0, 1], outputGenes: [2]));

		Assert.False(withDangling.Evaluate([true, true])[0]);
		Assert.True(withDangling.Evaluate([true, false])[0]);
	}

	#endregion

	#region Validation

	[Fact]
	public void Evaluate_WrongInputLength_Throws()
	{
		var net = new NandNetwork(new NandGenome(inputCount: 2, outputCount: 1,
			nodeGenes: [0, 1], outputGenes: [2]));

		Assert.Throws<ArgumentException>(() => net.Evaluate([true]));
	}

	#endregion
}
