namespace CoreTrainer.Networking.Nand;

/// <summary>
/// Decodes a <see cref="NandGenome"/> into an evaluable feed-forward NAND network and
/// computes its outputs for a given input vector.
/// </summary>
/// <remarks>
/// <para>
/// On construction the network performs an active-node analysis: starting from the
/// output sources, it marks every node reachable backwards through the wiring. Only
/// active nodes are evaluated, which is the standard CGP efficiency win — inactive
/// nodes carry genes that drift under mutation but cost nothing to run.
/// </para>
/// <para>
/// Because the genome's addressing is backwards-only, a single pass over nodes in index
/// order is already a valid evaluation order: every node's inputs (lower addresses) are
/// computed before the node itself. No topological sort is required.
/// </para>
/// <para>
/// Gates operate on single bits (boolean values), so cross-bit interaction comes from
/// the network's wiring, not from wide-word operations.
/// </para>
/// </remarks>
public sealed class NandNetwork
{
	#region Fields

	private readonly NandGenome _genome;
	private readonly NandGate _gate = new(bitSize: 1);
	private readonly bool[] _active;

	#endregion

	#region Constructors

	/// <summary>
	/// Builds an evaluable network from a genome, computing which nodes are active.
	/// </summary>
	/// <param name="genome">The genome to decode.</param>
	public NandNetwork(NandGenome genome)
	{
		ArgumentNullException.ThrowIfNull(genome);
		_genome = genome;
		_active = ComputeActiveNodes(genome);
	}

	#endregion

	#region Properties

	/// <summary>
	/// The genome this network was built from.
	/// </summary>
	public NandGenome Genome => _genome;

	/// <summary>
	/// Number of nodes that contribute to an output (and are therefore evaluated).
	/// </summary>
	public int ActiveNodeCount
	{
		get
		{
			var count = 0;
			foreach (var isActive in _active)
			{
				if (isActive) count++;
			}
			return count;
		}
	}

	#endregion

	#region Methods

	/// <summary>
	/// Evaluates the network for the given input bits and returns the output bits.
	/// </summary>
	/// <param name="inputs">
	/// Input bit values, one per network input (length must equal <c>Genome.InputCount</c>).
	/// </param>
	/// <returns>Output bit values, one per network output.</returns>
	/// <exception cref="ArgumentException"><paramref name="inputs"/> has the wrong length.</exception>
	public bool[] Evaluate(ReadOnlySpan<bool> inputs)
	{
		if (inputs.Length != _genome.InputCount)
			throw new ArgumentException(
				$"Expected {_genome.InputCount} inputs but got {inputs.Length}.", nameof(inputs));

		// values[address] holds the bit at each address: inputs occupy the low addresses,
		// nodes follow. Only active nodes are computed; inactive ones stay false (unused).
		var values = new bool[_genome.AddressableCount];
		for (var i = 0; i < inputs.Length; i++)
		{
			values[i] = inputs[i];
		}

		for (var nodeIndex = 0; nodeIndex < _genome.NodeCount; nodeIndex++)
		{
			if (!_active[nodeIndex])
				continue;

			var (aAddr, bAddr) = _genome.GetNode(nodeIndex);
			var a = values[aAddr] ? 1UL : 0UL;
			var b = values[bAddr] ? 1UL : 0UL;
			values[_genome.InputCount + nodeIndex] = _gate.Apply(a, b) != 0UL;
		}

		var outputs = new bool[_genome.OutputCount];
		for (var o = 0; o < _genome.OutputCount; o++)
		{
			outputs[o] = values[_genome.GetOutputSource(o)];
		}

		return outputs;
	}

	#endregion

	#region Private Methods

	/// <summary>
	/// Marks every node reachable backwards from the output sources. The backwards-only
	/// invariant lets this run in a single reverse pass: a node's inputs always have
	/// lower addresses, so they are visited after the node that references them.
	/// </summary>
	private static bool[] ComputeActiveNodes(NandGenome genome)
	{
		var active = new bool[genome.NodeCount];

		// Seed from outputs: any node an output reads from is active.
		for (var o = 0; o < genome.OutputCount; o++)
		{
			MarkIfNode(genome, active, genome.GetOutputSource(o));
		}

		// Reverse pass: when an active node is found, mark the nodes feeding it. Walking
		// high-to-low guarantees we reach a node before processing the ones it depends on.
		for (var nodeIndex = genome.NodeCount - 1; nodeIndex >= 0; nodeIndex--)
		{
			if (!active[nodeIndex])
				continue;

			var (aAddr, bAddr) = genome.GetNode(nodeIndex);
			MarkIfNode(genome, active, aAddr);
			MarkIfNode(genome, active, bAddr);
		}

		return active;
	}

	/// <summary>
	/// Marks the address as an active node, if it refers to a node (not an input).
	/// </summary>
	private static void MarkIfNode(NandGenome genome, bool[] active, int address)
	{
		if (address >= genome.InputCount)
		{
			active[address - genome.InputCount] = true;
		}
	}

	#endregion
}
