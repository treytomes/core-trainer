namespace CoreTrainer.Networking.Nand;

/// <summary>
/// A Cartesian Genetic Programming (CGP) genome describing a feed-forward network of
/// NAND gates. Because the only primitive is NAND, a node carries no function gene —
/// it is just two input addresses. A separate block of output genes selects which
/// addresses the network's outputs read from.
/// </summary>
/// <remarks>
/// <para>
/// Addressing is absolute and <b>backwards-only</b>: addresses <c>0 .. InputCount-1</c>
/// refer to network inputs, and address <c>InputCount + i</c> refers to node <c>i</c>.
/// A node may only reference addresses strictly less than its own, which guarantees an
/// acyclic, feed-forward graph by construction — no cycle check is ever needed. The
/// constructor enforces this invariant, so a <see cref="NandGenome"/> can never hold a
/// cyclic or out-of-range wiring.
/// </para>
/// <para>
/// Inactive nodes (those not reachable from any output) are intentionally retained.
/// Their genes drift freely under mutation and occasionally become wired in — this
/// neutral search is a key part of why CGP escapes local optima.
/// </para>
/// <para>
/// The genome is immutable; <see cref="Mutate"/> returns a new instance. It is
/// domain-agnostic: the meaning of each input and output bit (e.g. mapping outputs onto
/// an instruction's fields) belongs to a separate codec, not here.
/// </para>
/// </remarks>
public sealed class NandGenome
{
	#region Fields

	// Two input addresses per node, laid out flat: [a0, b0, a1, b1, ...].
	private readonly int[] _nodeGenes;

	// One source address per network output.
	private readonly int[] _outputGenes;

	#endregion

	#region Constructors

	/// <summary>
	/// Creates a genome from explicit gene arrays, validating the backwards-only invariant.
	/// </summary>
	/// <param name="inputCount">Number of network input bits (at least 1).</param>
	/// <param name="outputCount">Number of network output bits (at least 1).</param>
	/// <param name="nodeGenes">
	/// Flat array of node input addresses, two per node: <c>[a0, b0, a1, b1, ...]</c>.
	/// Length must be even.
	/// </param>
	/// <param name="outputGenes">One source address per network output.</param>
	/// <exception cref="ArgumentOutOfRangeException"><paramref name="inputCount"/> or <paramref name="outputCount"/> is less than 1.</exception>
	/// <exception cref="ArgumentNullException">A gene array is null.</exception>
	/// <exception cref="ArgumentException">A gene array has the wrong shape or violates the backwards-only invariant.</exception>
	public NandGenome(int inputCount, int outputCount, int[] nodeGenes, int[] outputGenes)
	{
		if (inputCount < 1)
			throw new ArgumentOutOfRangeException(nameof(inputCount), inputCount, "Input count must be at least 1.");
		if (outputCount < 1)
			throw new ArgumentOutOfRangeException(nameof(outputCount), outputCount, "Output count must be at least 1.");
		ArgumentNullException.ThrowIfNull(nodeGenes);
		ArgumentNullException.ThrowIfNull(outputGenes);
		if (nodeGenes.Length % 2 != 0)
			throw new ArgumentException("Node gene array must contain two addresses per node (even length).", nameof(nodeGenes));
		if (outputGenes.Length != outputCount)
			throw new ArgumentException("Output gene array length must equal the output count.", nameof(outputGenes));

		InputCount = inputCount;
		OutputCount = outputCount;

		// Defensive copies keep the genome immutable even if the caller mutates its arrays.
		_nodeGenes = (int[])nodeGenes.Clone();
		_outputGenes = (int[])outputGenes.Clone();

		ValidateWiring();
	}

	#endregion

	#region Properties

	/// <summary>
	/// Number of network input bits.
	/// </summary>
	public int InputCount { get; }

	/// <summary>
	/// Number of network output bits.
	/// </summary>
	public int OutputCount { get; }

	/// <summary>
	/// Number of NAND nodes in the genome (including inactive ones).
	/// </summary>
	public int NodeCount => _nodeGenes.Length / 2;

	/// <summary>
	/// Total number of addressable sources: inputs plus nodes. Output genes and node
	/// inputs reference addresses in <c>[0, AddressableCount)</c>.
	/// </summary>
	public int AddressableCount => InputCount + NodeCount;

	/// <summary>
	/// The flat node connection genes (two addresses per node). Read-only view.
	/// </summary>
	public IReadOnlyList<int> NodeGenes => _nodeGenes;

	/// <summary>
	/// The output source genes. Read-only view.
	/// </summary>
	public IReadOnlyList<int> OutputGenes => _outputGenes;

	#endregion

	#region Methods

	/// <summary>
	/// Gets the two input addresses feeding the given node.
	/// </summary>
	/// <param name="nodeIndex">Zero-based node index.</param>
	/// <returns>The node's two input source addresses.</returns>
	public (int A, int B) GetNode(int nodeIndex)
	{
		if (nodeIndex < 0 || nodeIndex >= NodeCount)
			throw new ArgumentOutOfRangeException(nameof(nodeIndex));

		return (_nodeGenes[nodeIndex * 2], _nodeGenes[nodeIndex * 2 + 1]);
	}

	/// <summary>
	/// Gets the source address that the given network output reads from.
	/// </summary>
	/// <param name="outputIndex">Zero-based output index.</param>
	/// <returns>The address feeding that output.</returns>
	public int GetOutputSource(int outputIndex)
	{
		if (outputIndex < 0 || outputIndex >= OutputCount)
			throw new ArgumentOutOfRangeException(nameof(outputIndex));

		return _outputGenes[outputIndex];
	}

	/// <summary>
	/// Creates a random valid genome of the given shape.
	/// </summary>
	/// <param name="inputCount">Number of network input bits (at least 1).</param>
	/// <param name="nodeCount">Number of NAND nodes (zero or more).</param>
	/// <param name="outputCount">Number of network output bits (at least 1).</param>
	/// <param name="random">Random source.</param>
	/// <returns>A new genome wired entirely within the backwards-only constraint.</returns>
	public static NandGenome CreateRandom(int inputCount, int nodeCount, int outputCount, Random random)
	{
		if (inputCount < 1)
			throw new ArgumentOutOfRangeException(nameof(inputCount), inputCount, "Input count must be at least 1.");
		if (nodeCount < 0)
			throw new ArgumentOutOfRangeException(nameof(nodeCount), nodeCount, "Node count cannot be negative.");
		if (outputCount < 1)
			throw new ArgumentOutOfRangeException(nameof(outputCount), outputCount, "Output count must be at least 1.");
		ArgumentNullException.ThrowIfNull(random);

		var nodeGenes = new int[nodeCount * 2];
		for (var i = 0; i < nodeCount; i++)
		{
			// Node i may only reference earlier addresses: inputs and nodes 0..i-1.
			var upperBound = inputCount + i;
			nodeGenes[i * 2] = random.Next(upperBound);
			nodeGenes[i * 2 + 1] = random.Next(upperBound);
		}

		var addressable = inputCount + nodeCount;
		var outputGenes = new int[outputCount];
		for (var o = 0; o < outputCount; o++)
		{
			outputGenes[o] = random.Next(addressable);
		}

		return new NandGenome(inputCount, outputCount, nodeGenes, outputGenes);
	}

	/// <summary>
	/// Produces a mutated copy via point mutation: each gene is independently
	/// re-randomized (within its valid backwards-only range) with probability
	/// <paramref name="mutationRate"/>. The original is unchanged.
	/// </summary>
	/// <param name="mutationRate">Per-gene mutation probability in [0, 1].</param>
	/// <param name="random">Random source.</param>
	/// <returns>A new, valid genome.</returns>
	public NandGenome Mutate(double mutationRate, Random random)
	{
		ArgumentNullException.ThrowIfNull(random);
		var rate = Math.Clamp(mutationRate, 0.0, 1.0);

		var nodeGenes = (int[])_nodeGenes.Clone();
		for (var i = 0; i < NodeCount; i++)
		{
			var upperBound = InputCount + i;
			if (random.NextDouble() < rate)
				nodeGenes[i * 2] = random.Next(upperBound);
			if (random.NextDouble() < rate)
				nodeGenes[i * 2 + 1] = random.Next(upperBound);
		}

		var addressable = AddressableCount;
		var outputGenes = (int[])_outputGenes.Clone();
		for (var o = 0; o < OutputCount; o++)
		{
			if (random.NextDouble() < rate)
				outputGenes[o] = random.Next(addressable);
		}

		return new NandGenome(InputCount, OutputCount, nodeGenes, outputGenes);
	}

	#endregion

	#region Private Methods

	/// <summary>
	/// Verifies that every node references only earlier addresses and every output
	/// references a valid address — the invariant guaranteeing an acyclic graph.
	/// </summary>
	private void ValidateWiring()
	{
		for (var i = 0; i < NodeCount; i++)
		{
			var upperBound = InputCount + i; // exclusive
			var a = _nodeGenes[i * 2];
			var b = _nodeGenes[i * 2 + 1];

			if (a < 0 || a >= upperBound)
				throw new ArgumentException(
					$"Node {i} input A address {a} must be in [0, {upperBound}) to keep the graph acyclic.", nameof(_nodeGenes));
			if (b < 0 || b >= upperBound)
				throw new ArgumentException(
					$"Node {i} input B address {b} must be in [0, {upperBound}) to keep the graph acyclic.", nameof(_nodeGenes));
		}

		for (var o = 0; o < OutputCount; o++)
		{
			var address = _outputGenes[o];
			if (address < 0 || address >= AddressableCount)
				throw new ArgumentException(
					$"Output {o} address {address} must be in [0, {AddressableCount}).", nameof(_outputGenes));
		}
	}

	#endregion
}
