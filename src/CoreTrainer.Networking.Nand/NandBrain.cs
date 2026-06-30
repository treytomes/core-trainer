using CoreTrainer.Core;

namespace CoreTrainer.Networking.Nand;

/// <summary>
/// A warrior brain driven by an evolved NAND network. Each cycle it encodes the
/// warrior's observation into input bits, evaluates the network, and decodes the output
/// bits into the instruction to execute.
/// </summary>
/// <remarks>
/// This is the glue layer between the pure CGP network (which knows only bits) and the
/// simulator (which knows only <see cref="IWarriorBrain"/>). It owns a
/// <see cref="WarriorObservationCodec"/> whose input/output widths must match the
/// network's, and validates that pairing on construction.
/// </remarks>
public sealed class NandBrain : IWarriorBrain
{
	#region Fields

	private readonly NandNetwork _network;
	private readonly WarriorObservationCodec _codec;

	#endregion

	#region Constructors

	/// <summary>
	/// Creates a brain from a network and an observation codec whose widths it matches.
	/// </summary>
	/// <param name="network">The evaluable NAND network.</param>
	/// <param name="codec">The observation codec defining input/output bit layout.</param>
	/// <exception cref="ArgumentException">
	/// The network's input/output widths do not match the codec's.
	/// </exception>
	public NandBrain(NandNetwork network, WarriorObservationCodec codec)
	{
		ArgumentNullException.ThrowIfNull(network);
		ArgumentNullException.ThrowIfNull(codec);

		if (network.Genome.InputCount != codec.InputBitCount)
			throw new ArgumentException(
				$"Network input count ({network.Genome.InputCount}) must match codec input bit count ({codec.InputBitCount}).",
				nameof(network));
		if (network.Genome.OutputCount != codec.OutputBitCount)
			throw new ArgumentException(
				$"Network output count ({network.Genome.OutputCount}) must match codec output bit count ({codec.OutputBitCount}).",
				nameof(network));

		_network = network;
		_codec = codec;
	}

	#endregion

	#region IWarriorBrain

	/// <inheritdoc />
	public int WindowRadius => _codec.WindowRadius;

	/// <inheritdoc />
	public Instruction Decide(IReadOnlyList<Instruction> window, int programCounter, int processCount)
	{
		var inputs = _codec.EncodeObservation(window, programCounter, processCount);
		var outputs = _network.Evaluate(inputs);
		return _codec.DecodeInstruction(outputs);
	}

	#endregion
}
