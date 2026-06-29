namespace CoreTrainer.Networking.Nand;

/// <summary>
/// A NAND gate operating on fixed-width bit vectors. NAND is functionally complete,
/// so networks of these gates can express any boolean function — the basis for a
/// trainable NAND network.
/// </summary>
public readonly struct NandGate
{
	#region Fields

	private readonly ulong _mask;

	#endregion

	#region Constructors

	/// <summary>
	/// Creates a NAND gate operating on <paramref name="bitSize"/>-wide inputs.
	/// </summary>
	/// <param name="bitSize">Number of bits per input (1 to 64).</param>
	/// <exception cref="ArgumentOutOfRangeException">
	/// <paramref name="bitSize"/> is less than 1 or greater than 64.
	/// </exception>
	public NandGate(int bitSize)
	{
		if (bitSize < 1 || bitSize > 64)
			throw new ArgumentOutOfRangeException(nameof(bitSize), bitSize, "Bit size must be between 1 and 64.");

		BitSize = bitSize;
		// All-ones for the low `bitSize` bits. Shifting by 64 is undefined in C#, so
		// the full-width case is handled explicitly.
		_mask = bitSize == 64 ? ulong.MaxValue : (1UL << bitSize) - 1;
	}

	#endregion

	#region Properties

	/// <summary>
	/// Number of bits this gate operates on.
	/// </summary>
	public int BitSize { get; }

	#endregion

	#region Methods

	/// <summary>
	/// Performs a bitwise NAND of the two inputs, truncated to <see cref="BitSize"/> bits.
	/// </summary>
	/// <param name="a">First input.</param>
	/// <param name="b">Second input.</param>
	/// <returns><c>~(a &amp; b)</c> masked to the gate's bit width.</returns>
	public ulong Apply(ulong a, ulong b) => ~(a & b) & _mask;

	#endregion
}
