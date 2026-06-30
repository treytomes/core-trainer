namespace CoreTrainer.Core;

/// <summary>
/// Canonical fixed-width binary encoding of an <see cref="Instruction"/>.
/// </summary>
/// <remarks>
/// <para>
/// The layout is exactly <see cref="BitCount"/> bits: a 4-bit opcode, two 2-bit
/// addressing modes, and two 32-bit operand values. Bits within each field are packed
/// least-significant-first.
/// </para>
/// <para>
/// <b>Decode is a total surjection:</b> every possible bit pattern maps to a valid,
/// executable instruction. This is by construction —
/// <list type="bullet">
/// <item>the 4-bit opcode field has exactly 16 patterns and there are exactly 16 opcodes (clean bijection);</item>
/// <item>the 2-bit mode fields have exactly 4 patterns and there are exactly 4 addressing modes;</item>
/// <item>the 32-bit value fields accept every pattern as a two's-complement <see cref="int"/>, which
/// the engine wraps into the core via <see cref="CoreMemory.Normalize"/>.</item>
/// </list>
/// A producer (such as an evolved network) therefore cannot emit an invalid instruction.
/// The clean-bijection guarantee is enforced by tests asserting the enum cardinalities.
/// </para>
/// </remarks>
public static class InstructionCodec
{
	#region Layout

	/// <summary>Bits used for the opcode field.</summary>
	public const int OpCodeBits = 4;

	/// <summary>Bits used for each addressing-mode field.</summary>
	public const int ModeBits = 2;

	/// <summary>Bits used for each operand value field.</summary>
	public const int ValueBits = 32;

	/// <summary>Total width of an encoded instruction.</summary>
	public const int BitCount = OpCodeBits + ModeBits + ValueBits + ModeBits + ValueBits;

	private const int OpCodeOffset = 0;
	private const int AModeOffset = OpCodeOffset + OpCodeBits;   // 4
	private const int AValueOffset = AModeOffset + ModeBits;     // 6
	private const int BModeOffset = AValueOffset + ValueBits;    // 38
	private const int BValueOffset = BModeOffset + ModeBits;     // 40

	#endregion

	#region Public Methods

	/// <summary>
	/// Encodes an instruction into a new <see cref="BitCount"/>-length bit array.
	/// </summary>
	public static bool[] Encode(Instruction instruction)
	{
		var bits = new bool[BitCount];
		Encode(instruction, bits);
		return bits;
	}

	/// <summary>
	/// Encodes an instruction into the given destination span.
	/// </summary>
	/// <param name="instruction">The instruction to encode.</param>
	/// <param name="destination">Destination span; length must equal <see cref="BitCount"/>.</param>
	public static void Encode(Instruction instruction, Span<bool> destination)
	{
		if (destination.Length != BitCount)
			throw new ArgumentException($"Destination must be exactly {BitCount} bits.", nameof(destination));

		PackBits((uint)(int)instruction.OpCode, destination.Slice(OpCodeOffset, OpCodeBits));
		PackBits((uint)(int)instruction.AMode, destination.Slice(AModeOffset, ModeBits));
		PackBits(unchecked((uint)instruction.AValue), destination.Slice(AValueOffset, ValueBits));
		PackBits((uint)(int)instruction.BMode, destination.Slice(BModeOffset, ModeBits));
		PackBits(unchecked((uint)instruction.BValue), destination.Slice(BValueOffset, ValueBits));
	}

	/// <summary>
	/// Decodes a bit pattern into an instruction. Every pattern yields a valid instruction.
	/// </summary>
	/// <param name="source">Source span; length must equal <see cref="BitCount"/>.</param>
	/// <returns>The decoded instruction.</returns>
	public static Instruction Decode(ReadOnlySpan<bool> source)
	{
		if (source.Length != BitCount)
			throw new ArgumentException($"Source must be exactly {BitCount} bits.", nameof(source));

		var opCode = (OpCode)(int)UnpackBits(source.Slice(OpCodeOffset, OpCodeBits));
		var aMode = (AddressingMode)(int)UnpackBits(source.Slice(AModeOffset, ModeBits));
		var aValue = unchecked((int)UnpackBits(source.Slice(AValueOffset, ValueBits)));
		var bMode = (AddressingMode)(int)UnpackBits(source.Slice(BModeOffset, ModeBits));
		var bValue = unchecked((int)UnpackBits(source.Slice(BValueOffset, ValueBits)));

		return new Instruction(opCode, aMode, aValue, bMode, bValue);
	}

	#endregion

	#region Private Methods

	/// <summary>
	/// Packs the low bits of <paramref name="value"/> into <paramref name="destination"/>,
	/// least-significant-bit first.
	/// </summary>
	private static void PackBits(uint value, Span<bool> destination)
	{
		for (var i = 0; i < destination.Length; i++)
		{
			destination[i] = ((value >> i) & 1u) != 0;
		}
	}

	/// <summary>
	/// Reads a bit span (least-significant-bit first) back into an unsigned integer.
	/// </summary>
	private static uint UnpackBits(ReadOnlySpan<bool> source)
	{
		uint value = 0;
		for (var i = 0; i < source.Length; i++)
		{
			if (source[i])
				value |= 1u << i;
		}
		return value;
	}

	#endregion
}
