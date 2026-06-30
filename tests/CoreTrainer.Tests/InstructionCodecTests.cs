using CoreTrainer.Core;

namespace CoreTrainer.Tests;

/// <summary>
/// Tests for the instruction binary codec — round-trip fidelity and the totality
/// guarantee that any bit pattern decodes to a valid instruction.
/// </summary>
public class InstructionCodecTests
{
	#region Layout invariants

	[Fact]
	public void OpCodeField_ExactlyCoversTheOpCodeEnum()
	{
		// The clean-bijection guarantee: 4 bits == 16 patterns == 16 opcodes.
		var opCodeCount = Enum.GetValues<OpCode>().Length;
		Assert.Equal(16, opCodeCount);
		Assert.Equal(opCodeCount, 1 << InstructionCodec.OpCodeBits);
	}

	[Fact]
	public void ModeField_ExactlyCoversTheAddressingModeEnum()
	{
		var modeCount = Enum.GetValues<AddressingMode>().Length;
		Assert.Equal(4, modeCount);
		Assert.Equal(modeCount, 1 << InstructionCodec.ModeBits);
	}

	[Fact]
	public void BitCount_Is72()
	{
		Assert.Equal(72, InstructionCodec.BitCount);
	}

	#endregion

	#region Round trip

	[Fact]
	public void EncodeThenDecode_PreservesAllFields()
	{
		var original = new Instruction(OpCode.SPL, AddressingMode.Indirect, -1234, AddressingMode.Decrement, 567890);

		var decoded = InstructionCodec.Decode(InstructionCodec.Encode(original));

		Assert.Equal(original.OpCode, decoded.OpCode);
		Assert.Equal(original.AMode, decoded.AMode);
		Assert.Equal(original.AValue, decoded.AValue);
		Assert.Equal(original.BMode, decoded.BMode);
		Assert.Equal(original.BValue, decoded.BValue);
	}

	[Theory]
	[InlineData(0)]
	[InlineData(int.MaxValue)]
	[InlineData(int.MinValue)]
	[InlineData(-1)]
	public void EncodeThenDecode_PreservesExtremeOperandValues(int value)
	{
		var original = new Instruction(OpCode.MOV, AddressingMode.Direct, value, AddressingMode.Immediate, value);

		var decoded = InstructionCodec.Decode(InstructionCodec.Encode(original));

		Assert.Equal(value, decoded.AValue);
		Assert.Equal(value, decoded.BValue);
	}

	[Fact]
	public void EncodeThenDecode_PreservesEveryOpCodeAndMode()
	{
		foreach (var op in Enum.GetValues<OpCode>())
		{
			foreach (var mode in Enum.GetValues<AddressingMode>())
			{
				var original = new Instruction(op, mode, 1, mode, -1);
				var decoded = InstructionCodec.Decode(InstructionCodec.Encode(original));
				Assert.Equal(op, decoded.OpCode);
				Assert.Equal(mode, decoded.AMode);
			}
		}
	}

	#endregion

	#region Totality

	[Fact]
	public void Decode_AnyBitPattern_YieldsValidInstruction()
	{
		var random = new Random(2024);
		var validOpCodes = Enum.GetValues<OpCode>().ToHashSet();
		var validModes = Enum.GetValues<AddressingMode>().ToHashSet();

		for (var trial = 0; trial < 5000; trial++)
		{
			var bits = new bool[InstructionCodec.BitCount];
			for (var i = 0; i < bits.Length; i++)
				bits[i] = random.Next(2) == 1;

			var instruction = InstructionCodec.Decode(bits);

			// "Valid" = opcode and both modes are defined enum members. Operand values
			// are always valid (every int is a legal, wrappable offset).
			Assert.Contains(instruction.OpCode, validOpCodes);
			Assert.Contains(instruction.AMode, validModes);
			Assert.Contains(instruction.BMode, validModes);
		}
	}

	[Fact]
	public void Decode_AllOnes_YieldsLastOpCodeAndMode()
	{
		var bits = new bool[InstructionCodec.BitCount];
		Array.Fill(bits, true);

		var instruction = InstructionCodec.Decode(bits);

		// 0b1111 = 15 = NOP; 0b11 = 3 = Decrement; all-ones value = -1 (two's complement).
		Assert.Equal(OpCode.NOP, instruction.OpCode);
		Assert.Equal(AddressingMode.Decrement, instruction.AMode);
		Assert.Equal(-1, instruction.AValue);
	}

	#endregion

	#region Validation

	[Fact]
	public void Decode_WrongLength_Throws()
	{
		Assert.Throws<ArgumentException>(() => InstructionCodec.Decode(new bool[10]));
	}

	#endregion
}
