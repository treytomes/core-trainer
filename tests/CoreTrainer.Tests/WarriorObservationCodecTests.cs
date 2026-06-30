using CoreTrainer.Core;

namespace CoreTrainer.Tests;

/// <summary>
/// Tests for the warrior observation codec: input sizing/assembly and output decoding.
/// </summary>
public class WarriorObservationCodecTests
{
	#region Sizing

	[Fact]
	public void InputBitCount_AccountsForWindowPcAndProcessCount()
	{
		var codec = new WarriorObservationCodec(windowRadius: 2, coreSize: 8000, maxProcesses: 8000);

		// 5 cells (radius 2) of 72 bits, plus PC bits (8000 -> 13) plus process bits (8001 -> 13).
		Assert.Equal(5, codec.WindowCellCount);
		Assert.Equal(13, codec.ProgramCounterBits);
		Assert.Equal(13, codec.ProcessCountBits);
		Assert.Equal(5 * InstructionCodec.BitCount + 13 + 13, codec.InputBitCount);
	}

	[Fact]
	public void OutputBitCount_EqualsOneEncodedInstruction()
	{
		var codec = new WarriorObservationCodec(windowRadius: 1, coreSize: 100, maxProcesses: 10);

		Assert.Equal(InstructionCodec.BitCount, codec.OutputBitCount);
	}

	[Fact]
	public void ZeroRadius_ProducesSingleCellWindow()
	{
		var codec = new WarriorObservationCodec(windowRadius: 0, coreSize: 100, maxProcesses: 10);

		Assert.Equal(1, codec.WindowCellCount);
	}

	#endregion

	#region Encoding

	[Fact]
	public void EncodeObservation_ProducesInputOfDeclaredLength()
	{
		var codec = new WarriorObservationCodec(windowRadius: 1, coreSize: 100, maxProcesses: 10);
		var window = new[]
		{
			new Instruction(OpCode.DAT, AddressingMode.Immediate, 0, AddressingMode.Immediate, 0),
			new Instruction(OpCode.MOV, AddressingMode.Direct, 1, AddressingMode.Direct, 2),
			new Instruction(OpCode.JMP, AddressingMode.Direct, -1, AddressingMode.Direct, 0)
		};

		var input = codec.EncodeObservation(window, programCounter: 42, processCount: 3);

		Assert.Equal(codec.InputBitCount, input.Length);
	}

	[Fact]
	public void EncodeObservation_EmbedsWindowCellsDecodableByInstructionCodec()
	{
		var codec = new WarriorObservationCodec(windowRadius: 1, coreSize: 100, maxProcesses: 10);
		var middle = new Instruction(OpCode.SPL, AddressingMode.Indirect, 7, AddressingMode.Decrement, -9);
		var window = new[]
		{
			new Instruction(),
			middle,
			new Instruction()
		};

		var input = codec.EncodeObservation(window, programCounter: 0, processCount: 0);

		// The middle cell starts at offset 72 (after the first cell). Decode it back.
		var middleBits = input.AsSpan(InstructionCodec.BitCount, InstructionCodec.BitCount);
		var decoded = InstructionCodec.Decode(middleBits);
		Assert.Equal(middle.OpCode, decoded.OpCode);
		Assert.Equal(middle.AValue, decoded.AValue);
		Assert.Equal(middle.BValue, decoded.BValue);
	}

	[Fact]
	public void EncodeObservation_WrongWindowSize_Throws()
	{
		var codec = new WarriorObservationCodec(windowRadius: 2, coreSize: 100, maxProcesses: 10);
		var tooSmall = new[] { new Instruction(), new Instruction() };

		Assert.Throws<ArgumentException>(() => codec.EncodeObservation(tooSmall, 0, 0));
	}

	#endregion

	#region Decoding

	[Fact]
	public void DecodeInstruction_RoundTripsThroughInstructionCodec()
	{
		var codec = new WarriorObservationCodec(windowRadius: 1, coreSize: 100, maxProcesses: 10);
		var original = new Instruction(OpCode.ADD, AddressingMode.Immediate, 5, AddressingMode.Direct, -3);

		var decoded = codec.DecodeInstruction(InstructionCodec.Encode(original));

		Assert.Equal(original.OpCode, decoded.OpCode);
		Assert.Equal(original.AValue, decoded.AValue);
		Assert.Equal(original.BValue, decoded.BValue);
	}

	#endregion

	#region Validation

	[Theory]
	[InlineData(-1)]
	public void Constructor_NegativeWindowRadius_Throws(int radius)
	{
		Assert.Throws<ArgumentOutOfRangeException>(
			() => new WarriorObservationCodec(radius, coreSize: 100, maxProcesses: 10));
	}

	#endregion
}
