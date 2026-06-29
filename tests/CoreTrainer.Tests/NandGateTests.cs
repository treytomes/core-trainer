using CoreTrainer.Networking.Nand;

namespace CoreTrainer.Tests;

/// <summary>
/// Tests for the fixed-width NAND gate primitive.
/// </summary>
public class NandGateTests
{
	[Fact]
	public void Apply_SingleBit_MatchesNandTruthTable()
	{
		var gate = new NandGate(1);

		Assert.Equal(1UL, gate.Apply(0, 0));
		Assert.Equal(1UL, gate.Apply(0, 1));
		Assert.Equal(1UL, gate.Apply(1, 0));
		Assert.Equal(0UL, gate.Apply(1, 1));
	}

	[Fact]
	public void Apply_MasksResultToBitWidth()
	{
		var gate = new NandGate(4);

		// ~(0 & 0) = all ones, truncated to 4 bits = 0b1111.
		Assert.Equal(0b1111UL, gate.Apply(0, 0));
	}

	[Fact]
	public void Apply_FullWidth_DoesNotOverflowShift()
	{
		// Width 64 is the boundary case: the mask must be all-ones, not a bad 1<<64 shift.
		var gate = new NandGate(64);

		Assert.Equal(ulong.MaxValue, gate.Apply(0, 0));
		Assert.Equal(0UL, gate.Apply(ulong.MaxValue, ulong.MaxValue));
	}

	[Theory]
	[InlineData(0)]
	[InlineData(65)]
	[InlineData(-1)]
	public void Constructor_RejectsBitSizeOutsideOneToSixtyFour(int bitSize)
	{
		Assert.Throws<ArgumentOutOfRangeException>(() => new NandGate(bitSize));
	}
}
