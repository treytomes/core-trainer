using CoreTrainer.Core;

namespace CoreTrainer.Tests;

/// <summary>
/// Tests for warrior starting-position placement.
/// </summary>
public class WarriorPlacementTests
{
	[Fact]
	public void GetSeparatedPosition_ResultIsAtLeastMinSeparationAway()
	{
		const int coreSize = 8000;
		const int reference = 1000;
		const int minSeparation = 100;
		var random = new Random(12345);

		for (var i = 0; i < 1000; i++)
		{
			var position = WarriorPlacement.GetSeparatedPosition(coreSize, reference, minSeparation, random);

			var linear = Math.Abs(reference - position);
			var circular = Math.Min(linear, coreSize - linear);
			Assert.True(circular >= minSeparation, $"Position {position} was only {circular} from {reference}");
		}
	}

	[Fact]
	public void GetSeparatedPosition_ResultIsWithinCoreBounds()
	{
		const int coreSize = 8000;
		var random = new Random(999);

		var position = WarriorPlacement.GetSeparatedPosition(coreSize, 0, 100, random);

		Assert.InRange(position, 0, coreSize - 1);
	}

	[Fact]
	public void GetSeparatedPosition_WhenSeparationExceedsHalfCore_DoesNotHang()
	{
		// Requesting more separation than the core can provide must not loop forever;
		// it is clamped to the maximum achievable (half the core).
		const int coreSize = 100;
		var random = new Random(7);

		var position = WarriorPlacement.GetSeparatedPosition(coreSize, 0, minSeparation: 9999, random);

		Assert.InRange(position, 0, coreSize - 1);
	}

	[Fact]
	public void GetSeparatedPosition_WithNonPositiveCoreSize_Throws()
	{
		Assert.Throws<ArgumentOutOfRangeException>(
			() => WarriorPlacement.GetSeparatedPosition(0, 0, 10, new Random(1)));
	}
}
