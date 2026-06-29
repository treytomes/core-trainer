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

	[Fact]
	public void GetSeparatedPositions_ReturnsRequestedCount()
	{
		var positions = WarriorPlacement.GetSeparatedPositions(8000, 6, 100, new Random(42));

		Assert.Equal(6, positions.Length);
		Assert.All(positions, p => Assert.InRange(p, 0, 7999));
	}

	[Fact]
	public void GetSeparatedPositions_WhenRoomy_KeepsPositionsSeparated()
	{
		const int coreSize = 8000;
		const int minSeparation = 100;
		var positions = WarriorPlacement.GetSeparatedPositions(coreSize, 4, minSeparation, new Random(7));

		// With only 4 warriors in an 8000-cell core, the constraint is easily satisfiable.
		for (var i = 0; i < positions.Length; i++)
		{
			for (var j = i + 1; j < positions.Length; j++)
			{
				var linear = Math.Abs(positions[i] - positions[j]);
				var circular = Math.Min(linear, coreSize - linear);
				Assert.True(circular >= minSeparation,
					$"Positions {positions[i]} and {positions[j]} were only {circular} apart");
			}
		}
	}

	[Fact]
	public void GetSeparatedPositions_WhenOvercrowded_DoesNotHangAndReturnsCount()
	{
		// Far more warriors than the separation allows: must terminate, best-effort.
		var positions = WarriorPlacement.GetSeparatedPositions(100, 50, 10, new Random(1));

		Assert.Equal(50, positions.Length);
	}

	[Fact]
	public void GetSeparatedPositions_WithZeroCount_ReturnsEmpty()
	{
		var positions = WarriorPlacement.GetSeparatedPositions(8000, 0, 100, new Random(1));

		Assert.Empty(positions);
	}
}
