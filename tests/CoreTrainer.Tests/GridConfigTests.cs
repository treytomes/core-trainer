using CoreTrainer.Graphics;

namespace CoreTrainer.Tests;

/// <summary>
/// Tests for grid layout configuration.
/// </summary>
public class GridConfigTests
{
	[Fact]
	public void ForCoreSize_GridHoldsEveryAddressWithinBounds()
	{
		// A non-default core size that does not divide evenly by the column count.
		const int coreSize = 8123;
		var grid = GridConfig.ForCoreSize(coreSize);

		// Every address must map to a row inside the grid (the bug this guards against
		// rendered out-of-range addresses on top of the toolbar/status area).
		var maxRow = 0;
		for (var address = 0; address < coreSize; address++)
		{
			var (_, row) = grid.AddressToGrid(address);
			maxRow = Math.Max(maxRow, row);
		}

		Assert.True(maxRow < grid.Rows, $"Max row {maxRow} must be less than grid rows {grid.Rows}");
		Assert.True(grid.TotalCells >= coreSize, "Grid must contain at least as many cells as the core");
	}

	[Fact]
	public void ForCoreSize_RoundsUpRowsForPartialFinalRow()
	{
		// 8001 cells over 100 columns needs 81 rows (80 full + 1 partial).
		var grid = GridConfig.ForCoreSize(8001, columns: 100);

		Assert.Equal(81, grid.Rows);
	}

	[Fact]
	public void ForCoreSize_WithNonPositiveCoreSize_Throws()
	{
		Assert.Throws<ArgumentOutOfRangeException>(() => GridConfig.ForCoreSize(0));
	}
}
