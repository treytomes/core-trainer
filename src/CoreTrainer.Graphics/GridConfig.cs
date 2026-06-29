namespace CoreTrainer.Graphics;

/// <summary>
/// Configuration for the memory grid visualization.
/// </summary>
public class GridConfig
{
	#region Constants

	/// <summary>
	/// Default number of columns in the grid.
	/// </summary>
	public const int DefaultColumns = 100;

	/// <summary>
	/// Default number of rows in the grid.
	/// </summary>
	public const int DefaultRows = 80;

	/// <summary>
	/// Default size of each cell in pixels.
	/// </summary>
	public const int DefaultCellSize = 8;

	/// <summary>
	/// Default horizontal offset from left edge.
	/// </summary>
	public const int DefaultOffsetX = 220;

	/// <summary>
	/// Default vertical offset from top edge.
	/// </summary>
	public const int DefaultOffsetY = 10;

	#endregion

	#region Properties

	/// <summary>
	/// Number of columns in the grid.
	/// </summary>
	public int Columns { get; init; } = DefaultColumns;

	/// <summary>
	/// Number of rows in the grid.
	/// </summary>
	public int Rows { get; init; } = DefaultRows;

	/// <summary>
	/// Size of each cell in pixels.
	/// </summary>
	public int CellSize { get; init; } = DefaultCellSize;

	/// <summary>
	/// Horizontal offset from left edge of window.
	/// </summary>
	public int OffsetX { get; init; } = DefaultOffsetX;

	/// <summary>
	/// Vertical offset from top edge of window.
	/// </summary>
	public int OffsetY { get; init; } = DefaultOffsetY;

	/// <summary>
	/// Total width of the grid in pixels.
	/// </summary>
	public int TotalWidth => Columns * CellSize;

	/// <summary>
	/// Total height of the grid in pixels.
	/// </summary>
	public int TotalHeight => Rows * CellSize;

	/// <summary>
	/// Total number of cells in the grid.
	/// </summary>
	public int TotalCells => Columns * Rows;

	#endregion

	#region Methods

	/// <summary>
	/// Converts a linear address to grid coordinates.
	/// </summary>
	/// <param name="address">Linear memory address (0-based).</param>
	/// <returns>Grid column and row.</returns>
	public (int col, int row) AddressToGrid(int address)
	{
		var col = address % Columns;
		var row = address / Columns;
		return (col, row);
	}

	/// <summary>
	/// Converts grid coordinates to a linear address.
	/// </summary>
	/// <param name="col">Grid column.</param>
	/// <param name="row">Grid row.</param>
	/// <returns>Linear memory address.</returns>
	public int GridToAddress(int col, int row)
	{
		return row * Columns + col;
	}

	/// <summary>
	/// Converts grid coordinates to screen pixel position.
	/// </summary>
	/// <param name="col">Grid column.</param>
	/// <param name="row">Grid row.</param>
	/// <returns>Screen X and Y coordinates.</returns>
	public (int x, int y) GridToScreen(int col, int row)
	{
		var x = OffsetX + col * CellSize;
		var y = OffsetY + row * CellSize;
		return (x, y);
	}

	/// <summary>
	/// Converts a linear address to screen pixel position.
	/// </summary>
	/// <param name="address">Linear memory address.</param>
	/// <returns>Screen X and Y coordinates.</returns>
	public (int x, int y) AddressToScreen(int address)
	{
		var (col, row) = AddressToGrid(address);
		return GridToScreen(col, row);
	}

	/// <summary>
	/// Converts screen pixel position to grid coordinates.
	/// </summary>
	/// <param name="screenX">Screen X coordinate.</param>
	/// <param name="screenY">Screen Y coordinate.</param>
	/// <returns>Grid column and row, or null if outside grid.</returns>
	public (int col, int row)? ScreenToGrid(int screenX, int screenY)
	{
		var relX = screenX - OffsetX;
		var relY = screenY - OffsetY;

		if (relX < 0 || relY < 0)
			return null;

		var col = relX / CellSize;
		var row = relY / CellSize;

		if (col >= Columns || row >= Rows)
			return null;

		return (col, row);
	}

	#endregion
}
