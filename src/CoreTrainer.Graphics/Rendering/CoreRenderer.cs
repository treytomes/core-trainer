using CoreTrainer.Core;
using Raylib_cs;

namespace CoreTrainer.Graphics.Rendering;

/// <summary>
/// Renders the core memory grid for visualization.
/// </summary>
public class CoreRenderer
{
	#region Fields

	private readonly GridConfig _gridConfig;
	private readonly ColorScheme _colorScheme;
	private readonly WarriorPalette _palette;

	#endregion

	#region Constructors

	public CoreRenderer(ColorScheme colorScheme, WarriorPalette palette, GridConfig? gridConfig = null)
	{
		ArgumentNullException.ThrowIfNull(colorScheme);
		ArgumentNullException.ThrowIfNull(palette);
		_colorScheme = colorScheme;
		_palette = palette;
		_gridConfig = gridConfig ?? new GridConfig();
	}

	#endregion

	#region Properties

	public GridConfig GridConfig => _gridConfig;

	#endregion

	#region Public Methods

	/// <summary>
	/// Renders the entire memory grid, coloring each cell by its current owner.
	/// </summary>
	/// <param name="mars">The MARS simulation instance.</param>
	public void Render(Mars mars)
	{
		// Draw grid cells
		for (var address = 0; address < mars.Core.Size; address++)
		{
			var (x, y) = _gridConfig.AddressToScreen(address);
			var owner = mars.GetCellOwner(address);
			var color = owner == Mars.NoOwner ? _colorScheme.Empty : _palette.GetColor(owner);

			Raylib.DrawRectangle(x, y, _gridConfig.CellSize, _gridConfig.CellSize, color);
		}

		// Draw grid lines (optional, can be disabled for performance)
		DrawGridLines();
	}

	/// <summary>
	/// Renders process position markers.
	/// </summary>
	/// <param name="mars">The MARS simulation instance.</param>
	public void RenderProcessMarkers(Mars mars)
	{
		for (var warriorIndex = 0; warriorIndex < mars.WarriorCount; warriorIndex++)
		{
			var processes = mars.GetProcessQueue(warriorIndex);

			foreach (var pc in processes)
			{
				var (x, y) = _gridConfig.AddressToScreen(pc);

				// Draw a small bright dot in the center of the cell
				var centerX = x + _gridConfig.CellSize / 2;
				var centerY = y + _gridConfig.CellSize / 2;
				var radius = Math.Max(1, _gridConfig.CellSize / 4);

				Raylib.DrawCircle(centerX, centerY, radius, _colorScheme.ProcessMarker);
			}
		}
	}

	#endregion

	#region Private Methods

	/// <summary>
	/// Draws grid lines for visual separation.
	/// </summary>
	private void DrawGridLines()
	{
		// Vertical lines
		for (var col = 0; col <= _gridConfig.Columns; col++)
		{
			var x = _gridConfig.OffsetX + col * _gridConfig.CellSize;
			var y1 = _gridConfig.OffsetY;
			var y2 = _gridConfig.OffsetY + _gridConfig.TotalHeight;
			Raylib.DrawLine(x, y1, x, y2, _colorScheme.GridLine);
		}

		// Horizontal lines
		for (var row = 0; row <= _gridConfig.Rows; row++)
		{
			var y = _gridConfig.OffsetY + row * _gridConfig.CellSize;
			var x1 = _gridConfig.OffsetX;
			var x2 = _gridConfig.OffsetX + _gridConfig.TotalWidth;
			Raylib.DrawLine(x1, y, x2, y, _colorScheme.GridLine);
		}
	}

	#endregion
}
