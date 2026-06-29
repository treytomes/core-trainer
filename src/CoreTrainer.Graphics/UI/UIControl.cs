using Raylib_cs;

namespace CoreTrainer.Graphics.UI;

/// <summary>
/// Base class for all UI controls.
/// </summary>
public abstract class UIControl
{
	#region Properties

	/// <summary>
	/// X position of the control.
	/// </summary>
	public int X { get; set; }

	/// <summary>
	/// Y position of the control.
	/// </summary>
	public int Y { get; set; }

	/// <summary>
	/// Width of the control.
	/// </summary>
	public int Width { get; set; }

	/// <summary>
	/// Height of the control.
	/// </summary>
	public int Height { get; set; }

	/// <summary>
	/// Whether the control is visible.
	/// </summary>
	public bool Visible { get; set; } = true;

	/// <summary>
	/// Gets the bounds of the control as a rectangle.
	/// </summary>
	public Rectangle Bounds => new(X, Y, Width, Height);

	#endregion

	#region Constructors

	protected UIControl(int x, int y, int width, int height)
	{
		X = x;
		Y = y;
		Width = width;
		Height = height;
	}

	#endregion

	#region Public Methods

	/// <summary>
	/// Updates the control state (called each frame).
	/// </summary>
	public virtual void Update()
	{
		// Base implementation does nothing
	}

	/// <summary>
	/// Renders the control.
	/// </summary>
	public abstract void Render();

	#endregion
}
