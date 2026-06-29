using Raylib_cs;

namespace CoreTrainer.Graphics.UI;

/// <summary>
/// Base class for all UI controls.
/// </summary>
public abstract class UIControl
{
	#region Fields

	private ColorScheme? _theme;

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

	/// <summary>
	/// The control's parent in the visual tree, or null if it is a root or detached.
	/// Set automatically when added to a <see cref="UIContainer"/>.
	/// </summary>
	public UIControl? Parent { get; internal set; }

	/// <summary>
	/// The color scheme governing this control. If not set explicitly, it is inherited
	/// from the parent chain. The root container is expected to carry the theme.
	/// </summary>
	/// <exception cref="InvalidOperationException">
	/// Thrown when neither this control nor any ancestor provides a theme.
	/// </exception>
	public ColorScheme Theme
	{
		get => _theme
			?? Parent?.Theme
			?? throw new InvalidOperationException(
				$"{GetType().Name} has no theme and no ancestor provides one. " +
				"Add it beneath a root container that carries a theme, or set Theme explicitly.");
		set => _theme = value;
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
