using Raylib_cs;

namespace CoreTrainer.Graphics.UI;

/// <summary>
/// A toolbar container that lays out child controls horizontally.
/// </summary>
public class Toolbar : UIControl
{
	#region Fields

	private readonly List<UIControl> _controls = [];

	#endregion

	#region Constructors

	public Toolbar(int x, int y, int width, int height) : base(x, y, width, height)
	{
	}

	#endregion

	#region Properties

	/// <summary>
	/// Background color of the toolbar.
	/// </summary>
	public Color BackgroundColor { get; set; } = ColorScheme.Default.PanelBackground;

	/// <summary>
	/// Border color of the toolbar.
	/// </summary>
	public Color BorderColor { get; set; } = ColorScheme.Default.GridLine;

	/// <summary>
	/// Border thickness in pixels.
	/// </summary>
	public int BorderThickness { get; set; } = 2;

	/// <summary>
	/// Padding inside the toolbar.
	/// </summary>
	public int Padding { get; set; } = 10;

	/// <summary>
	/// Spacing between controls.
	/// </summary>
	public int ControlSpacing { get; set; } = 10;

	/// <summary>
	/// Default button width.
	/// </summary>
	public int DefaultButtonWidth { get; set; } = 80;

	/// <summary>
	/// Default control height.
	/// </summary>
	public int DefaultControlHeight { get; set; } = 40;

	/// <summary>
	/// Read-only collection of controls in the toolbar, in layout order.
	/// </summary>
	public IReadOnlyList<UIControl> Controls => _controls;

	#endregion

	#region Public Methods

	/// <summary>
	/// Adds a button to the toolbar with automatic positioning.
	/// </summary>
	/// <param name="text">Button text.</param>
	/// <returns>The created button.</returns>
	public Button AddButton(string text)
		=> AddButton(text, DefaultButtonWidth);

	/// <summary>
	/// Adds a button to the toolbar with custom width.
	/// </summary>
	/// <param name="text">Button text.</param>
	/// <param name="width">Button width.</param>
	/// <returns>The created button.</returns>
	public Button AddButton(string text, int width)
		=> Add(new Button(0, 0, width, DefaultControlHeight, text));

	/// <summary>
	/// Adds a label to the toolbar with automatic positioning.
	/// </summary>
	/// <param name="text">Label text.</param>
	/// <param name="width">Label width.</param>
	/// <returns>The created label.</returns>
	public Label AddLabel(string text, int width)
		=> Add(new Label(0, 0, width, DefaultControlHeight, text));

	/// <summary>
	/// Adds an existing control to the toolbar, positioning it automatically.
	/// </summary>
	/// <param name="control">Control to add.</param>
	/// <returns>The added control.</returns>
	public T Add<T>(T control) where T : UIControl
	{
		control.X = GetNextXPosition();
		control.Y = Y + Padding;
		_controls.Add(control);
		return control;
	}

	/// <summary>
	/// Removes a control from the toolbar and repositions the remainder.
	/// </summary>
	/// <param name="control">Control to remove.</param>
	/// <returns>True if the control was removed.</returns>
	public bool Remove(UIControl control)
	{
		var removed = _controls.Remove(control);
		if (removed)
		{
			RepositionControls();
		}
		return removed;
	}

	/// <summary>
	/// Clears all controls from the toolbar.
	/// </summary>
	public void Clear()
		=> _controls.Clear();

	/// <summary>
	/// Updates all controls in the toolbar.
	/// </summary>
	public override void Update()
	{
		if (!Visible) return;

		foreach (var control in _controls)
		{
			control.Update();
		}
	}

	/// <summary>
	/// Renders the toolbar and all its controls.
	/// </summary>
	public override void Render()
	{
		if (!Visible) return;

		Raylib.DrawRectangle(X, Y, Width, Height, BackgroundColor);
		Raylib.DrawRectangleLinesEx(Bounds, BorderThickness, BorderColor);

		foreach (var control in _controls)
		{
			control.Render();
		}
	}

	#endregion

	#region Private Methods

	/// <summary>
	/// Calculates the next X position for a new control based on existing controls.
	/// </summary>
	private int GetNextXPosition()
	{
		var currentX = X + Padding;

		foreach (var control in _controls)
		{
			currentX += control.Width + ControlSpacing;
		}

		return currentX;
	}

	/// <summary>
	/// Repositions all controls in layout order after one is removed.
	/// </summary>
	private void RepositionControls()
	{
		var currentX = X + Padding;

		foreach (var control in _controls)
		{
			control.X = currentX;
			control.Y = Y + Padding;
			currentX += control.Width + ControlSpacing;
		}
	}

	#endregion
}
