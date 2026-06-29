using Raylib_cs;

namespace CoreTrainer.Graphics.UI;

/// <summary>
/// A container that lays out its child controls horizontally.
/// </summary>
public class Toolbar : UIContainer
{
	#region Constructors

	public Toolbar(int x, int y, int width, int height) : base(x, y, width, height)
	{
	}

	#endregion

	#region Properties

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
	/// Renders the toolbar chrome (background and border), then its child controls.
	/// </summary>
	public override void Render()
	{
		if (!Visible) return;

		Raylib.DrawRectangle(X, Y, Width, Height, Theme.PanelBackground);
		Raylib.DrawRectangleLinesEx(Bounds, BorderThickness, Theme.GridLine);

		base.Render();
	}

	#endregion

	#region Protected Methods

	/// <summary>
	/// Positions a newly added control at the end of the horizontal layout.
	/// </summary>
	protected override void OnChildAdded(UIControl child)
	{
		// The child is already in Children; place it after the controls before it.
		var currentX = X + Padding;
		foreach (var control in Children)
		{
			if (ReferenceEquals(control, child))
				break;
			currentX += control.Width + ControlSpacing;
		}

		child.X = currentX;
		child.Y = Y + Padding;
	}

	/// <summary>
	/// Reflows all controls in layout order after the child collection changes.
	/// </summary>
	protected override void OnChildrenChanged()
	{
		var currentX = X + Padding;

		foreach (var control in Children)
		{
			control.X = currentX;
			control.Y = Y + Padding;
			currentX += control.Width + ControlSpacing;
		}
	}

	#endregion
}
