using Raylib_cs;

namespace CoreTrainer.Graphics.UI;

/// <summary>
/// A non-interactive text label UI element.
/// </summary>
public class Label : UIControl
{
	#region Constructors

	public Label(int x, int y, int width, int height, string text) : base(x, y, width, height)
	{
		Text = text;
	}

	#endregion

	#region Properties

	/// <summary>
	/// Text displayed in the label.
	/// </summary>
	public string Text { get; set; }

	/// <summary>
	/// Font size for label text.
	/// </summary>
	public int FontSize { get; set; } = 20;

	/// <summary>
	/// Background color (optional, transparent by default).
	/// </summary>
	public Color? BackgroundColor { get; set; }

	/// <summary>
	/// Text color override. When null, inherits the theme's primary text color.
	/// </summary>
	public Color? TextColor { get; set; }

	/// <summary>
	/// Border color (optional, no border by default).
	/// </summary>
	public Color? BorderColor { get; set; }

	/// <summary>
	/// Border thickness in pixels (only used if BorderColor is set).
	/// </summary>
	public int BorderThickness { get; set; } = 2;

	/// <summary>
	/// Text alignment within the label.
	/// </summary>
	public TextAlignment Alignment { get; set; } = TextAlignment.Center;

	#endregion

	#region Public Methods

	/// <summary>
	/// Renders the label.
	/// </summary>
	public override void Render()
	{
		if (!Visible) return;

		// Draw background if specified
		if (BackgroundColor.HasValue)
		{
			Raylib.DrawRectangle(X, Y, Width, Height, BackgroundColor.Value);
		}

		// Draw border if specified
		if (BorderColor.HasValue)
		{
			Raylib.DrawRectangleLinesEx(Bounds, BorderThickness, BorderColor.Value);
		}

		// Calculate text position based on alignment
		var textWidth = Raylib.MeasureText(Text, FontSize);
		var textX = Alignment switch
		{
			TextAlignment.Left => X + 5,
			TextAlignment.Right => X + Width - textWidth - 5,
			TextAlignment.Center => X + (Width - textWidth) / 2,
			_ => X + (Width - textWidth) / 2
		};
		var textY = Y + (Height - FontSize) / 2;

		// Draw text
		Raylib.DrawText(Text, textX, textY, FontSize, TextColor ?? Theme.TextPrimary);
	}

	#endregion
}

/// <summary>
/// Text alignment options for labels.
/// </summary>
public enum TextAlignment
{
	/// <summary>
	/// Align text to the left.
	/// </summary>
	Left,

	/// <summary>
	/// Center text horizontally.
	/// </summary>
	Center,

	/// <summary>
	/// Align text to the right.
	/// </summary>
	Right
}
