using System.Reactive;
using System.Reactive.Linq;
using System.Reactive.Subjects;
using Raylib_cs;

namespace CoreTrainer.Graphics.UI;

/// <summary>
/// A clickable button UI element.
/// </summary>
public class Button : UIControl
{
	#region Fields

	private bool _isHovered;
	private bool _isPressed;
	private readonly Subject<Unit> _clickSubject = new();

	#endregion

	#region Constructors

	public Button(int x, int y, int width, int height, string text) : base(x, y, width, height)
	{
		Text = text;
	}

	#endregion

	#region Properties

	/// <summary>
	/// Text displayed on the button.
	/// </summary>
	public string Text { get; set; }

	/// <summary>
	/// Font size for button text.
	/// </summary>
	public int FontSize { get; set; } = 20;

	/// <summary>
	/// Background color when button is in normal state.
	/// </summary>
	public Color BackgroundColor { get; set; } = ColorScheme.Default.PanelBackground;

	/// <summary>
	/// Background color when button is hovered.
	/// </summary>
	public Color HoverColor { get; set; } = new Color(50, 50, 50, 255);

	/// <summary>
	/// Background color when button is pressed.
	/// </summary>
	public Color PressedColor { get; set; } = new Color(70, 70, 70, 255);

	/// <summary>
	/// Text color.
	/// </summary>
	public Color TextColor { get; set; } = ColorScheme.Default.TextPrimary;

	/// <summary>
	/// Border color.
	/// </summary>
	public Color BorderColor { get; set; } = ColorScheme.Default.GridLine;

	/// <summary>
	/// Border thickness in pixels.
	/// </summary>
	public int BorderThickness { get; set; } = 2;

	/// <summary>
	/// Whether the button is currently enabled.
	/// </summary>
	public bool Enabled { get; set; } = true;

	/// <summary>
	/// Observable that emits when the button is clicked.
	/// </summary>
	public IObservable<Unit> Clicked => _clickSubject.AsObservable();

	#endregion

	#region Public Methods

	/// <summary>
	/// Updates the button state based on mouse input, emitting a click event when clicked.
	/// </summary>
	public override void Update()
	{
		if (!Enabled || !Visible)
		{
			_isHovered = false;
			_isPressed = false;
			return;
		}

		var mousePos = Raylib.GetMousePosition();
		_isHovered = Raylib.CheckCollisionPointRec(mousePos, Bounds);

		var wasPressed = _isPressed;
		_isPressed = _isHovered && Raylib.IsMouseButtonDown(MouseButton.Left);

		// Button was clicked if it was pressed last frame and released this frame while hovered
		var clicked = wasPressed && !_isPressed && _isHovered;

		// Emit click event
		if (clicked)
		{
			_clickSubject.OnNext(Unit.Default);
		}
	}

	/// <summary>
	/// Renders the button.
	/// </summary>
	public override void Render()
	{
		if (!Visible) return;

		// Determine background color based on state
		var bgColor = BackgroundColor;
		if (!Enabled)
		{
			bgColor = ColorScheme.Fade(BackgroundColor, 0.5f);
		}
		else if (_isPressed)
		{
			bgColor = PressedColor;
		}
		else if (_isHovered)
		{
			bgColor = HoverColor;
		}

		// Draw background
		Raylib.DrawRectangle(X, Y, Width, Height, bgColor);

		// Draw border
		Raylib.DrawRectangleLinesEx(Bounds, BorderThickness, BorderColor);

		// Draw text centered
		var textColor = Enabled ? TextColor : ColorScheme.Fade(TextColor, 0.5f);
		var textWidth = Raylib.MeasureText(Text, FontSize);
		var textX = X + (Width - textWidth) / 2;
		var textY = Y + (Height - FontSize) / 2;
		Raylib.DrawText(Text, textX, textY, FontSize, textColor);
	}

	#endregion
}
