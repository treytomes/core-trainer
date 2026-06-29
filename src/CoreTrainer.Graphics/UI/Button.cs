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
	/// Background color override. When null, inherits the theme's panel background.
	/// </summary>
	public Color? BackgroundColor { get; set; }

	/// <summary>
	/// Background color override when hovered. When null, inherits the theme's hover color.
	/// </summary>
	public Color? HoverColor { get; set; }

	/// <summary>
	/// Background color override when pressed. When null, inherits the theme's pressed color.
	/// </summary>
	public Color? PressedColor { get; set; }

	/// <summary>
	/// Text color override. When null, inherits the theme's primary text color.
	/// </summary>
	public Color? TextColor { get; set; }

	/// <summary>
	/// Border color override. When null, inherits the theme's grid-line color.
	/// </summary>
	public Color? BorderColor { get; set; }

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

		var theme = Theme;

		// Resolve colors from explicit overrides, falling back to the theme.
		var baseBackground = BackgroundColor ?? theme.PanelBackground;
		var borderColor = BorderColor ?? theme.GridLine;
		var textColorBase = TextColor ?? theme.TextPrimary;

		// Determine background color based on state
		Color bgColor;
		if (!Enabled)
		{
			bgColor = ColorScheme.Fade(baseBackground, 0.5f);
		}
		else if (_isPressed)
		{
			bgColor = PressedColor ?? theme.ButtonPressed;
		}
		else if (_isHovered)
		{
			bgColor = HoverColor ?? theme.ButtonHover;
		}
		else
		{
			bgColor = baseBackground;
		}

		// Draw background
		Raylib.DrawRectangle(X, Y, Width, Height, bgColor);

		// Draw border
		Raylib.DrawRectangleLinesEx(Bounds, BorderThickness, borderColor);

		// Draw text centered
		var textColor = Enabled ? textColorBase : ColorScheme.Fade(textColorBase, 0.5f);
		var textWidth = Raylib.MeasureText(Text, FontSize);
		var textX = X + (Width - textWidth) / 2;
		var textY = Y + (Height - FontSize) / 2;
		Raylib.DrawText(Text, textX, textY, FontSize, textColor);
	}

	#endregion
}
