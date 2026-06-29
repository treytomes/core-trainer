using Raylib_cs;

namespace CoreTrainer.Graphics;

/// <summary>
/// Color scheme for the Core Wars visualizer.
/// Can be loaded from configuration or use defaults.
/// </summary>
public class ColorScheme
{
	#region Fields

	private readonly ColorSchemeData _data;

	#endregion

	#region Constructors

	public ColorScheme(ColorSchemeData? data = null)
	{
		_data = data ?? new ColorSchemeData();
	}

	#endregion

	#region Background Colors

	/// <summary>
	/// Main background color (dark gray).
	/// </summary>
	public Color Background => ToRaylibColor(_data.Background.Main);

	/// <summary>
	/// Grid line color (slightly lighter gray).
	/// </summary>
	public Color GridLine => ToRaylibColor(_data.Background.GridLine);

	#endregion

	#region Memory Cell Colors

	/// <summary>
	/// Empty or DAT instruction cell (very dark gray).
	/// </summary>
	public Color Empty => ToRaylibColor(_data.MemoryCells.Empty);

	/// <summary>
	/// Warrior 0 code (red).
	/// </summary>
	public Color Warrior0 => ToRaylibColor(_data.Warriors.Warrior0);

	/// <summary>
	/// Warrior 1 code (blue).
	/// </summary>
	public Color Warrior1 => ToRaylibColor(_data.Warriors.Warrior1);

	#endregion

	#region Execution Indicators

	/// <summary>
	/// Currently executing instruction (bright yellow).
	/// </summary>
	public Color CurrentExecution => ToRaylibColor(_data.Execution.Current);

	/// <summary>
	/// Process marker (white, semi-transparent).
	/// </summary>
	public Color ProcessMarker => ToRaylibColor(_data.Execution.ProcessMarker);

	#endregion

	#region UI Colors

	/// <summary>
	/// Panel background (dark gray, slightly lighter than main background).
	/// </summary>
	public Color PanelBackground => ToRaylibColor(_data.UI.PanelBackground);

	/// <summary>
	/// Primary text color (light gray).
	/// </summary>
	public Color TextPrimary => ToRaylibColor(_data.UI.TextPrimary);

	/// <summary>
	/// Secondary text color (medium gray).
	/// </summary>
	public Color TextSecondary => ToRaylibColor(_data.UI.TextSecondary);

	/// <summary>
	/// Button background when hovered.
	/// </summary>
	public Color ButtonHover => ToRaylibColor(_data.UI.ButtonHover);

	/// <summary>
	/// Button background when pressed.
	/// </summary>
	public Color ButtonPressed => ToRaylibColor(_data.UI.ButtonPressed);

	#endregion

	#region Public Methods

	/// <summary>
	/// Gets the color for a specific warrior by index.
	/// </summary>
	/// <param name="warriorIndex">Warrior index (0, 1, etc.).</param>
	/// <returns>Color for the warrior.</returns>
	public Color GetWarriorColor(int warriorIndex)
	{
		return warriorIndex switch
		{
			0 => Warrior0,
			1 => Warrior1,
			_ => Empty
		};
	}

	/// <summary>
	/// Creates a faded version of a color by adjusting alpha.
	/// </summary>
	/// <param name="color">Original color.</param>
	/// <param name="fadeAmount">Fade factor (0.0 = transparent, 1.0 = opaque).</param>
	/// <returns>Faded color.</returns>
	public static Color Fade(Color color, float fadeAmount)
	{
		fadeAmount = Math.Clamp(fadeAmount, 0.0f, 1.0f);
		return new Color(color.R, color.G, color.B, (byte)(color.A * fadeAmount));
	}

	#endregion

	#region Private Methods

	private static Color ToRaylibColor(ColorData data)
	{
		return new Color(data.R, data.G, data.B, data.A);
	}

	#endregion
}
