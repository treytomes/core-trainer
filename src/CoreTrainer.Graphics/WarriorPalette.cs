using Raylib_cs;

namespace CoreTrainer.Graphics;

/// <summary>
/// Generates a distinct, legible color per warrior. Colors are produced in HSV with
/// fixed saturation and value so they stay bright enough to read against the dark
/// arena background (unlike raw random RGB, which can land on near-black or muddy
/// tones). Hues are spread evenly around the wheel for maximum separation.
/// </summary>
public class WarriorPalette
{
	#region Fields

	private readonly Color[] _colors;

	#endregion

	#region Constructors

	/// <summary>
	/// Builds a palette of <paramref name="count"/> colors.
	/// </summary>
	/// <param name="count">Number of warrior colors to generate.</param>
	/// <param name="random">
	/// Optional random source used to offset the starting hue so successive battles
	/// differ. When null, the palette is deterministic (hue starts at 0).
	/// </param>
	public WarriorPalette(int count, Random? random = null)
	{
		if (count < 0)
			throw new ArgumentOutOfRangeException(nameof(count), "Count cannot be negative");

		_colors = new Color[count];
		if (count == 0)
			return;

		// Vary saturation/value slightly per index so adjacent hues are extra
		// distinguishable, while staying in a legible range.
		const float saturation = 0.70f;
		const float value = 0.95f;

		var hueOffset = random is null ? 0f : (float)(random.NextDouble() * 360.0);

		for (var i = 0; i < count; i++)
		{
			var hue = (hueOffset + i * (360f / count)) % 360f;
			_colors[i] = FromHsv(hue, saturation, value);
		}
	}

	#endregion

	#region Properties

	/// <summary>
	/// Number of colors in the palette.
	/// </summary>
	public int Count => _colors.Length;

	#endregion

	#region Public Methods

	/// <summary>
	/// Gets the color for a warrior index. Indices outside the palette wrap around,
	/// so the palette is safe to query for any non-negative index.
	/// </summary>
	/// <param name="warriorIndex">Zero-based warrior index.</param>
	/// <returns>The warrior's color.</returns>
	public Color GetColor(int warriorIndex)
	{
		if (_colors.Length == 0)
			throw new InvalidOperationException("The palette is empty.");
		if (warriorIndex < 0)
			throw new ArgumentOutOfRangeException(nameof(warriorIndex), "Index cannot be negative");

		return _colors[warriorIndex % _colors.Length];
	}

	#endregion

	#region Private Methods

	/// <summary>
	/// Converts an HSV color to RGB. Hue is in [0, 360), saturation and value in [0, 1].
	/// </summary>
	private static Color FromHsv(float hue, float saturation, float value)
	{
		var c = value * saturation;
		var x = c * (1 - Math.Abs((hue / 60f) % 2 - 1));
		var m = value - c;

		var (r, g, b) = hue switch
		{
			< 60f => (c, x, 0f),
			< 120f => (x, c, 0f),
			< 180f => (0f, c, x),
			< 240f => (0f, x, c),
			< 300f => (x, 0f, c),
			_ => (c, 0f, x)
		};

		return new Color(
			(byte)Math.Round((r + m) * 255),
			(byte)Math.Round((g + m) * 255),
			(byte)Math.Round((b + m) * 255),
			(byte)255);
	}

	#endregion
}
