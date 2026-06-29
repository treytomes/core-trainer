namespace CoreTrainer.Graphics;

/// <summary>
/// Data structure for color scheme loaded from YAML configuration.
/// </summary>
public class ColorSchemeData
{
	/// <summary>
	/// Background colors.
	/// </summary>
	public BackgroundColors Background { get; set; } = new();

	/// <summary>
	/// Memory cell colors.
	/// </summary>
	public MemoryCellColors MemoryCells { get; set; } = new();

	/// <summary>
	/// Execution indicator colors.
	/// </summary>
	public ExecutionColors Execution { get; set; } = new();

	/// <summary>
	/// UI element colors.
	/// </summary>
	public UIColors UI { get; set; } = new();

	/// <summary>
	/// Warrior-specific colors.
	/// </summary>
	public WarriorColors Warriors { get; set; } = new();
}

public class BackgroundColors
{
	public ColorData Main { get; set; } = new(20, 20, 20, 255);
	public ColorData GridLine { get; set; } = new(40, 40, 40, 255);
}

public class MemoryCellColors
{
	public ColorData Empty { get; set; } = new(26, 26, 26, 255);
}

public class ExecutionColors
{
	public ColorData Current { get; set; } = new(255, 255, 0, 255);
	public ColorData ProcessMarker { get; set; } = new(255, 255, 255, 200);
}

public class UIColors
{
	public ColorData PanelBackground { get; set; } = new(30, 30, 30, 255);
	public ColorData TextPrimary { get; set; } = new(220, 220, 220, 255);
	public ColorData TextSecondary { get; set; } = new(140, 140, 140, 255);
	public ColorData ButtonHover { get; set; } = new(50, 50, 50, 255);
	public ColorData ButtonPressed { get; set; } = new(70, 70, 70, 255);
}

public class WarriorColors
{
	public ColorData Warrior0 { get; set; } = new(255, 68, 68, 255);
	public ColorData Warrior1 { get; set; } = new(68, 68, 255, 255);
}

/// <summary>
/// RGBA color data for YAML serialization.
/// </summary>
public class ColorData
{
	public byte R { get; set; }
	public byte G { get; set; }
	public byte B { get; set; }
	public byte A { get; set; }

	public ColorData() { }

	public ColorData(byte r, byte g, byte b, byte a)
	{
		R = r;
		G = g;
		B = b;
		A = a;
	}
}
