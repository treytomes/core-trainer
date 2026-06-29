namespace CoreTrainer.Console;

/// <summary>
/// Configuration options for the Raylib window.
/// </summary>
public class WindowOptions
{
	/// <summary>
	/// Window width in pixels.
	/// </summary>
	public int Width { get; set; } = 1200;

	/// <summary>
	/// Window height in pixels.
	/// </summary>
	public int Height { get; set; } = 800;

	/// <summary>
	/// Window title.
	/// </summary>
	public string Title { get; set; } = "Core Wars Visualizer";

	/// <summary>
	/// Target frames per second.
	/// </summary>
	public int TargetFPS { get; set; } = 60;
}
