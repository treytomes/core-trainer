namespace CoreTrainer.Graphics.UI;

/// <summary>
/// The root of the UI visual tree, sized to the display. It carries the theme that
/// all descendants inherit. Background clearing and any non-UI layers (such as the
/// memory grid) are the caller's responsibility, since they sit below the UI in
/// z-order and outside the control tree.
/// </summary>
public class RootContainer : UIContainer
{
	#region Constructors

	/// <summary>
	/// Creates a root container covering the given area with the given theme.
	/// </summary>
	/// <param name="width">Display width in pixels.</param>
	/// <param name="height">Display height in pixels.</param>
	/// <param name="theme">The color scheme inherited by all descendants.</param>
	public RootContainer(int width, int height, ColorScheme theme) : base(0, 0, width, height)
	{
		ArgumentNullException.ThrowIfNull(theme);
		Theme = theme;
	}

	#endregion
}
