using CoreTrainer.Graphics;
using CoreTrainer.Graphics.UI;

namespace CoreTrainer.Tests;

/// <summary>
/// Tests for the UI visual tree: parent/child links and theme inheritance.
/// </summary>
public class UITreeTests
{
	[Fact]
	public void Theme_WhenInheritedFromRoot_ResolvesThroughParentChain()
	{
		var theme = new ColorScheme();
		var root = new RootContainer(800, 600, theme);

		var toolbar = root.Add(new Toolbar(0, 0, 400, 60));
		var button = toolbar.AddButton("Step");

		// The button sets no theme of its own; it should resolve the root's theme
		// through toolbar -> root.
		Assert.Same(theme, button.Theme);
	}

	[Fact]
	public void Add_SetsParentLink()
	{
		var root = new RootContainer(800, 600, new ColorScheme());
		var toolbar = root.Add(new Toolbar(0, 0, 400, 60));

		Assert.Same(root, toolbar.Parent);
	}

	[Fact]
	public void Theme_OnDetachedControl_Throws()
	{
		// A control with no theme and no parent cannot resolve a theme.
		var orphan = new Button(0, 0, 80, 40, "Orphan");

		Assert.Throws<InvalidOperationException>(() => orphan.Theme);
	}

	[Fact]
	public void Theme_ExplicitOnControl_OverridesInheritance()
	{
		var rootTheme = new ColorScheme();
		var ownTheme = new ColorScheme();
		var root = new RootContainer(800, 600, rootTheme);

		var button = root.Add(new Button(0, 0, 80, 40, "Custom"));
		button.Theme = ownTheme;

		Assert.Same(ownTheme, button.Theme);
	}

	[Fact]
	public void Remove_DetachesChildFromTree()
	{
		var root = new RootContainer(800, 600, new ColorScheme());
		var button = root.Add(new Button(0, 0, 80, 40, "Temp"));

		var removed = root.Remove(button);

		Assert.True(removed);
		Assert.Null(button.Parent);
		Assert.DoesNotContain(button, root.Children);
		// Detached control can no longer resolve a theme.
		Assert.Throws<InvalidOperationException>(() => button.Theme);
	}

	[Fact]
	public void RootContainer_WithNullTheme_Throws()
	{
		Assert.Throws<ArgumentNullException>(() => new RootContainer(800, 600, null!));
	}

	[Fact]
	public void Toolbar_PositionsChildrenHorizontally()
	{
		var root = new RootContainer(800, 600, new ColorScheme());
		var toolbar = root.Add(new Toolbar(10, 10, 400, 60) { Padding = 10, ControlSpacing = 10 });

		var first = toolbar.AddButton("A", 80);
		var second = toolbar.AddButton("B", 50);

		// First sits at X + Padding; second follows after first's width + spacing.
		Assert.Equal(20, first.X);          // 10 + 10
		Assert.Equal(110, second.X);        // 20 + 80 + 10
	}
}
