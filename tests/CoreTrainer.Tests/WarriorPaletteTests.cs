using CoreTrainer.Graphics;
using Raylib_cs;

namespace CoreTrainer.Tests;

/// <summary>
/// Tests for warrior color generation.
/// </summary>
public class WarriorPaletteTests
{
	[Fact]
	public void GetColor_ForEachWarrior_ReturnsDistinctColors()
	{
		var palette = new WarriorPalette(8);

		var seen = new HashSet<(byte, byte, byte)>();
		for (var i = 0; i < palette.Count; i++)
		{
			var c = palette.GetColor(i);
			Assert.True(seen.Add((c.R, c.G, c.B)), $"Color at index {i} was not distinct");
		}
	}

	[Fact]
	public void GetColor_GeneratesLegibleColors_NotNearBlack()
	{
		// HSV with fixed value keeps every color bright enough to read on a dark background.
		var palette = new WarriorPalette(12);

		for (var i = 0; i < palette.Count; i++)
		{
			var c = palette.GetColor(i);
			var brightest = Math.Max(c.R, Math.Max(c.G, c.B));
			Assert.True(brightest > 150, $"Color at index {i} ({c.R},{c.G},{c.B}) is too dark");
		}
	}

	[Fact]
	public void GetColor_IndexBeyondCount_WrapsAround()
	{
		var palette = new WarriorPalette(3);

		var first = palette.GetColor(0);
		var wrapped = palette.GetColor(3);

		Assert.Equal((first.R, first.G, first.B), (wrapped.R, wrapped.G, wrapped.B));
	}

	[Fact]
	public void GetColor_OnEmptyPalette_Throws()
	{
		var palette = new WarriorPalette(0);

		Assert.Throws<InvalidOperationException>(() => palette.GetColor(0));
	}

	[Fact]
	public void GetColor_NegativeIndex_Throws()
	{
		var palette = new WarriorPalette(3);

		Assert.Throws<ArgumentOutOfRangeException>(() => palette.GetColor(-1));
	}
}
