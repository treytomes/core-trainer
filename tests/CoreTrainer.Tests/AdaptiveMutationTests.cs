using CoreTrainer.Networking.Nand;

namespace CoreTrainer.Tests;

/// <summary>
/// Tests for the stagnation-triggered mutation-rate ramp.
/// </summary>
public class AdaptiveMutationTests
{
	[Fact]
	public void EffectiveRate_WithinPatience_ReturnsBaseRate()
	{
		var adaptive = new AdaptiveMutation(Patience: 500, Growth: 0.001);

		Assert.Equal(0.03, adaptive.EffectiveRate(baseRate: 0.03, stagnationDuration: 0));
		Assert.Equal(0.03, adaptive.EffectiveRate(0.03, 499));
	}

	[Fact]
	public void EffectiveRate_PastPatience_RampsLinearly()
	{
		var adaptive = new AdaptiveMutation(Patience: 500, Growth: 0.001);

		// At exactly patience, still base (zero generations "over").
		Assert.Equal(0.03, adaptive.EffectiveRate(0.03, 500), precision: 10);
		// 100 generations past patience: base + 100 * growth.
		Assert.Equal(0.03 + 100 * 0.001, adaptive.EffectiveRate(0.03, 600), precision: 10);
	}

	[Fact]
	public void EffectiveRate_IsCappedAtMaxRate()
	{
		var adaptive = new AdaptiveMutation(Patience: 10, Growth: 0.1, MaxRate: 0.5);

		// Far past patience the linear ramp would exceed the cap; it must clamp.
		Assert.Equal(0.5, adaptive.EffectiveRate(0.03, stagnationDuration: 100_000));
	}

	[Fact]
	public void EffectiveRate_Sawtooth_ResetsToBaseAfterReachingCap()
	{
		// base 0.0, growth 0.1, cap 0.5 -> one climb spans 5 stagnant generations.
		var adaptive = new AdaptiveMutation(Patience: 0, Growth: 0.1, MaxRate: 0.5, Sawtooth: true);

		// Within the first climb: 0.0, 0.1, 0.2, 0.3, 0.4 ...
		Assert.Equal(0.0, adaptive.EffectiveRate(0.0, 0), precision: 10);
		Assert.Equal(0.3, adaptive.EffectiveRate(0.0, 3), precision: 10);
		// ... then it wraps back toward base rather than pinning at the cap.
		Assert.Equal(0.0, adaptive.EffectiveRate(0.0, 5), precision: 10);
		Assert.Equal(0.3, adaptive.EffectiveRate(0.0, 8), precision: 10);
	}

	[Fact]
	public void UsesStallTrigger_TrueOnlyWhenStallSet()
	{
		Assert.False(new AdaptiveMutation(Patience: 100, Growth: 0.001).UsesStallTrigger);
		Assert.True(new AdaptiveMutation(Patience: 0, Growth: 0.001, Stall: new StallTrigger()).UsesStallTrigger);
	}
}
