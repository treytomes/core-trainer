using CoreTrainer.Networking.Nand;

namespace CoreTrainer.Tests;

/// <summary>
/// Tests for the smoothed level/velocity tracker and the velocity stagnation trigger.
/// </summary>
public class ProgressTrackerTests
{
	[Fact]
	public void Level_SeedsToFirstValue_VelocityStartsAtZero()
	{
		var tracker = new ProgressTracker(levelSmoothing: 0.5, velocitySmoothing: 0.5);
		tracker.Update(0.8);

		Assert.Equal(0.8, tracker.Level, precision: 10);
		Assert.Equal(0.0, tracker.Velocity, precision: 10);
	}

	[Fact]
	public void Velocity_PositiveWhileSignalClimbs()
	{
		var tracker = new ProgressTracker(levelSmoothing: 0.5, velocitySmoothing: 0.5);
		foreach (var v in new[] { 0.5, 0.6, 0.7, 0.8, 0.9 })
			tracker.Update(v);

		Assert.True(tracker.Velocity > 0, $"Expected positive velocity, got {tracker.Velocity}");
	}

	[Fact]
	public void Velocity_DecaysTowardZeroOnAPlateau()
	{
		var tracker = new ProgressTracker(levelSmoothing: 0.3, velocitySmoothing: 0.3);
		// Climb, then flatten.
		foreach (var v in new[] { 0.2, 0.4, 0.6, 0.8 })
			tracker.Update(v);
		var climbingVelocity = tracker.Velocity;

		// Hold flat for a while.
		for (var i = 0; i < 50; i++)
			tracker.Update(0.8);

		Assert.True(tracker.Velocity < climbingVelocity);
		Assert.True(Math.Abs(tracker.Velocity) < 1e-4, $"Velocity should be ~0 on a plateau, got {tracker.Velocity}");
	}

	[Fact]
	public void Velocity_IsRobustToNoiseAroundAFlatMean()
	{
		// A noisy-but-trendless signal: the smoothed velocity should hover near zero rather
		// than tracking the per-sample jitter (the whole point under dynamic sampling).
		var tracker = new ProgressTracker(levelSmoothing: 0.05, velocitySmoothing: 0.05);
		var rng = new Random(1);
		for (var i = 0; i < 500; i++)
			tracker.Update(0.8 + (rng.NextDouble() - 0.5) * 0.1); // mean 0.8, +/-0.05 noise

		Assert.True(Math.Abs(tracker.Velocity) < 1e-3, $"Velocity under flat noise should be ~0, got {tracker.Velocity}");
	}

	[Fact]
	public void StallTrigger_Arms_OnAPersistentPlateau()
	{
		// Velocity sits below the real-climb rate (here ~0): the CUSUM should accumulate and cross
		// the threshold after roughly Threshold / MinVelocity generations.
		var trigger = new StallTrigger(MinVelocity: 0.001, Threshold: 0.05);
		var sum = 0.0;

		Assert.False(trigger.IsStalling(sum));
		for (var i = 0; i < 60; i++)
			sum = trigger.Accumulate(sum, smoothedVelocity: 0.0);

		Assert.True(trigger.IsStalling(sum), $"Expected stall after a long plateau, sum={sum}");
	}

	[Fact]
	public void StallTrigger_SelfDrains_WhenProgressResumes()
	{
		// This is the anti-latch property: once armed on a plateau, a genuine climb (velocity well
		// above the reference rate) must drain the sum back below threshold and release the ramp —
		// unlike a level high-water mark, which could never reclaim a regressed high.
		var trigger = new StallTrigger(MinVelocity: 0.001, Threshold: 0.05);
		var sum = 0.0;
		for (var i = 0; i < 60; i++)
			sum = trigger.Accumulate(sum, 0.0);
		Assert.True(trigger.IsStalling(sum));

		// Real progress resumes.
		for (var i = 0; i < 60; i++)
			sum = trigger.Accumulate(sum, smoothedVelocity: 0.01);

		Assert.False(trigger.IsStalling(sum), $"Expected the CUSUM to drain once climbing, sum={sum}");
		Assert.Equal(0.0, sum, precision: 10); // clamped at zero, not banking negative credit
	}

	[Fact]
	public void StallTrigger_Accumulator_IsCappedSoItDrainsPromptly()
	{
		// A long stall must not bank an unbounded backlog: without a ceiling the sum reaches many
		// multiples of the threshold, and then a real climb has to drain all of it before the ramp
		// releases — re-creating the stuck-on behavior. The cap bounds how much must drain.
		var trigger = new StallTrigger(MinVelocity: 0.001, Threshold: 0.05);
		var sum = 0.0;

		// Stall for far longer than it takes to arm.
		for (var i = 0; i < 10_000; i++)
			sum = trigger.Accumulate(sum, smoothedVelocity: 0.0);

		// The sum is bounded (here at twice the threshold), not runaway.
		Assert.True(sum <= 0.05 * 2.0 + 1e-9, $"Accumulator should be capped, got {sum}");

		// A modest climb then drains it below threshold in a bounded number of generations.
		for (var i = 0; i < 200; i++)
			sum = trigger.Accumulate(sum, smoothedVelocity: 0.01);
		Assert.False(trigger.IsStalling(sum));
	}

	[Fact]
	public void StallTrigger_StaysQuiet_DuringAGenuineClimb()
	{
		// Velocity above the reference rate every step: the sum stays clamped at zero and never arms.
		var trigger = new StallTrigger(MinVelocity: 0.001, Threshold: 0.05);
		var sum = 0.0;

		for (var i = 0; i < 200; i++)
			sum = trigger.Accumulate(sum, smoothedVelocity: 0.005);

		Assert.Equal(0.0, sum, precision: 10);
		Assert.False(trigger.IsStalling(sum));
	}

	[Fact]
	public void Constructor_RejectsSmoothingOutOfRange()
	{
		Assert.Throws<ArgumentOutOfRangeException>(() => new ProgressTracker(0.0, 0.5));
		Assert.Throws<ArgumentOutOfRangeException>(() => new ProgressTracker(0.5, 1.5));
	}
}
