namespace CoreTrainer.Networking.Nand;

/// <summary>
/// Configures stagnation-triggered mutation-rate scaling for <see cref="CgpEvolution"/>.
/// </summary>
/// <remarks>
/// <para>
/// When the best fitness has not strictly improved for <see cref="Patience"/> generations,
/// the mutation rate begins climbing linearly — by <see cref="Growth"/> per additional
/// stagnant generation — up to <see cref="MaxRate"/>. The instant fitness strictly improves,
/// the rate snaps back to the base rate and the stagnation counter resets.
/// </para>
/// <para>
/// This is a step-size sweep that escapes elitist plateaus: as the rate climbs, offspring
/// jump further from the stuck parent until one lands on a better region. Because selection
/// stays elitist (an offspring is only kept if it is no worse), an over-large rate merely
/// wastes generations — it cannot damage the incumbent — so ramping is safe to overshoot.
/// </para>
/// <para>
/// With <see cref="Sawtooth"/> enabled, reaching <see cref="MaxRate"/> resets the rate to base
/// and the climb restarts, sweeping the step size repeatedly rather than camping at the ceiling.
/// This prevents the rate from dead-ending at the cap with no way to back off when the cap
/// itself is not the productive step size.
/// </para>
/// <para>
/// Stagnation is measured by fitness <i>improvement</i>, not genome change: neutral drift
/// moves the genome at equal fitness every generation, and that flat fitness is exactly the
/// stall this mechanism is meant to break.
/// </para>
/// </remarks>
/// <para>
/// <b>Two ways to detect stagnation.</b> By default the trigger is a fixed
/// <see cref="Patience"/> counter: ramp after that many consecutive generations without a
/// strict improvement. Supplying a <see cref="Stall"/> trigger switches to a CUSUM test on the
/// smoothed fitness velocity instead: ramp when the velocity's accumulated shortfall below a
/// real-climb rate crosses a threshold. This survives a <i>noisy</i> fitness (e.g. resampled test
/// cases) where the raw score bobbles every generation — the per-generation improvement counter is
/// fooled by that noise — and, unlike a level high-water mark, it cannot latch the ramp on
/// permanently, because the accumulated sum drains as soon as real progress resumes.
/// </para>
/// </remarks>
/// <param name="Patience">
/// Generations the stagnation condition must hold before ramping begins. With the fixed trigger,
/// that condition is "no strict improvement". Ignored by the <see cref="Stall"/> trigger, which
/// has its own threshold; the ramp position there counts from when the CUSUM first armed.
/// </param>
/// <param name="Growth">Amount added to the mutation rate per stagnant generation past <see cref="Patience"/> (&gt; 0).</param>
/// <param name="MaxRate">Ceiling for the scaled mutation rate, in (0, 1].</param>
/// <param name="Sawtooth">When true, the ramp resets to base on reaching <see cref="MaxRate"/> and climbs again.</param>
/// <param name="Stall">
/// Optional CUSUM velocity stagnation trigger. When supplied, replaces the improvement-counter
/// trigger; when null (default), the fixed trigger is used.
/// </param>
public sealed record AdaptiveMutation(
	int Patience,
	double Growth,
	double MaxRate = 1.0,
	bool Sawtooth = false,
	StallTrigger? Stall = null)
{
	/// <summary>
	/// Whether the CUSUM velocity stall trigger is in use (rather than the improvement counter).
	/// </summary>
	public bool UsesStallTrigger => Stall is not null;

	/// <summary>
	/// Computes the effective mutation rate given the base rate and how long the search has been
	/// stagnating (measured by whichever trigger is active — see remarks).
	/// </summary>
	/// <param name="baseRate">The configured base mutation rate.</param>
	/// <param name="stagnationDuration">Generations spent stagnating, as tracked by the caller.</param>
	/// <returns>
	/// The base rate while within patience. Past patience, a linearly-scaled rate capped at
	/// <see cref="MaxRate"/> — or, with <see cref="Sawtooth"/>, a rate that climbs from base to
	/// the cap and repeats.
	/// </returns>
	public double EffectiveRate(double baseRate, int stagnationDuration)
	{
		if (stagnationDuration < Patience)
			return baseRate;

		var over = stagnationDuration - Patience;

		if (!Sawtooth)
			return Math.Min(MaxRate, baseRate + Growth * over);

		// Sawtooth: climb from base toward the cap, then reset to base and climb again.
		// The span is how many stagnant generations one climb takes; modulo gives the
		// position within the current climb.
		var span = (MaxRate - baseRate) / Growth;
		if (span <= 0)
			return baseRate;

		var positionInClimb = over % (long)Math.Ceiling(span);
		return Math.Min(MaxRate, baseRate + Growth * positionInClimb);
	}
}
