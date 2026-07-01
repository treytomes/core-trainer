namespace CoreTrainer.Networking.Nand;

/// <summary>
/// Configures a stagnation trigger based on a leaky-integrator (CUSUM) of the smoothed fitness
/// <i>velocity</i>: each generation accumulates how far the velocity falls short of a minimum
/// "real climb" rate, and the search counts as stalling once that accumulated shortfall crosses a
/// threshold. Designed to work under a noisy fitness, where the raw score — and even the sign of
/// the smoothed velocity — bobbles every generation while the circuit is genuinely stuck.
/// </summary>
/// <remarks>
/// <para>
/// This is the third trigger design, and it exists to fix the failure modes of the first two:
/// </para>
/// <list type="bullet">
/// <item>
/// A <b>consecutive-velocity</b> counter (count generations with velocity below a floor, reset on
/// any step above it) never armed: on a plateau the smoothed velocity bobs symmetrically around
/// zero, so it pokes above the floor roughly every other generation and the counter keeps
/// resetting before it can reach patience.
/// </item>
/// <item>
/// A <b>level high-water mark</b> (count generations since the smoothed level last made a new high)
/// armed reliably but <i>latched</i>: once the level regressed below an earlier high — which the
/// mutation ramp's own disruption can cause — it could never reclaim that high, so the counter
/// never reset and the ramp stayed engaged forever.
/// </item>
/// </list>
/// <para>
/// The CUSUM cures both. Symmetric velocity noise cancels in the sum instead of resetting a count,
/// so it is noise-robust; and the sum <b>self-drains</b> — when the level genuinely starts climbing
/// again, velocity exceeds <see cref="MinVelocity"/>, the per-step term goes negative, and the sum
/// falls back below the threshold, snapping the ramp off. There is no permanent latch: a stall can
/// always be exited by real progress. The clamp at zero keeps a long stretch of progress from
/// banking negative "credit" that would delay detecting the next stall.
/// </para>
/// </remarks>
/// <param name="LevelSmoothing">EMA factor for smoothing the raw fitness, in (0, 1].</param>
/// <param name="VelocitySmoothing">EMA factor for smoothing the velocity of that level, in (0, 1].</param>
/// <param name="MinVelocity">
/// The minimum smoothed velocity considered "real progress" (the CUSUM reference value, <c>k</c>).
/// Each generation the sum accumulates <c>MinVelocity − velocity</c>: on a plateau (velocity ≈ 0)
/// the sum grows by about this much per generation; during a genuine climb (velocity &gt; this) the
/// term is negative and the sum drains. Set between zero and a typical climb velocity.
/// </param>
/// <param name="Threshold">
/// Accumulated shortfall (the CUSUM limit, <c>h</c>) at which the search is declared stalled. Plays
/// the role of patience: roughly <c>Threshold / MinVelocity</c> stalled generations to engage.
/// </param>
public sealed record StallTrigger(
	double LevelSmoothing = 0.05,
	double VelocitySmoothing = 0.05,
	double MinVelocity = 0.0002,
	double Threshold = 0.06)
{
	/// <summary>
	/// Creates a <see cref="ProgressTracker"/> matching this trigger's smoothing factors.
	/// </summary>
	public ProgressTracker CreateTracker() => new(LevelSmoothing, VelocitySmoothing);

	/// <summary>
	/// The accumulator's upper bound: <see cref="Threshold"/> plus a one-threshold reservoir. Without
	/// a ceiling the sum runs away during a long stall (it can reach many multiples of the
	/// threshold), and then a genuine climb has to drain that entire backlog before the ramp
	/// releases — re-creating the "stuck on" behavior the self-draining design exists to avoid.
	/// Bounding it at twice the threshold keeps a small hysteresis (so a single noisy positive blip
	/// does not release the ramp) while guaranteeing the sum drains below threshold within a short,
	/// bounded number of climbing generations.
	/// </summary>
	private double Ceiling => Threshold * 2.0;

	/// <summary>
	/// Advances the CUSUM by one generation: adds the velocity's shortfall below
	/// <see cref="MinVelocity"/> to the running sum, clamped at zero so genuine progress drains the
	/// sum rather than banking negative credit, and capped at <see cref="Ceiling"/> so a long stall
	/// cannot bank a backlog that delays release once progress resumes.
	/// </summary>
	/// <param name="stallSum">The current accumulated shortfall.</param>
	/// <param name="smoothedVelocity">The current smoothed velocity from a <see cref="ProgressTracker"/>.</param>
	/// <returns>The updated accumulated shortfall.</returns>
	public double Accumulate(double stallSum, double smoothedVelocity)
		=> Math.Clamp(stallSum + (MinVelocity - smoothedVelocity), 0.0, Ceiling);

	/// <summary>
	/// True once the accumulated shortfall has reached the threshold — i.e. velocity has been below
	/// <see cref="MinVelocity"/> by enough, for long enough, to count as a stall.
	/// </summary>
	/// <param name="stallSum">The current accumulated shortfall from <see cref="Accumulate"/>.</param>
	public bool IsStalling(double stallSum) => stallSum >= Threshold;
}
