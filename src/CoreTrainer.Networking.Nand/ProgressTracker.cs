namespace CoreTrainer.Networking.Nand;

/// <summary>
/// Tracks a smoothed (rolling-average) view of a noisy progress signal and the velocity
/// (rate of change) of that smoothed value.
/// </summary>
/// <remarks>
/// <para>
/// Both quantities are exponential moving averages (EMAs), so the tracker is O(1) in time
/// and memory — no history buffer. The <see cref="Level"/> EMA smooths the raw signal; the
/// <see cref="Velocity"/> EMA smooths the per-update change in that level (a double-smoothed
/// derivative).
/// </para>
/// <para>
/// This exists to detect genuine stagnation under a <i>noisy</i> fitness (e.g. when test cases
/// are resampled each generation, so the raw score bobbles even when the circuit is truly
/// stuck). Counting per-generation improvements is fooled by that noise; the velocity of the
/// smoothed level is not — it stays positive only while the underlying trend is actually
/// climbing, and falls to ~0 on a real plateau.
/// </para>
/// </remarks>
public sealed class ProgressTracker
{
	#region Fields

	private readonly double _levelSmoothing;
	private readonly double _velocitySmoothing;

	private double _level;
	private double _velocity;
	private int _count;

	#endregion

	#region Constructors

	/// <summary>
	/// Creates a tracker with the given EMA smoothing factors.
	/// </summary>
	/// <param name="levelSmoothing">
	/// EMA factor for the level, in (0, 1]. Larger reacts faster but smooths less; a factor of
	/// α behaves roughly like a moving average over 2/α − 1 samples.
	/// </param>
	/// <param name="velocitySmoothing">EMA factor for the velocity signal, in (0, 1].</param>
	public ProgressTracker(double levelSmoothing, double velocitySmoothing)
	{
		if (levelSmoothing is <= 0.0 or > 1.0)
			throw new ArgumentOutOfRangeException(nameof(levelSmoothing), levelSmoothing, "Smoothing must be in (0, 1].");
		if (velocitySmoothing is <= 0.0 or > 1.0)
			throw new ArgumentOutOfRangeException(nameof(velocitySmoothing), velocitySmoothing, "Smoothing must be in (0, 1].");

		_levelSmoothing = levelSmoothing;
		_velocitySmoothing = velocitySmoothing;
	}

	#endregion

	#region Properties

	/// <summary>
	/// The smoothed (rolling-average) value of the signal.
	/// </summary>
	public double Level => _level;

	/// <summary>
	/// The smoothed velocity: the rate of change of <see cref="Level"/> per update. Positive
	/// while the signal trends up, near zero on a plateau, negative while trending down.
	/// </summary>
	public double Velocity => _velocity;

	#endregion

	#region Methods

	/// <summary>
	/// Feeds the latest raw signal value into the tracker, updating the level and velocity.
	/// </summary>
	/// <param name="value">The latest raw value (e.g. this generation's fitness).</param>
	public void Update(double value)
	{
		if (_count == 0)
		{
			// Seed the level with the first value so the average starts on-scale rather than
			// ramping up from zero; velocity stays 0 until there is a change to measure.
			_level = value;
			_count = 1;
			return;
		}

		var previousLevel = _level;
		_level = _levelSmoothing * value + (1 - _levelSmoothing) * _level;

		var delta = _level - previousLevel;
		_velocity = _velocitySmoothing * delta + (1 - _velocitySmoothing) * _velocity;
		_count++;
	}

	#endregion
}
