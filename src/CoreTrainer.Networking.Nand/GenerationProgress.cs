namespace CoreTrainer.Networking.Nand;

/// <summary>
/// A snapshot reported after each generation of a <see cref="CgpEvolution"/> run, for
/// progress reporting.
/// </summary>
/// <param name="Generation">Zero-based index of the generation just completed.</param>
/// <param name="Fitness">The current best (parent) fitness after this generation.</param>
/// <param name="Best">The current best (parent) genome after this generation.</param>
/// <param name="MutationRate">
/// The effective per-gene mutation rate used to produce this generation's offspring. With
/// adaptive mutation this rises above the base rate while fitness is stuck and snaps back on
/// improvement; without it, this equals the base rate every generation.
/// </param>
/// <param name="SmoothedFitness">
/// The smoothed (rolling-average) fitness level from the velocity stagnation tracker, or null
/// when no velocity trigger is configured. This is the de-noised view of <paramref name="Fitness"/>
/// that the stall detector actually watches.
/// </param>
/// <param name="FitnessVelocity">
/// The smoothed velocity (rate of change) of <paramref name="SmoothedFitness"/>, or null when no
/// stall trigger is configured. This is the signal the CUSUM trigger integrates: positive while the
/// trend climbs, bobbing around zero on a plateau.
/// </param>
/// <param name="StallSum">
/// The CUSUM trigger's accumulated velocity shortfall, or null when no stall trigger is configured.
/// It fills toward the trigger's threshold while velocity stays below the real-climb rate (arming
/// the ramp) and drains back toward zero when progress resumes (releasing it); watching it shows
/// exactly when the ramp engages and disengages.
/// </param>
public sealed record GenerationProgress(
	int Generation,
	double Fitness,
	NandGenome Best,
	double MutationRate,
	double? SmoothedFitness = null,
	double? FitnessVelocity = null,
	double? StallSum = null);
