namespace CoreTrainer.Core;

/// <summary>
/// Helpers for choosing warrior starting positions within the circular core.
/// </summary>
public static class WarriorPlacement
{
	/// <summary>
	/// Finds a position at least <paramref name="minSeparation"/> cells away (by circular
	/// distance) from <paramref name="referencePos"/>.
	/// </summary>
	/// <param name="coreSize">Size of the core memory arena.</param>
	/// <param name="referencePos">Position to stay away from.</param>
	/// <param name="minSeparation">Minimum circular distance required.</param>
	/// <param name="random">Random source used to pick candidate positions.</param>
	/// <returns>A position satisfying the separation constraint.</returns>
	public static int GetSeparatedPosition(int coreSize, int referencePos, int minSeparation, Random random)
	{
		if (coreSize <= 0)
			throw new ArgumentOutOfRangeException(nameof(coreSize), "Core size must be positive");
		ArgumentNullException.ThrowIfNull(random);

		// The greatest achievable circular distance is half the core. Clamp the requested
		// separation so an impossible request can't spin this loop forever.
		var maxSeparation = coreSize / 2;
		var effectiveSeparation = Math.Min(minSeparation, maxSeparation);

		int position;
		int circularDistance;
		do
		{
			position = random.Next(coreSize);
			var linearDistance = Math.Abs(referencePos - position);
			circularDistance = Math.Min(linearDistance, coreSize - linearDistance);
		} while (circularDistance < effectiveSeparation);

		return position;
	}
}
