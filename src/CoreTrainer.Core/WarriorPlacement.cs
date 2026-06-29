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

	/// <summary>
	/// Chooses <paramref name="count"/> starting positions, each kept at least
	/// <paramref name="minSeparation"/> cells (circular distance) from every other.
	/// </summary>
	/// <remarks>
	/// Perfect pairwise separation may be impossible when many warriors share a small
	/// core, so placement is best-effort: each position gets a bounded number of
	/// attempts to satisfy the constraint, after which the last candidate is accepted.
	/// The result therefore always contains exactly <paramref name="count"/> positions.
	/// </remarks>
	/// <param name="coreSize">Size of the core memory arena.</param>
	/// <param name="count">Number of positions to choose.</param>
	/// <param name="minSeparation">Minimum circular distance between any two positions.</param>
	/// <param name="random">Random source used to pick candidate positions.</param>
	/// <returns>An array of <paramref name="count"/> positions.</returns>
	public static int[] GetSeparatedPositions(int coreSize, int count, int minSeparation, Random random)
	{
		if (coreSize <= 0)
			throw new ArgumentOutOfRangeException(nameof(coreSize), "Core size must be positive");
		if (count < 0)
			throw new ArgumentOutOfRangeException(nameof(count), "Count cannot be negative");
		ArgumentNullException.ThrowIfNull(random);

		var effectiveSeparation = Math.Min(minSeparation, coreSize / 2);
		var positions = new int[count];

		// Bound the work per position so an over-crowded core can't loop forever.
		const int maxAttemptsPerPosition = 100;

		for (var i = 0; i < count; i++)
		{
			var candidate = random.Next(coreSize);

			for (var attempt = 0; attempt < maxAttemptsPerPosition; attempt++)
			{
				if (IsFarEnough(candidate, positions, i, coreSize, effectiveSeparation))
					break;

				candidate = random.Next(coreSize);
			}

			positions[i] = candidate;
		}

		return positions;
	}

	/// <summary>
	/// Returns true when <paramref name="candidate"/> is at least <paramref name="minSeparation"/>
	/// cells (circular distance) from each of the first <paramref name="placedCount"/> positions.
	/// </summary>
	private static bool IsFarEnough(int candidate, int[] positions, int placedCount, int coreSize, int minSeparation)
	{
		for (var j = 0; j < placedCount; j++)
		{
			var linear = Math.Abs(positions[j] - candidate);
			var circular = Math.Min(linear, coreSize - linear);
			if (circular < minSeparation)
				return false;
		}

		return true;
	}
}
