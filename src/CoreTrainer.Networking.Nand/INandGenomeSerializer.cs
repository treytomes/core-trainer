namespace CoreTrainer.Networking.Nand;

/// <summary>
/// Serializes <see cref="NandGenome"/> instances to and from a string representation,
/// so the persistence format can be chosen (and swapped) via dependency injection.
/// </summary>
/// <remarks>
/// Implementations define the format via <see cref="Serialize"/> / <see cref="Deserialize"/>.
/// File persistence is provided here as default methods built on those two, so a strategy
/// only has to implement the format itself.
/// </remarks>
public interface INandGenomeSerializer
{
	/// <summary>
	/// Serializes a genome to its string representation.
	/// </summary>
	string Serialize(NandGenome genome);

	/// <summary>
	/// Parses a genome from its string representation.
	/// </summary>
	/// <exception cref="FormatException">The text is not a recognized, well-formed genome.</exception>
	NandGenome Deserialize(string text);

	/// <summary>
	/// Writes a genome to a file.
	/// </summary>
	void SaveToFile(NandGenome genome, string path)
	{
		ArgumentNullException.ThrowIfNull(genome);
		ArgumentException.ThrowIfNullOrWhiteSpace(path);
		File.WriteAllText(path, Serialize(genome));
	}

	/// <summary>
	/// Reads a genome from a file.
	/// </summary>
	NandGenome LoadFromFile(string path)
	{
		ArgumentException.ThrowIfNullOrWhiteSpace(path);
		return Deserialize(File.ReadAllText(path));
	}
}
