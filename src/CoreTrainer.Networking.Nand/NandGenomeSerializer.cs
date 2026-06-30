using System.Globalization;
using System.Text;

namespace CoreTrainer.Networking.Nand;

/// <summary>
/// Serializes <see cref="NandGenome"/> instances to and from a compact, versioned text
/// format so trained genomes can be persisted and reloaded.
/// </summary>
/// <remarks>
/// <para>
/// The format is line-based and human-inspectable:
/// <code>
/// nandgenome v1
/// inputs &lt;count&gt;
/// outputs &lt;count&gt;
/// nodes &lt;a0&gt; &lt;b0&gt; &lt;a1&gt; &lt;b1&gt; ...
/// outputs-src &lt;o0&gt; &lt;o1&gt; ...
/// </code>
/// </para>
/// <para>
/// Deserialization constructs the genome through its validating constructor, so a
/// malformed or tampered file (wrong shape, or wiring that violates the backwards-only
/// invariant) raises a clear exception rather than yielding an invalid genome.
/// </para>
/// </remarks>
public static class NandGenomeSerializer
{
	#region Constants

	private const string Header = "nandgenome v1";

	#endregion

	#region Public Methods

	/// <summary>
	/// Serializes a genome to its text representation.
	/// </summary>
	public static string Serialize(NandGenome genome)
	{
		ArgumentNullException.ThrowIfNull(genome);

		var sb = new StringBuilder();
		sb.AppendLine(Header);
		sb.Append("inputs ").Append(genome.InputCount.ToString(CultureInfo.InvariantCulture)).AppendLine();
		sb.Append("outputs ").Append(genome.OutputCount.ToString(CultureInfo.InvariantCulture)).AppendLine();
		sb.Append("nodes").AppendLine(FormatInts(genome.NodeGenes));
		sb.Append("outputs-src").AppendLine(FormatInts(genome.OutputGenes));
		return sb.ToString();
	}

	/// <summary>
	/// Parses a genome from its text representation.
	/// </summary>
	/// <exception cref="FormatException">The text is not a recognized, well-formed genome.</exception>
	/// <exception cref="ArgumentException">The parsed genome violates the genome invariants.</exception>
	public static NandGenome Deserialize(string text)
	{
		ArgumentNullException.ThrowIfNull(text);

		var lines = text.Split('\n', StringSplitOptions.RemoveEmptyEntries);
		if (lines.Length < 5)
			throw new FormatException("Genome text is incomplete; expected header, inputs, outputs, nodes, and outputs-src lines.");

		if (lines[0].Trim() != Header)
			throw new FormatException($"Unrecognized genome header: '{lines[0].Trim()}'. Expected '{Header}'.");

		var inputCount = ParseScalar(lines[1], "inputs");
		var outputCount = ParseScalar(lines[2], "outputs");
		var nodeGenes = ParseVector(lines[3], "nodes");
		var outputGenes = ParseVector(lines[4], "outputs-src");

		// The validating constructor enforces shape and the backwards-only invariant.
		try
		{
			return new NandGenome(inputCount, outputCount, nodeGenes, outputGenes);
		}
		catch (Exception ex) when (ex is ArgumentException or ArgumentOutOfRangeException)
		{
			throw new FormatException($"Genome text describes an invalid genome: {ex.Message}", ex);
		}
	}

	/// <summary>
	/// Writes a genome to a file.
	/// </summary>
	public static void SaveToFile(NandGenome genome, string path)
	{
		ArgumentNullException.ThrowIfNull(genome);
		ArgumentException.ThrowIfNullOrWhiteSpace(path);
		File.WriteAllText(path, Serialize(genome));
	}

	/// <summary>
	/// Reads a genome from a file.
	/// </summary>
	public static NandGenome LoadFromFile(string path)
	{
		ArgumentException.ThrowIfNullOrWhiteSpace(path);
		return Deserialize(File.ReadAllText(path));
	}

	#endregion

	#region Private Methods

	private static string FormatInts(IReadOnlyList<int> values)
	{
		var sb = new StringBuilder();
		foreach (var value in values)
		{
			sb.Append(' ').Append(value.ToString(CultureInfo.InvariantCulture));
		}
		return sb.ToString();
	}

	/// <summary>
	/// Parses a "<paramref name="key"/> &lt;int&gt;" line.
	/// </summary>
	private static int ParseScalar(string line, string key)
	{
		var tokens = Tokenize(line, key);
		if (tokens.Length != 1)
			throw new FormatException($"The '{key}' line must contain exactly one integer.");
		return ParseInt(tokens[0], key);
	}

	/// <summary>
	/// Parses a "<paramref name="key"/> &lt;int&gt; &lt;int&gt; ..." line into an array
	/// (which may be empty).
	/// </summary>
	private static int[] ParseVector(string line, string key)
	{
		var tokens = Tokenize(line, key);
		var result = new int[tokens.Length];
		for (var i = 0; i < tokens.Length; i++)
		{
			result[i] = ParseInt(tokens[i], key);
		}
		return result;
	}

	/// <summary>
	/// Splits a line on whitespace, verifies its first token is <paramref name="key"/>,
	/// and returns the remaining value tokens.
	/// </summary>
	private static string[] Tokenize(string line, string key)
	{
		var parts = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
		if (parts.Length == 0 || parts[0] != key)
			throw new FormatException($"Expected a line beginning with '{key}' but found '{line.Trim()}'.");
		return parts[1..];
	}

	private static int ParseInt(string token, string key)
	{
		if (!int.TryParse(token, NumberStyles.Integer, CultureInfo.InvariantCulture, out var value))
			throw new FormatException($"The '{key}' line contains a non-integer token: '{token}'.");
		return value;
	}

	#endregion
}
