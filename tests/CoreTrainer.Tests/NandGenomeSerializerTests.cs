using CoreTrainer.Networking.Nand;

namespace CoreTrainer.Tests;

/// <summary>
/// Tests for genome serialization: round-trip fidelity and rejection of malformed
/// or invalid genome text. Exercises the default text strategy through the interface.
/// </summary>
public class NandGenomeSerializerTests
{
	private readonly INandGenomeSerializer _serializer = new TextNandGenomeSerializer();

	#region Round trip

	[Fact]
	public void SerializeThenDeserialize_PreservesEveryGene()
	{
		var random = new Random(31);
		var original = NandGenome.CreateRandom(inputCount: 6, nodeCount: 40, outputCount: 8, random);

		var restored = _serializer.Deserialize(_serializer.Serialize(original));

		Assert.Equal(original.InputCount, restored.InputCount);
		Assert.Equal(original.OutputCount, restored.OutputCount);
		Assert.Equal(original.NodeCount, restored.NodeCount);
		Assert.Equal(original.NodeGenes.ToArray(), restored.NodeGenes.ToArray());
		Assert.Equal(original.OutputGenes.ToArray(), restored.OutputGenes.ToArray());
	}

	[Fact]
	public void SerializeThenDeserialize_PreservesNodelessGenome()
	{
		// A genome with no nodes (outputs read inputs directly) is a valid edge case.
		var original = new NandGenome(inputCount: 3, outputCount: 2, nodeGenes: [], outputGenes: [0, 2]);

		var restored = _serializer.Deserialize(_serializer.Serialize(original));

		Assert.Equal(0, restored.NodeCount);
		Assert.Equal(original.OutputGenes.ToArray(), restored.OutputGenes.ToArray());
	}

	[Fact]
	public void SaveAndLoadFile_RoundTripsThroughDisk()
	{
		var random = new Random(99);
		var original = NandGenome.CreateRandom(inputCount: 4, nodeCount: 20, outputCount: 3, random);
		var path = Path.Combine(Path.GetTempPath(), $"genome_{Guid.NewGuid():N}.txt");

		try
		{
			_serializer.SaveToFile(original, path);
			var restored = _serializer.LoadFromFile(path);

			Assert.Equal(original.NodeGenes.ToArray(), restored.NodeGenes.ToArray());
			Assert.Equal(original.OutputGenes.ToArray(), restored.OutputGenes.ToArray());
		}
		finally
		{
			if (File.Exists(path)) File.Delete(path);
		}
	}

	#endregion

	#region Malformed input

	[Fact]
	public void Deserialize_UnknownHeader_Throws()
	{
		Assert.Throws<FormatException>(() => _serializer.Deserialize("totally not a genome\n"));
	}

	[Fact]
	public void Deserialize_Truncated_Throws()
	{
		Assert.Throws<FormatException>(() => _serializer.Deserialize("nandgenome v1\ninputs 2\n"));
	}

	[Fact]
	public void Deserialize_NonIntegerToken_Throws()
	{
		var text = "nandgenome v1\ninputs 2\noutputs 1\nnodes 0 x\noutputs-src 2\n";
		Assert.Throws<FormatException>(() => _serializer.Deserialize(text));
	}

	[Fact]
	public void Deserialize_InvalidWiring_ThrowsFormatException()
	{
		// Node 0 (address 2) references address 2 — a self-loop. The validating constructor
		// rejects it; the serializer surfaces that as a FormatException.
		var text = "nandgenome v1\ninputs 2\noutputs 1\nnodes 2 0\noutputs-src 2\n";

		var ex = Assert.Throws<FormatException>(() => _serializer.Deserialize(text));
		Assert.NotNull(ex.InnerException); // wraps the constructor's ArgumentException
	}

	#endregion
}
