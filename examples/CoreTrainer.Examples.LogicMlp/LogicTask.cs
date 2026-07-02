using static TorchSharp.torch;

namespace CoreTrainer.Examples.LogicMlp;

/// <summary>
/// A learnable Boolean task: a name, its input/output widths, and its full truth table as tensors
/// (inputs and targets in {0,1}). Every task is small enough to fit its <i>entire</i> function in
/// memory, so there is no train/test split — accuracy over the table is exhaustive by definition.
/// </summary>
/// <remarks>
/// The tasks form a deliberate <b>difficulty ladder</b>, because a single hard task is a poor proof of
/// mechanism: the 4-bit adder is known to be hard for <i>every</i> method here (CGP walls on its carry
/// chain), so its failure would not distinguish "the learned-gate mechanism is broken" from "this task
/// is just hard." The ladder isolates one capability per rung, so a failure points somewhere:
/// <list type="number">
/// <item><b>Bitwise</b> (AND/OR/XOR) — depth 1, no carry, no routing: can a node learn to pick the
/// right gate at all? The floor. If this fails, the mechanism is broken.</item>
/// <item><b>Parity</b> — a balanced XOR tree, depth ~log(N): the canonical "impossible for a linear
/// model, natural for logic" task, and forgiving of fixed random wiring (XOR is associative).</item>
/// <item><b>Adder</b> (1→2→4 bit) — introduces the carry chain, increasingly deep. The stretch.</item>
/// </list>
/// </remarks>
public sealed class LogicTask
{
	#region Properties

	/// <summary>A short human-readable task name, for logs and the per-run log filename.</summary>
	public required string Name { get; init; }

	/// <summary>Number of input bits.</summary>
	public required int InputCount { get; init; }

	/// <summary>Number of output bits.</summary>
	public required int OutputCount { get; init; }

	/// <summary>Full truth-table inputs, shape <c>[rows, InputCount]</c>, values in {0,1}.</summary>
	public required Tensor Inputs { get; init; }

	/// <summary>Full truth-table targets, shape <c>[rows, OutputCount]</c>, values in {0,1}.</summary>
	public required Tensor Targets { get; init; }

	/// <summary>Number of rows in the truth table (<c>2^InputCount</c>).</summary>
	public required int Rows { get; init; }

	#endregion

	#region Factories

	/// <summary>
	/// Element-wise binary op on two <paramref name="width"/>-bit inputs: output bit i = op(a_i, b_i).
	/// Depth-1, no interaction between bit positions — the simplest possible "learn to pick a gate" test.
	/// </summary>
	/// <param name="op">Which bitwise function: <c>"AND"</c>, <c>"OR"</c>, or <c>"XOR"</c>.</param>
	public static LogicTask Bitwise(string op, int width)
	{
		Func<int, int, int> f = op switch
		{
			"AND" => (a, b) => a & b,
			"OR" => (a, b) => a | b,
			"XOR" => (a, b) => a ^ b,
			_ => throw new ArgumentException($"Unknown bitwise op '{op}'.", nameof(op)),
		};

		var inputCount = 2 * width;
		return Build($"bitwise-{op.ToLowerInvariant()}-{width}bit", inputCount, width, inputs =>
		{
			// inputs holds [a bits][b bits]; produce op per bit position.
			var outputs = new int[width];
			for (var i = 0; i < width; i++)
				outputs[i] = f(inputs[i], inputs[width + i]);
			return outputs;
		});
	}

	/// <summary>
	/// N-bit parity: a single output bit that is the XOR of all input bits. A balanced XOR tree of depth
	/// ~log2(N); the classic function a linear model cannot represent but a logic net can.
	/// </summary>
	public static LogicTask Parity(int width) =>
		Build($"parity-{width}bit", width, 1, inputs =>
		{
			var p = 0;
			foreach (var bit in inputs)
				p ^= bit;
			return new[] { p };
		});

	/// <summary>
	/// A <paramref name="bitWidth"/>-bit adder/subtractor, identical layout to the CGP and DiffNand
	/// examples: inputs <c>[A][B][opSelect]</c>, outputs interleaved <c>sum0,c1,sum1,...,carryOut</c>.
	/// The stretch rung — a ripple carry chain several NAND-levels deep per bit.
	/// </summary>
	public static LogicTask Adder(int bitWidth)
	{
		var inputCount = 2 * bitWidth + 1;
		var outputCount = 2 * bitWidth + 1;
		return Build($"adder-{bitWidth}bit", inputCount, outputCount, inputs =>
		{
			var a = 0;
			var b = 0;
			for (var bit = 0; bit < bitWidth; bit++)
			{
				a |= inputs[bit] << bit;
				b |= inputs[bitWidth + bit] << bit;
			}
			var subtract = inputs[2 * bitWidth] == 1;
			var packed = ExpectedAdder(a, b, subtract, bitWidth);

			var outputs = new int[outputCount];
			for (var bit = 0; bit < outputCount; bit++)
				outputs[bit] = (packed >> bit) & 1;
			return outputs;
		});
	}

	#endregion

	#region Helpers

	// Enumerates all 2^inputCount input rows, calls `logic` to get each row's output bits, and packs
	// both into {0,1} tensors. Input bit i of a row is bit i of the row index (LSB-first).
	private static LogicTask Build(string name, int inputCount, int outputCount, Func<int[], int[]> logic)
	{
		var rows = 1 << inputCount;
		var inputData = new float[rows, inputCount];
		var targetData = new float[rows, outputCount];
		var bits = new int[inputCount];

		for (var row = 0; row < rows; row++)
		{
			for (var i = 0; i < inputCount; i++)
			{
				bits[i] = (row >> i) & 1;
				inputData[row, i] = bits[i];
			}
			var outputs = logic(bits);
			for (var o = 0; o < outputCount; o++)
				targetData[row, o] = outputs[o];
		}

		return new LogicTask
		{
			Name = name,
			InputCount = inputCount,
			OutputCount = outputCount,
			Inputs = tensor(inputData),
			Targets = tensor(targetData),
			Rows = rows,
		};
	}

	// The correct interleaved sum/carry output for (a, b, op), packed LSB-first — identical to the CGP
	// and DiffNand examples: ripple-carry adder, two's-complement subtraction (a + ~b + 1).
	private static int ExpectedAdder(int a, int b, bool subtract, int bitWidth)
	{
		var packed = 0;
		var carry = subtract ? 1 : 0;
		for (var k = 0; k < bitWidth; k++)
		{
			var ak = (a >> k) & 1;
			var bk = (b >> k) & 1;
			var op2 = subtract ? bk ^ 1 : bk;
			var full = ak + op2 + carry;

			packed |= (full & 1) << (2 * k);
			carry = full >> 1;
			packed |= carry << (2 * k + 1);
		}
		packed |= ((subtract ? 1 : 0) ^ carry) << (2 * bitWidth);
		return packed;
	}

	#endregion
}
