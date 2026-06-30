namespace CoreTrainer.Core;

/// <summary>
/// Encodes a warrior's local observation — the window of cells around its program
/// counter plus its own state (PC and process count) — into a fixed-width bit vector
/// suitable as input to a decision network, and decodes the network's output bits back
/// into the <see cref="Instruction"/> to execute.
/// </summary>
/// <remarks>
/// <para>
/// Input layout (least-significant-first within each field):
/// <list type="number">
/// <item><c>2 * WindowRadius + 1</c> cells, each <see cref="InstructionCodec.BitCount"/> bits,
/// ordered from offset <c>-WindowRadius</c> to <c>+WindowRadius</c> relative to the PC;</item>
/// <item>the program counter, <see cref="ProgramCounterBits"/> bits;</item>
/// <item>the process count, <see cref="ProcessCountBits"/> bits.</item>
/// </list>
/// </para>
/// <para>
/// PC and process count are encoded as plain binary. Plain binary is compact and total
/// (every pattern is a valid number), at the cost that numerically adjacent values can
/// differ in many bits. If training struggles to use these inputs, a unary/thermometer
/// encoding (adjacent values differ by one bit) is the usual alternative — it is wider
/// but smoother. The window cells dominate the input width regardless, so this choice
/// is not the primary cost driver.
/// </para>
/// <para>
/// The output side is simply <see cref="InstructionCodec"/>: the network emits
/// <see cref="OutputBitCount"/> bits, decoded into a guaranteed-valid instruction.
/// </para>
/// </remarks>
public sealed class WarriorObservationCodec
{
	#region Constructors

	/// <summary>
	/// Creates an observation codec.
	/// </summary>
	/// <param name="windowRadius">Number of cells on each side of the PC to include (≥ 0).</param>
	/// <param name="coreSize">Core size, used to size the program-counter field.</param>
	/// <param name="maxProcesses">Maximum process count, used to size the process-count field.</param>
	public WarriorObservationCodec(int windowRadius, int coreSize, int maxProcesses)
	{
		if (windowRadius < 0)
			throw new ArgumentOutOfRangeException(nameof(windowRadius), windowRadius, "Window radius cannot be negative.");
		if (coreSize < 1)
			throw new ArgumentOutOfRangeException(nameof(coreSize), coreSize, "Core size must be at least 1.");
		if (maxProcesses < 1)
			throw new ArgumentOutOfRangeException(nameof(maxProcesses), maxProcesses, "Max processes must be at least 1.");

		WindowRadius = windowRadius;
		CoreSize = coreSize;
		MaxProcesses = maxProcesses;

		WindowCellCount = 2 * windowRadius + 1;
		ProgramCounterBits = BitsToRepresent(coreSize);          // covers 0 .. coreSize-1
		ProcessCountBits = BitsToRepresent(maxProcesses + 1);    // covers 0 .. maxProcesses
		InputBitCount = WindowCellCount * InstructionCodec.BitCount + ProgramCounterBits + ProcessCountBits;
	}

	#endregion

	#region Properties

	/// <summary>Number of cells on each side of the PC included in the window.</summary>
	public int WindowRadius { get; }

	/// <summary>Core size used to size the PC field.</summary>
	public int CoreSize { get; }

	/// <summary>Maximum process count used to size the process-count field.</summary>
	public int MaxProcesses { get; }

	/// <summary>Total cells in the window (<c>2 * WindowRadius + 1</c>).</summary>
	public int WindowCellCount { get; }

	/// <summary>Bits used to encode the program counter.</summary>
	public int ProgramCounterBits { get; }

	/// <summary>Bits used to encode the process count.</summary>
	public int ProcessCountBits { get; }

	/// <summary>Total number of input bits the network receives.</summary>
	public int InputBitCount { get; }

	/// <summary>Number of output bits the network must produce (one encoded instruction).</summary>
	public int OutputBitCount => InstructionCodec.BitCount;

	#endregion

	#region Public Methods

	/// <summary>
	/// Encodes an observation into a new input bit vector of length <see cref="InputBitCount"/>.
	/// </summary>
	/// <param name="window">
	/// The window cells, ordered from <c>-WindowRadius</c> to <c>+WindowRadius</c>;
	/// length must equal <see cref="WindowCellCount"/>.
	/// </param>
	/// <param name="programCounter">The warrior's current program counter.</param>
	/// <param name="processCount">The warrior's current process count.</param>
	public bool[] EncodeObservation(IReadOnlyList<Instruction> window, int programCounter, int processCount)
	{
		ArgumentNullException.ThrowIfNull(window);
		if (window.Count != WindowCellCount)
			throw new ArgumentException($"Window must contain exactly {WindowCellCount} cells.", nameof(window));

		var bits = new bool[InputBitCount];
		var span = bits.AsSpan();
		var offset = 0;

		// Window cells, each a full encoded instruction.
		foreach (var cell in window)
		{
			InstructionCodec.Encode(cell, span.Slice(offset, InstructionCodec.BitCount));
			offset += InstructionCodec.BitCount;
		}

		// Program counter, clamped into range then packed.
		var pc = (uint)Math.Clamp(programCounter, 0, CoreSize - 1);
		PackBits(pc, span.Slice(offset, ProgramCounterBits));
		offset += ProgramCounterBits;

		// Process count, clamped into range then packed.
		var count = (uint)Math.Clamp(processCount, 0, MaxProcesses);
		PackBits(count, span.Slice(offset, ProcessCountBits));

		return bits;
	}

	/// <summary>
	/// Decodes the network's output bits into the instruction to execute. Every output
	/// pattern yields a valid instruction (see <see cref="InstructionCodec"/>).
	/// </summary>
	/// <param name="outputBits">Output bits; length must equal <see cref="OutputBitCount"/>.</param>
	public Instruction DecodeInstruction(ReadOnlySpan<bool> outputBits) => InstructionCodec.Decode(outputBits);

	#endregion

	#region Private Methods

	/// <summary>
	/// Minimal number of bits needed to represent the values <c>0 .. count-1</c>, at least 1.
	/// </summary>
	private static int BitsToRepresent(int count)
	{
		var bits = 1;
		long capacity = 2;
		while (capacity < count)
		{
			capacity <<= 1;
			bits++;
		}
		return bits;
	}

	/// <summary>
	/// Packs the low bits of <paramref name="value"/> into <paramref name="destination"/>,
	/// least-significant-bit first.
	/// </summary>
	private static void PackBits(uint value, Span<bool> destination)
	{
		for (var i = 0; i < destination.Length; i++)
		{
			destination[i] = ((value >> i) & 1u) != 0;
		}
	}

	#endregion
}
