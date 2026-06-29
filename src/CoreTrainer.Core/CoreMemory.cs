namespace CoreTrainer.Core;

public class CoreMemory
{
	#region Fields

	private readonly Instruction[] _memory;

	#endregion

	#region Constructors

	public CoreMemory(int size = 8000)
	{
		_memory = new Instruction[size];

		// Initialize with DAT instructions.
		for (var i = 0; i < size; i++)
		{
			_memory[i] = new Instruction();
		}
	}

	#endregion

	#region Properties

	public int Size => _memory.Length;

	#endregion

	#region Methods

	public Instruction Read(int address)
	{
		return _memory[Normalize(address)];
	}

	public void Write(int address, Instruction instruction)
	{
		_memory[Normalize(address)] = instruction;
	}

	public int Normalize(int address)
	{
		// Wrap around the core memory.
		return ((address % Size) + Size) % Size;
	}

	public void Load(int startAddress, IEnumerable<Instruction> instructions)
	{
		var address = Normalize(startAddress);
		foreach (var instruction in instructions)
		{
			_memory[address] = instruction;
			address = Normalize(address + 1);
		}
	}

	#endregion
}
