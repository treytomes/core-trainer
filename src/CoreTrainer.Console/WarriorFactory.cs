using CoreTrainer.Core;

namespace CoreTrainer.Console;

/// <summary>
/// Factory for creating classic Core Wars warriors.
/// </summary>
public class WarriorFactory : IWarriorFactory
{
	/// <summary>
	/// Creates the Dwarf warrior - a bomber that writes DAT instructions every 4 locations.
	/// </summary>
	public Warrior CreateDwarf()
	{
		var instructions = new Instruction[]
		{
			new(OpCode.ADD, AddressingMode.Immediate, 4, AddressingMode.Direct, 3),
			new(OpCode.MOV, AddressingMode.Direct, 2, AddressingMode.Indirect, 2),
			new(OpCode.JMP, AddressingMode.Direct, -2, AddressingMode.Direct, 0),
			new(OpCode.DAT, AddressingMode.Immediate, 0, AddressingMode.Immediate, 0)
		};

		return new Warrior("Dwarf", instructions);
	}

	/// <summary>
	/// Creates the Imp warrior - a simple one-instruction warrior that copies itself forward.
	/// </summary>
	public Warrior CreateImp()
	{
		var instructions = new Instruction[]
		{
			new(OpCode.MOV, AddressingMode.Direct, 0, AddressingMode.Direct, 1)
		};

		return new Warrior("Imp", instructions);
	}
}

