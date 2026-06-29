using CoreTrainer.Core;

namespace CoreTrainer.Console;

public static class Program
{
	public static void Main()
	{
		System.Console.WriteLine("=== Core Wars Simulator ===\n");

		// Create two simple warriors
		var dwarf = CreateDwarf();
		var imp = CreateImp();

		// Create MARS
		var mars = new Mars(coreSize: 8000, maxCycles: 80000);

		// Load warriors at different positions
		var random = new Random();
		var warrior1Pos = random.Next(mars.Core.Size);
		var warrior2Pos = 0;
		var circularDistance = 0;
		do
		{
			warrior2Pos = random.Next(mars.Core.Size);
			// Calculate circular distance accounting for wraparound
			var linearDistance = Math.Abs(warrior1Pos - warrior2Pos);
			circularDistance = Math.Min(linearDistance, mars.Core.Size - linearDistance);
		} while (circularDistance < 100);

		System.Console.WriteLine($"Loading {dwarf.Name} at position {warrior1Pos}");
		mars.LoadWarrior(0, dwarf, warrior1Pos);

		System.Console.WriteLine($"Loading {imp.Name} at position {warrior2Pos}");
		System.Console.WriteLine();
		mars.LoadWarrior(1, imp, warrior2Pos);

		// Run the battle
		System.Console.WriteLine("Starting battle...");
		var winner = mars.Run();

		System.Console.WriteLine($"\nBattle finished after {mars.CurrentCycle} cycles");

		if (winner.HasValue)
		{
			var winnerName = winner.Value == 0 ? dwarf.Name : imp.Name;
			System.Console.WriteLine($"Winner: {winnerName}");
		}
		else
		{
			System.Console.WriteLine("Result: Draw or time limit reached");
		}
	}

	// Helper methods to create warriors
	static Warrior CreateDwarf()
	{
		// Dwarf: Bombs every 4th location
		var instructions = new Instruction[]
		{
			new(OpCode.ADD, AddressingMode.Immediate, 4, AddressingMode.Direct, 3),
			new(OpCode.MOV, AddressingMode.Direct, 2, AddressingMode.Indirect, 2),
			new(OpCode.JMP, AddressingMode.Direct, -2, AddressingMode.Direct, 0),
			new(OpCode.DAT, AddressingMode.Immediate, 0, AddressingMode.Immediate, 0)
		};

		return new Warrior("Dwarf", instructions);
	}

	static Warrior CreateImp()
	{
		// Imp: Simple one-instruction warrior that copies itself forward
		var instructions = new Instruction[]
		{
			new(OpCode.MOV, AddressingMode.Direct, 0, AddressingMode.Direct, 1)
		};

		return new Warrior("Imp", instructions);
	}
}
