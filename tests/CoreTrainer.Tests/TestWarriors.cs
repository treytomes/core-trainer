using CoreTrainer.Core;

namespace CoreTrainer.Tests;

/// <summary>
/// Shared warrior definitions used across tests.
/// </summary>
internal static class TestWarriors
{
	/// <summary>
	/// The classic Dwarf - a bomber that drops DAT bombs every 4 cells.
	/// </summary>
	public static Warrior CreateDwarf() => new("Dwarf",
	[
		new Instruction(OpCode.ADD, AddressingMode.Immediate, 4, AddressingMode.Direct, 3),
		new Instruction(OpCode.MOV, AddressingMode.Direct, 2, AddressingMode.Indirect, 2),
		new Instruction(OpCode.JMP, AddressingMode.Direct, -2, AddressingMode.Direct, 0),
		new Instruction(OpCode.DAT, AddressingMode.Immediate, 0, AddressingMode.Immediate, 0)
	]);

	/// <summary>
	/// The classic Imp - a single instruction that copies itself forward.
	/// </summary>
	public static Warrior CreateImp() => new("Imp",
	[
		new Instruction(OpCode.MOV, AddressingMode.Direct, 0, AddressingMode.Direct, 1)
	]);

	/// <summary>
	/// Determines whether an instruction is the Imp signature (MOV $0, $1).
	/// </summary>
	public static bool IsImpInstruction(Instruction instruction) =>
		instruction.OpCode == OpCode.MOV
		&& instruction.AMode == AddressingMode.Direct && instruction.AValue == 0
		&& instruction.BMode == AddressingMode.Direct && instruction.BValue == 1;
}
