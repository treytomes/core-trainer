using CoreTrainer.Core;

namespace CoreTrainer.Tests;

/// <summary>
/// Tests for the arithmetic and skip opcodes that were previously defined but
/// not executed (MUL, DIV, MOD, SLT) plus the new SNE.
/// </summary>
public class OpCodeExecutionTests
{
	#region Helpers

	// Reads the B-value of the cell a warrior's lone instruction targets, after one step.
	// Layout: instruction at 0 operates on a data cell, exercised via direct B addressing.
	private static Mars SingleWarrior(Instruction program, int startAddress = 0)
	{
		var mars = new Mars(warriorCount: 1);
		mars.LoadWarrior(0, new Warrior("T", [program]), startAddress);
		return mars;
	}

	#endregion

	#region Arithmetic

	[Fact]
	public void Mul_MultipliesBValuesIntoTarget()
	{
		// MUL $1, $2 : target cell 2's B-field becomes cell2.B * cell1.B = 5 * 3 = 15.
		var mars = new Mars(warriorCount: 1);
		var program = new Instruction[]
		{
			new(OpCode.MUL, AddressingMode.Direct, 1, AddressingMode.Direct, 2),
			new(OpCode.DAT, AddressingMode.Immediate, 0, AddressingMode.Immediate, 3),
			new(OpCode.DAT, AddressingMode.Immediate, 0, AddressingMode.Immediate, 5)
		};
		mars.LoadWarrior(0, new Warrior("T", program), 0);

		mars.StepCycle();

		Assert.Equal(15, mars.Core.Read(2).BValue);
	}

	[Fact]
	public void Div_ByZero_KillsProcess()
	{
		// DIV #0, $0 : divide by zero must terminate the process, not throw.
		var mars = SingleWarrior(
			new Instruction(OpCode.DIV, AddressingMode.Immediate, 0, AddressingMode.Direct, 0));

		var result = mars.StepCycle();

		// The lone warrior died; with a single warrior that is a draw (no rivals).
		Assert.Empty(mars.GetProcessQueue(0));
		Assert.Equal(ExecutionResult.Draw, result);
	}

	[Fact]
	public void Mod_ByZero_KillsProcess()
	{
		var mars = SingleWarrior(
			new Instruction(OpCode.MOD, AddressingMode.Immediate, 0, AddressingMode.Direct, 0));

		var result = mars.StepCycle();

		Assert.Empty(mars.GetProcessQueue(0));
		Assert.Equal(ExecutionResult.Draw, result);
	}

	#endregion

	#region Skips

	[Fact]
	public void Sne_SkipsNextWhenOperandsDiffer()
	{
		// SNE compares the two referenced cells; differing cells skip the next instruction.
		var mars = new Mars(warriorCount: 1);
		var program = new Instruction[]
		{
			new(OpCode.SNE, AddressingMode.Direct, 1, AddressingMode.Direct, 2),
			new(OpCode.DAT, AddressingMode.Immediate, 0, AddressingMode.Immediate, 0), // skipped if differ
			new(OpCode.NOP, AddressingMode.Direct, 0, AddressingMode.Direct, 7)         // distinct from cell 1
		};
		mars.LoadWarrior(0, new Warrior("T", program), 0);

		var next = StepToNextPc(mars);

		// Cells 1 and 2 differ, so the process should land on address 2, not 1.
		Assert.Equal(2, next);
	}

	[Fact]
	public void Slt_SkipsNextWhenAValueLessThanBValue()
	{
		// SLT $1, $2 : cell1.B (1) < cell2.B (4) -> skip the next instruction.
		var mars = new Mars(warriorCount: 1);
		var program = new Instruction[]
		{
			new(OpCode.SLT, AddressingMode.Direct, 1, AddressingMode.Direct, 2),
			new(OpCode.DAT, AddressingMode.Immediate, 0, AddressingMode.Immediate, 1),
			new(OpCode.DAT, AddressingMode.Immediate, 0, AddressingMode.Immediate, 4)
		};
		mars.LoadWarrior(0, new Warrior("T", program), 0);

		var next = StepToNextPc(mars);

		// 1 < 4 is true, so the process skips cell 1 and lands on cell 2.
		Assert.Equal(2, next);
	}

	[Fact]
	public void Slt_DoesNotSkipWhenAValueNotLessThanBValue()
	{
		// SLT $1, $2 : cell1.B (4) < cell2.B (1) is false -> no skip, advance to cell 1.
		var mars = new Mars(warriorCount: 1);
		var program = new Instruction[]
		{
			new(OpCode.SLT, AddressingMode.Direct, 1, AddressingMode.Direct, 2),
			new(OpCode.DAT, AddressingMode.Immediate, 0, AddressingMode.Immediate, 4),
			new(OpCode.DAT, AddressingMode.Immediate, 0, AddressingMode.Immediate, 1)
		};
		mars.LoadWarrior(0, new Warrior("T", program), 0);

		var next = StepToNextPc(mars);

		Assert.Equal(1, next);
	}

	#endregion

	#region Step helper

	// Steps one cycle and returns the warrior's resulting PC (the head of its queue).
	private static int StepToNextPc(Mars mars)
	{
		mars.StepCycle();
		var queue = mars.GetProcessQueue(0);
		return queue.First();
	}

	#endregion
}
