using CoreTrainer.Core;

namespace CoreTrainer.Tests;

/// <summary>
/// Tests for the MARS execution engine.
/// </summary>
public class MarsTests
{
	#region Process Lifecycle

	[Fact]
	public void StepCycle_WhenWarriorExecutesDat_WarriorDiesAndOpponentWins()
	{
		// Arrange - warrior 0 immediately executes a DAT (death), warrior 1 loops harmlessly.
		var suicide = new Warrior("Suicide", [new Instruction(OpCode.DAT, AddressingMode.Immediate, 0, AddressingMode.Immediate, 0)]);
		var survivor = new Warrior("Survivor", [new Instruction(OpCode.JMP, AddressingMode.Direct, 0, AddressingMode.Direct, 0)]);

		var mars = new Mars();
		mars.LoadWarrior(0, suicide, 10);
		mars.LoadWarrior(1, survivor, 4000);

		// Act
		var result = mars.StepCycle();

		// Assert
		Assert.Equal(ExecutionResult.Win, result);
		Assert.Equal(1, mars.Winner);
		Assert.Empty(mars.GetProcessQueue(0));
		Assert.Single(mars.GetProcessQueue(1));
	}

	[Fact]
	public void StepCycle_WhenSplExecuted_CreatesAdditionalProcess()
	{
		// Arrange - SPL to self then loop; the warrior should accumulate processes.
		var splitter = new Warrior("Splitter",
		[
			new Instruction(OpCode.SPL, AddressingMode.Direct, 0, AddressingMode.Direct, 0),
			new Instruction(OpCode.JMP, AddressingMode.Direct, 0, AddressingMode.Direct, 0)
		]);

		var mars = new Mars();
		mars.LoadWarrior(0, splitter, 0);
		mars.LoadWarrior(1, TestWarriors.CreateImp(), 4000);

		// Act - one cycle executes the SPL, which enqueues a second process for warrior 0.
		mars.StepCycle();

		// Assert
		Assert.Equal(2, mars.GetProcessQueue(0).Count);
	}

	[Fact]
	public void Run_WithTwoImmortalWarriors_EndsInDrawAtCycleLimit()
	{
		// Arrange - two harmless infinite loops can never kill each other.
		var loopA = new Warrior("LoopA", [new Instruction(OpCode.JMP, AddressingMode.Direct, 0, AddressingMode.Direct, 0)]);
		var loopB = new Warrior("LoopB", [new Instruction(OpCode.JMP, AddressingMode.Direct, 0, AddressingMode.Direct, 0)]);

		var mars = new Mars(maxCycles: 500);
		mars.LoadWarrior(0, loopA, 100);
		mars.LoadWarrior(1, loopB, 4000);

		// Act
		var winner = mars.Run();

		// Assert
		Assert.Null(winner);
		Assert.Equal(500, mars.CurrentCycle);
	}

	#endregion

	#region Multiple Warriors

	[Fact]
	public void Mars_SupportsMoreThanTwoWarriors()
	{
		var mars = new Mars(warriorCount: 5);

		Assert.Equal(5, mars.WarriorCount);
	}

	[Fact]
	public void StepCycle_WithManyWarriors_LastSurvivorWins()
	{
		// Three suicides and one survivor: after one cycle only the survivor is alive.
		Warrior Suicide(string name) =>
			new(name, [new Instruction(OpCode.DAT, AddressingMode.Immediate, 0, AddressingMode.Immediate, 0)]);
		var survivor = new Warrior("Survivor", [new Instruction(OpCode.JMP, AddressingMode.Direct, 0, AddressingMode.Direct, 0)]);

		var mars = new Mars(warriorCount: 4);
		mars.LoadWarrior(0, Suicide("A"), 10);
		mars.LoadWarrior(1, Suicide("B"), 2000);
		mars.LoadWarrior(2, survivor, 4000);
		mars.LoadWarrior(3, Suicide("C"), 6000);

		var result = mars.StepCycle();

		Assert.Equal(ExecutionResult.Win, result);
		Assert.Equal(2, mars.Winner);
	}

	[Fact]
	public void StepCycle_WhenAllWarriorsDieSameCycle_IsDraw()
	{
		Warrior Suicide(string name) =>
			new(name, [new Instruction(OpCode.DAT, AddressingMode.Immediate, 0, AddressingMode.Immediate, 0)]);

		var mars = new Mars(warriorCount: 3);
		mars.LoadWarrior(0, Suicide("A"), 10);
		mars.LoadWarrior(1, Suicide("B"), 2000);
		mars.LoadWarrior(2, Suicide("C"), 4000);

		var result = mars.StepCycle();

		Assert.Equal(ExecutionResult.Draw, result);
	}

	[Fact]
	public void Constructor_WithZeroWarriors_Throws()
	{
		Assert.Throws<ArgumentOutOfRangeException>(() => new Mars(warriorCount: 0));
	}

	#endregion

	#region Ownership Tracking

	[Fact]
	public void GetCellOwner_BeforeAnyWrites_ReturnsNoOwner()
	{
		var mars = new Mars();

		// A cell far from any loaded warrior has never been written.
		Assert.Equal(Mars.NoOwner, mars.GetCellOwner(2000));
	}

	[Fact]
	public void LoadWarrior_MarksLoadedCellsWithWarriorOwnership()
	{
		var mars = new Mars();
		var dwarf = TestWarriors.CreateDwarf();
		mars.LoadWarrior(0, dwarf, 100);

		// All four Dwarf instruction cells should be owned by warrior 0.
		for (var offset = 0; offset < dwarf.Instructions.Length; offset++)
		{
			Assert.Equal(0, mars.GetCellOwner(100 + offset));
		}
	}

	[Fact]
	public void GetCellOwner_AfterMov_AttributesWrittenCellToExecutingWarrior()
	{
		// Arrange - warrior 1 is a lone Imp; after one step it copies itself one cell forward.
		var mars = new Mars();
		mars.LoadWarrior(0, TestWarriors.CreateDwarf(), 100);
		mars.LoadWarrior(1, TestWarriors.CreateImp(), 4000);

		// Act - step until the Imp (warrior 1) has executed at least once.
		mars.StepCycle();
		mars.StepCycle();

		// Assert - the cell ahead of the Imp's start is now owned by warrior 1.
		Assert.Equal(1, mars.GetCellOwner(4001));
	}

	#endregion

	#region Classic Behavior - Imp Conversion

	// Documents a real Core Wars dynamic verified during development: the Imp crawls
	// 3x faster than the Dwarf bombs, so it overwrites the Dwarf's loop with its own
	// MOV $0,$1 before a bomb can land. Since MOV never kills a process, neither
	// warrior can die and the battle is a guaranteed draw.
	[Fact]
	public void DwarfVsImp_RunsToCompletion_AsDrawWithDwarfConvertedToImp()
	{
		// Arrange
		var mars = new Mars();
		const int dwarfStart = 100;
		const int impStart = 4100;
		mars.LoadWarrior(0, TestWarriors.CreateDwarf(), dwarfStart);
		mars.LoadWarrior(1, TestWarriors.CreateImp(), impStart);

		// Act
		var winner = mars.Run();

		// Assert - draw, both warriors still alive at the cycle limit.
		Assert.Null(winner);
		Assert.NotEmpty(mars.GetProcessQueue(0));
		Assert.NotEmpty(mars.GetProcessQueue(1));

		// The Dwarf's original loop cells have been overwritten into Imp code.
		for (var offset = 0; offset < 3; offset++)
		{
			Assert.True(
				TestWarriors.IsImpInstruction(mars.Core.Read(dwarfStart + offset)),
				$"Expected Dwarf loop cell at offset {offset} to be converted to an Imp instruction.");
		}
	}

	#endregion
}
