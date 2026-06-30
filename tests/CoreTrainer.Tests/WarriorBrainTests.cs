using CoreTrainer.Core;
using CoreTrainer.Networking.Nand;

namespace CoreTrainer.Tests;

/// <summary>
/// Tests for brain-driven warriors and their coexistence with standard warriors.
/// </summary>
public class WarriorBrainTests
{
	#region Test brain

	/// <summary>
	/// A deterministic brain that always emits a fixed instruction, ignoring its input.
	/// </summary>
	private sealed class FixedBrain(Instruction instruction, int windowRadius = 1) : IWarriorBrain
	{
		public int WindowRadius { get; } = windowRadius;

		public Instruction Decide(IReadOnlyList<Instruction> window, int programCounter, int processCount)
			=> instruction;
	}

	#endregion

	#region Brain drives execution

	[Fact]
	public void BrainWarrior_WritesDecidedInstructionToPcBeforeExecuting()
	{
		// The brain always emits a MOV $0,$1 (the Imp). The warrior is loaded as a NOP,
		// but the brain should overwrite the PC cell with its decision before execution.
		var imp = new Instruction(OpCode.MOV, AddressingMode.Direct, 0, AddressingMode.Direct, 1);
		var brain = new FixedBrain(imp);

		var mars = new Mars(warriorCount: 1);
		mars.LoadWarrior(0, new Warrior("Seed", [new Instruction(OpCode.NOP, AddressingMode.Direct, 0, AddressingMode.Direct, 0)]), 100, brain);

		mars.StepCycle();

		// The PC cell now holds the brain's instruction, attributed to warrior 0.
		var atPc = mars.Core.Read(100);
		Assert.Equal(OpCode.MOV, atPc.OpCode);
		Assert.Equal(0, mars.GetCellOwner(100));
		// And it executed as an Imp: cell 101 is now a copy.
		Assert.Equal(OpCode.MOV, mars.Core.Read(101).OpCode);
	}

	[Fact]
	public void BrainWarrior_EmittingDat_DiesLikeAnyOtherWarrior()
	{
		// A brain that emits DAT kills its own process — proving brain output flows
		// through the normal execution semantics.
		var dat = new Instruction(OpCode.DAT, AddressingMode.Immediate, 0, AddressingMode.Immediate, 0);
		var brain = new FixedBrain(dat);

		var mars = new Mars(warriorCount: 2);
		mars.LoadWarrior(0, new Warrior("Brainy", [new Instruction()]), 10, brain);
		mars.LoadWarrior(1, new Warrior("Loop", [new Instruction(OpCode.JMP, AddressingMode.Direct, 0, AddressingMode.Direct, 0)]), 4000);

		var result = mars.StepCycle();

		Assert.Equal(ExecutionResult.Win, result);
		Assert.Equal(1, mars.Winner); // the standard warrior outlives the suicidal brain
	}

	#endregion

	#region Coexistence

	[Fact]
	public void StandardAndBrainWarriors_CoexistInSameBattle()
	{
		// One classic in-memory warrior (Dwarf) and one brain-driven warrior share a core.
		var dwarf = new Warrior("Dwarf",
		[
			new Instruction(OpCode.ADD, AddressingMode.Immediate, 4, AddressingMode.Direct, 3),
			new Instruction(OpCode.MOV, AddressingMode.Direct, 2, AddressingMode.Indirect, 2),
			new Instruction(OpCode.JMP, AddressingMode.Direct, -2, AddressingMode.Direct, 0),
			new Instruction(OpCode.DAT, AddressingMode.Immediate, 0, AddressingMode.Immediate, 0)
		]);
		var impBrain = new FixedBrain(new Instruction(OpCode.MOV, AddressingMode.Direct, 0, AddressingMode.Direct, 1));

		var mars = new Mars(coreSize: 8000, maxCycles: 5000, maxProcesses: 8000, warriorCount: 2);
		mars.LoadWarrior(0, dwarf, 100);
		mars.LoadWarrior(1, new Warrior("ImpBrain", [new Instruction()]), 4000, impBrain);

		// The battle runs to completion without error; both kinds stepped each cycle.
		var winner = mars.Run();

		// Both warriors took actions: warrior 0 (Dwarf) bombed, warrior 1 (brain) crawled,
		// so each owns cells. The result is a valid terminal state (no exception is the point).
		Assert.True(winner is null or 0 or 1);
		Assert.True(mars.CurrentCycle > 0);
	}

	#endregion

	#region NandBrain integration

	[Fact]
	public void NandBrain_ProducesValidInstructionsThroughMars()
	{
		// A random NAND network wired to a matching codec must always yield valid,
		// executable instructions — the totality guarantee, end to end through Mars.
		var random = new Random(2024);
		var codec = new WarriorObservationCodec(windowRadius: 1, coreSize: 8000, maxProcesses: 8000);
		var genome = NandGenome.CreateRandom(
			inputCount: codec.InputBitCount,
			nodeCount: 200,
			outputCount: codec.OutputBitCount,
			random);
		var brain = new NandBrain(new NandNetwork(genome), codec);

		var mars = new Mars(coreSize: 8000, maxCycles: 1000, maxProcesses: 8000, warriorCount: 2);
		mars.LoadWarrior(0, new Warrior("Nand", [new Instruction()]), 100, brain);
		mars.LoadWarrior(1, new Warrior("Loop", [new Instruction(OpCode.JMP, AddressingMode.Direct, 0, AddressingMode.Direct, 0)]), 4000);

		// Running to completion without throwing proves every emitted instruction was valid.
		var winner = mars.Run();
		Assert.True(winner is null or 0 or 1);
	}

	[Fact]
	public void NandBrain_MismatchedCodecWidths_Throws()
	{
		var codec = new WarriorObservationCodec(windowRadius: 1, coreSize: 8000, maxProcesses: 8000);
		// Genome with the wrong input width for this codec.
		var genome = NandGenome.CreateRandom(inputCount: 5, nodeCount: 10, outputCount: codec.OutputBitCount, new Random(1));

		Assert.Throws<ArgumentException>(() => new NandBrain(new NandNetwork(genome), codec));
	}

	#endregion
}
