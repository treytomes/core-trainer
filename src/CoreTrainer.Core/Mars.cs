namespace CoreTrainer.Core;

/// <summary>
/// MARS (Memory Array Redcode Simulator) execution engine that manages warrior processes and executes Redcode instructions.
/// </summary>
public class Mars
{
	#region Fields

	/// <summary>
	/// Cell owner value indicating no warrior has written to a cell.
	/// </summary>
	public const int NoOwner = -1;

	private readonly CoreMemory _core;
	private readonly Queue<int>[] _processQueues;
	private readonly IWarriorBrain?[] _brains;
	private readonly int[] _cellOwners;
	private readonly int _maxCycles;
	private readonly int _maxProcesses;
	private int _currentCycle;
	private int _winner = NoOwner;

	#endregion

	#region Constructors

	public Mars(int coreSize = 8000, int maxCycles = 80000, int maxProcesses = 8000, int warriorCount = 2)
	{
		if (maxCycles <= 0)
			throw new ArgumentOutOfRangeException(nameof(maxCycles), "Max cycles must be positive");
		if (maxProcesses <= 0)
			throw new ArgumentOutOfRangeException(nameof(maxProcesses), "Max processes must be positive");
		if (warriorCount < 1)
			throw new ArgumentOutOfRangeException(nameof(warriorCount), "Warrior count must be at least 1");

		_core = new CoreMemory(coreSize);
		_processQueues = new Queue<int>[warriorCount];
		for (var i = 0; i < warriorCount; i++)
		{
			_processQueues[i] = new Queue<int>();
		}
		_brains = new IWarriorBrain?[warriorCount];
		_cellOwners = new int[_core.Size];
		Array.Fill(_cellOwners, NoOwner);
		_maxCycles = maxCycles;
		_maxProcesses = maxProcesses;
		_currentCycle = 0;
	}

	#endregion


	#region Properties

	public CoreMemory Core => _core;
	public int CurrentCycle => _currentCycle;
	public int MaxCycles => _maxCycles;
	public int WarriorCount => _processQueues.Length;

	/// <summary>
	/// Index of the winning warrior once a battle ends in a win, or <see cref="NoOwner"/>
	/// if no single warrior has won (still running, draw, or timeout).
	/// </summary>
	public int Winner => _winner;

	#endregion

	#region Ownership

	/// <summary>
	/// Gets the index of the warrior that last wrote to the given address,
	/// or <see cref="NoOwner"/> if the cell has never been written.
	/// </summary>
	/// <param name="address">Memory address (will be normalized).</param>
	/// <returns>Owning warrior index, or <see cref="NoOwner"/>.</returns>
	public int GetCellOwner(int address) => _cellOwners[_core.Normalize(address)];

	/// <summary>
	/// Writes an instruction to core memory and records the writing warrior as the cell's owner.
	/// </summary>
	private void WriteOwned(int warriorIndex, int address, Instruction instruction)
	{
		var normalized = _core.Normalize(address);
		_core.Write(normalized, instruction);
		_cellOwners[normalized] = warriorIndex;
	}

	/// <summary>
	/// Reads the window of cells centered on <paramref name="center"/>, ordered from
	/// offset <c>-radius</c> to <c>+radius</c> (length <c>2 * radius + 1</c>), wrapping
	/// around the core.
	/// </summary>
	private Instruction[] ReadWindow(int center, int radius)
	{
		var window = new Instruction[2 * radius + 1];
		for (var offset = -radius; offset <= radius; offset++)
		{
			window[offset + radius] = _core.Read(center + offset);
		}
		return window;
	}

	#endregion

	#region Public Methods

	/// <summary>
	/// Loads a warrior into the core at the given address.
	/// </summary>
	/// <param name="warriorIndex">Slot to load into.</param>
	/// <param name="warrior">The warrior program to place in memory.</param>
	/// <param name="startAddress">Address of the warrior's first instruction and initial PC.</param>
	/// <param name="brain">
	/// Optional decision maker. When supplied, the warrior synthesizes the instruction it
	/// executes each cycle via the brain instead of running its in-memory program. When
	/// null, the warrior runs classically. This is what lets standard and brain-driven
	/// warriors share one battle.
	/// </param>
	public void LoadWarrior(int warriorIndex, Warrior warrior, int startAddress, IWarriorBrain? brain = null)
	{
		if (warriorIndex < 0 || warriorIndex >= _processQueues.Length)
			throw new ArgumentOutOfRangeException(nameof(warriorIndex),
				$"Warrior index must be between 0 and {_processQueues.Length - 1}");
		if (warrior == null)
			throw new ArgumentNullException(nameof(warrior));

		warrior.StartAddress = startAddress;
		for (var i = 0; i < warrior.Instructions.Length; i++)
		{
			WriteOwned(warriorIndex, startAddress + i, warrior.Instructions[i]);
		}
		_brains[warriorIndex] = brain;
		_processQueues[warriorIndex].Enqueue(startAddress);
	}

	/// <summary>
	/// Executes a single simulation cycle (one instruction per warrior).
	/// </summary>
	/// <returns>The result of this cycle - Continue, or a terminal state (winner/draw/timeout).</returns>
	public ExecutionResult StepCycle()
	{
		if (_currentCycle >= _maxCycles)
			return ExecutionResult.MaxCyclesReached;

		// Execute one instruction for each warrior
		for (var i = 0; i < _processQueues.Length; i++)
		{
			if (_processQueues[i].Count == 0)
				continue;

			var pc = _processQueues[i].Dequeue();
			var nextPc = ExecuteInstruction(i, pc);

			if (nextPc.HasValue)
			{
				_processQueues[i].Enqueue(nextPc.Value);
			}
		}

		_currentCycle++;

		// Determine result after this cycle
		return DetermineResult();
	}

	/// <summary>
	/// Runs the complete battle until a winner is determined or max cycles reached.
	/// </summary>
	/// <returns>Index of the winning warrior, or null for draw/timeout.</returns>
	public int? Run()
	{
		_currentCycle = 0;

		while (true)
		{
			var result = StepCycle();

			switch (result)
			{
				case ExecutionResult.Win:
					return _winner;
				case ExecutionResult.Draw:
				case ExecutionResult.MaxCyclesReached:
					return null;
				case ExecutionResult.Continue:
					continue;
				default:
					return null;
			}
		}
	}

	/// <summary>
	/// Gets a read-only view of the process queue for a specific warrior.
	/// </summary>
	/// <param name="warriorIndex">Index of the warrior (0-based).</param>
	/// <returns>Read-only collection of process counters.</returns>
	public IReadOnlyCollection<int> GetProcessQueue(int warriorIndex)
	{
		if (warriorIndex < 0 || warriorIndex >= _processQueues.Length)
			throw new ArgumentOutOfRangeException(nameof(warriorIndex),
				$"Warrior index must be between 0 and {_processQueues.Length - 1}");

		return [.. _processQueues[warriorIndex]];
	}

	#endregion

	#region Private Methods

	/// <summary>
	/// Determines the battle result based on current process queue states:
	/// exactly one warrior alive is a win, zero alive is a draw, otherwise continue.
	/// </summary>
	private ExecutionResult DetermineResult()
	{
		var aliveCount = 0;
		var lastAlive = NoOwner;

		for (var i = 0; i < _processQueues.Length; i++)
		{
			if (_processQueues[i].Count > 0)
			{
				aliveCount++;
				lastAlive = i;
			}
		}

		if (aliveCount == 0)
			return ExecutionResult.Draw;

		// A single survivor wins only when there were rivals to begin with. With one
		// warrior loaded, it never "wins" — the battle runs to the cycle limit.
		if (aliveCount == 1 && _processQueues.Length > 1)
		{
			_winner = lastAlive;
			return ExecutionResult.Win;
		}

		return ExecutionResult.Continue;
	}

	private int? ExecuteInstruction(int warriorIndex, int pc)
	{
		// A brain-driven warrior synthesizes the instruction to run this cycle and writes
		// it to the PC before execution. Standard warriors (no brain) run what is already
		// in memory — letting both kinds share the same battle.
		var brain = _brains[warriorIndex];
		if (brain is not null)
		{
			var window = ReadWindow(pc, brain.WindowRadius);
			// The executing process was dequeued in StepCycle, so add it back for a count
			// that includes the process currently making the decision.
			var processCount = _processQueues[warriorIndex].Count + 1;
			var decided = brain.Decide(window, pc, processCount);
			WriteOwned(warriorIndex, pc, decided);
		}

		var instruction = _core.Read(pc);

		switch (instruction.OpCode)
		{
			case OpCode.DAT:
				return null; // Process dies

			case OpCode.MOV:
				{
					var aAddr = ResolveAddress(warriorIndex, pc, instruction.AMode, instruction.AValue);
					var bAddr = ResolveAddress(warriorIndex, pc, instruction.BMode, instruction.BValue);
					var source = _core.Read(aAddr);
					WriteOwned(warriorIndex, bAddr, source);
					return _core.Normalize(pc + 1);
				}

			case OpCode.ADD:
				{
					var aAddr = ResolveAddress(warriorIndex, pc, instruction.AMode, instruction.AValue);
					var bAddr = ResolveAddress(warriorIndex, pc, instruction.BMode, instruction.BValue);
					var aInst = _core.Read(aAddr);
					var bInst = _core.Read(bAddr);
					var result = bInst with { BValue = bInst.BValue + aInst.BValue };
					WriteOwned(warriorIndex, bAddr, result);
					return _core.Normalize(pc + 1);
				}

			case OpCode.SUB:
				{
					var aAddr = ResolveAddress(warriorIndex, pc, instruction.AMode, instruction.AValue);
					var bAddr = ResolveAddress(warriorIndex, pc, instruction.BMode, instruction.BValue);
					var aInst = _core.Read(aAddr);
					var bInst = _core.Read(bAddr);
					var result = bInst with { BValue = bInst.BValue - aInst.BValue };
					WriteOwned(warriorIndex, bAddr, result);
					return _core.Normalize(pc + 1);
				}

			case OpCode.MUL:
				{
					var aAddr = ResolveAddress(warriorIndex, pc, instruction.AMode, instruction.AValue);
					var bAddr = ResolveAddress(warriorIndex, pc, instruction.BMode, instruction.BValue);
					var aInst = _core.Read(aAddr);
					var bInst = _core.Read(bAddr);
					var result = bInst with { BValue = bInst.BValue * aInst.BValue };
					WriteOwned(warriorIndex, bAddr, result);
					return _core.Normalize(pc + 1);
				}

			case OpCode.DIV:
				{
					var aAddr = ResolveAddress(warriorIndex, pc, instruction.AMode, instruction.AValue);
					var bAddr = ResolveAddress(warriorIndex, pc, instruction.BMode, instruction.BValue);
					var aInst = _core.Read(aAddr);
					var bInst = _core.Read(bAddr);
					// Division by zero kills the process (ICWS-94).
					if (aInst.BValue == 0)
						return null;
					var result = bInst with { BValue = bInst.BValue / aInst.BValue };
					WriteOwned(warriorIndex, bAddr, result);
					return _core.Normalize(pc + 1);
				}

			case OpCode.MOD:
				{
					var aAddr = ResolveAddress(warriorIndex, pc, instruction.AMode, instruction.AValue);
					var bAddr = ResolveAddress(warriorIndex, pc, instruction.BMode, instruction.BValue);
					var aInst = _core.Read(aAddr);
					var bInst = _core.Read(bAddr);
					// Modulo by zero kills the process (ICWS-94).
					if (aInst.BValue == 0)
						return null;
					var result = bInst with { BValue = bInst.BValue % aInst.BValue };
					WriteOwned(warriorIndex, bAddr, result);
					return _core.Normalize(pc + 1);
				}

			case OpCode.JMP:
				{
					var aAddr = ResolveAddress(warriorIndex, pc, instruction.AMode, instruction.AValue);
					return aAddr;
				}

			case OpCode.JMZ:
				{
					var aAddr = ResolveAddress(warriorIndex, pc, instruction.AMode, instruction.AValue);
					var bAddr = ResolveAddress(warriorIndex, pc, instruction.BMode, instruction.BValue);
					var bInst = _core.Read(bAddr);
					if (bInst.BValue == 0)
						return aAddr;
					return _core.Normalize(pc + 1);
				}

			case OpCode.JMN:
				{
					var aAddr = ResolveAddress(warriorIndex, pc, instruction.AMode, instruction.AValue);
					var bAddr = ResolveAddress(warriorIndex, pc, instruction.BMode, instruction.BValue);
					var bInst = _core.Read(bAddr);
					if (bInst.BValue != 0)
						return aAddr;
					return _core.Normalize(pc + 1);
				}

			case OpCode.DJN:
				{
					var aAddr = ResolveAddress(warriorIndex, pc, instruction.AMode, instruction.AValue);
					var bAddr = ResolveAddress(warriorIndex, pc, instruction.BMode, instruction.BValue);
					var bInst = _core.Read(bAddr);
					var decremented = bInst with { BValue = bInst.BValue - 1 };
					WriteOwned(warriorIndex, bAddr, decremented);
					if (decremented.BValue != 0)
						return aAddr;
					return _core.Normalize(pc + 1);
				}

			case OpCode.CMP:
				{
					var aAddr = ResolveAddress(warriorIndex, pc, instruction.AMode, instruction.AValue);
					var bAddr = ResolveAddress(warriorIndex, pc, instruction.BMode, instruction.BValue);
					var aInst = _core.Read(aAddr);
					var bInst = _core.Read(bAddr);
					if (aInst.Equals(bInst))
						return _core.Normalize(pc + 2); // Skip next instruction
					return _core.Normalize(pc + 1);
				}

			case OpCode.SNE:
				{
					var aAddr = ResolveAddress(warriorIndex, pc, instruction.AMode, instruction.AValue);
					var bAddr = ResolveAddress(warriorIndex, pc, instruction.BMode, instruction.BValue);
					var aInst = _core.Read(aAddr);
					var bInst = _core.Read(bAddr);
					if (!aInst.Equals(bInst))
						return _core.Normalize(pc + 2); // Skip next instruction
					return _core.Normalize(pc + 1);
				}

			case OpCode.SLT:
				{
					var aAddr = ResolveAddress(warriorIndex, pc, instruction.AMode, instruction.AValue);
					var bAddr = ResolveAddress(warriorIndex, pc, instruction.BMode, instruction.BValue);
					var aInst = _core.Read(aAddr);
					var bInst = _core.Read(bAddr);
					if (aInst.BValue < bInst.BValue)
						return _core.Normalize(pc + 2); // Skip next instruction
					return _core.Normalize(pc + 1);
				}

			case OpCode.SPL:
				{
					var aAddr = ResolveAddress(warriorIndex, pc, instruction.AMode, instruction.AValue);
					// Only create new process if under the limit
					if (_processQueues[warriorIndex].Count < _maxProcesses)
					{
						_processQueues[warriorIndex].Enqueue(aAddr);
					}
					return _core.Normalize(pc + 1);
				}

			case OpCode.NOP:
				return _core.Normalize(pc + 1);

			default:
				return _core.Normalize(pc + 1);
		}
	}

	private int ResolveAddress(int warriorIndex, int pc, AddressingMode mode, int value)
	{
		switch (mode)
		{
			case AddressingMode.Immediate:
				return pc; // Return current location for immediate values

			case AddressingMode.Direct:
				return _core.Normalize(pc + value);

			case AddressingMode.Indirect:
				{
					var directAddr = _core.Normalize(pc + value);
					var pointer = _core.Read(directAddr);
					return _core.Normalize(directAddr + pointer.BValue);
				}

			case AddressingMode.Decrement:
				{
					var directAddr = _core.Normalize(pc + value);
					var pointer = _core.Read(directAddr);
					var decremented = pointer with { BValue = pointer.BValue - 1 };
					WriteOwned(warriorIndex, directAddr, decremented);
					return _core.Normalize(directAddr + decremented.BValue);
				}

			default:
				return _core.Normalize(pc + value);
		}
	}

	#endregion
}
