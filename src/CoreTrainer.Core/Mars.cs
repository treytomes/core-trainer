namespace CoreTrainer.Core;

/// <summary>
/// MARS (Memory Array Redcode Simulator) execution engine that manages warrior processes and executes Redcode instructions.
/// </summary>
public class Mars
{
	#region Fields

	private readonly CoreMemory _core;
	private readonly Queue<int>[] _processQueues;
	private readonly int _maxCycles;
	private readonly int _maxProcesses;
	private int _currentCycle;

	#endregion

	#region Constructors

	public Mars(int coreSize = 8000, int maxCycles = 80000, int maxProcesses = 8000)
	{
		if (maxCycles <= 0)
			throw new ArgumentOutOfRangeException(nameof(maxCycles), "Max cycles must be positive");
		if (maxProcesses <= 0)
			throw new ArgumentOutOfRangeException(nameof(maxProcesses), "Max processes must be positive");

		_core = new CoreMemory(coreSize);
		_processQueues = new Queue<int>[2];
		_processQueues[0] = new Queue<int>();
		_processQueues[1] = new Queue<int>();
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

	#endregion

	#region Public Methods

	public void LoadWarrior(int warriorIndex, Warrior warrior, int startAddress)
	{
		if (warriorIndex < 0 || warriorIndex >= _processQueues.Length)
			throw new ArgumentOutOfRangeException(nameof(warriorIndex),
				$"Warrior index must be between 0 and {_processQueues.Length - 1}");
		if (warrior == null)
			throw new ArgumentNullException(nameof(warrior));

		warrior.StartAddress = startAddress;
		_core.Load(startAddress, warrior.Instructions);
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
	/// <returns>Index of winning warrior (0 or 1), or null for draw/timeout.</returns>
	public int? Run()
	{
		_currentCycle = 0;

		while (true)
		{
			var result = StepCycle();

			switch (result)
			{
				case ExecutionResult.Warrior0Wins:
					return 0;
				case ExecutionResult.Warrior1Wins:
					return 1;
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
	/// Determines the battle result based on current process queue states.
	/// </summary>
	private ExecutionResult DetermineResult()
	{
		var warrior0Alive = _processQueues[0].Count > 0;
		var warrior1Alive = _processQueues[1].Count > 0;

		if (!warrior0Alive && warrior1Alive)
			return ExecutionResult.Warrior1Wins;
		if (warrior1Alive && !warrior0Alive)
			return ExecutionResult.Warrior0Wins;
		if (!warrior0Alive && !warrior1Alive)
			return ExecutionResult.Draw;

		return ExecutionResult.Continue;
	}

	private int? ExecuteInstruction(int warriorIndex, int pc)
	{
		var instruction = _core.Read(pc);

		switch (instruction.OpCode)
		{
			case OpCode.DAT:
				return null; // Process dies

			case OpCode.MOV:
				{
					var aAddr = ResolveAddress(pc, instruction.AMode, instruction.AValue);
					var bAddr = ResolveAddress(pc, instruction.BMode, instruction.BValue);
					var source = _core.Read(aAddr);
					_core.Write(bAddr, source);
					return _core.Normalize(pc + 1);
				}

			case OpCode.ADD:
				{
					var aAddr = ResolveAddress(pc, instruction.AMode, instruction.AValue);
					var bAddr = ResolveAddress(pc, instruction.BMode, instruction.BValue);
					var aInst = _core.Read(aAddr);
					var bInst = _core.Read(bAddr);
					var result = bInst with { BValue = bInst.BValue + aInst.BValue };
					_core.Write(bAddr, result);
					return _core.Normalize(pc + 1);
				}

			case OpCode.SUB:
				{
					var aAddr = ResolveAddress(pc, instruction.AMode, instruction.AValue);
					var bAddr = ResolveAddress(pc, instruction.BMode, instruction.BValue);
					var aInst = _core.Read(aAddr);
					var bInst = _core.Read(bAddr);
					var result = bInst with { BValue = bInst.BValue - aInst.BValue };
					_core.Write(bAddr, result);
					return _core.Normalize(pc + 1);
				}

			case OpCode.JMP:
				{
					var aAddr = ResolveAddress(pc, instruction.AMode, instruction.AValue);
					return aAddr;
				}

			case OpCode.JMZ:
				{
					var aAddr = ResolveAddress(pc, instruction.AMode, instruction.AValue);
					var bAddr = ResolveAddress(pc, instruction.BMode, instruction.BValue);
					var bInst = _core.Read(bAddr);
					if (bInst.BValue == 0)
						return aAddr;
					return _core.Normalize(pc + 1);
				}

			case OpCode.JMN:
				{
					var aAddr = ResolveAddress(pc, instruction.AMode, instruction.AValue);
					var bAddr = ResolveAddress(pc, instruction.BMode, instruction.BValue);
					var bInst = _core.Read(bAddr);
					if (bInst.BValue != 0)
						return aAddr;
					return _core.Normalize(pc + 1);
				}

			case OpCode.DJN:
				{
					var aAddr = ResolveAddress(pc, instruction.AMode, instruction.AValue);
					var bAddr = ResolveAddress(pc, instruction.BMode, instruction.BValue);
					var bInst = _core.Read(bAddr);
					var decremented = bInst with { BValue = bInst.BValue - 1 };
					_core.Write(bAddr, decremented);
					if (decremented.BValue != 0)
						return aAddr;
					return _core.Normalize(pc + 1);
				}

			case OpCode.CMP:
				{
					var aAddr = ResolveAddress(pc, instruction.AMode, instruction.AValue);
					var bAddr = ResolveAddress(pc, instruction.BMode, instruction.BValue);
					var aInst = _core.Read(aAddr);
					var bInst = _core.Read(bAddr);
					if (aInst.Equals(bInst))
						return _core.Normalize(pc + 2); // Skip next instruction
					return _core.Normalize(pc + 1);
				}

			case OpCode.SPL:
				{
					var aAddr = ResolveAddress(pc, instruction.AMode, instruction.AValue);
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

	private int ResolveAddress(int pc, AddressingMode mode, int value)
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
					_core.Write(directAddr, decremented);
					return _core.Normalize(directAddr + decremented.BValue);
				}

			default:
				return _core.Normalize(pc + value);
		}
	}

	#endregion
}
