namespace CoreTrainer.Core;

public record Instruction
{
	#region Constructors

	public Instruction()
	{
		OpCode = OpCode.DAT;
		AMode = AddressingMode.Direct;
		AValue = 0;
		BMode = AddressingMode.Direct;
		BValue = 0;
	}


	public Instruction(OpCode opCode, AddressingMode aMode, int aValue, AddressingMode bMode, int bValue)
	{
		OpCode = opCode;
		AMode = aMode;
		AValue = aValue;
		BMode = bMode;
		BValue = bValue;
	}

	#endregion

	#region Properties

	public OpCode OpCode { get; init; }
	public AddressingMode AMode { get; init; }
	public int AValue { get; init; }
	public AddressingMode BMode { get; init; }
	public int BValue { get; init; }

	#endregion

	#region Methods

	public override string ToString()
	{
		var aModeStr = AMode switch
		{
			AddressingMode.Immediate => "#",
			AddressingMode.Indirect => "@",
			AddressingMode.Decrement => "<",
			_ => "$"
		};

		var bModeStr = BMode switch
		{
			AddressingMode.Immediate => "#",
			AddressingMode.Indirect => "@",
			AddressingMode.Decrement => "<",
			_ => "$"
		};

		return $"{OpCode} {aModeStr}{AValue}, {bModeStr}{BValue}";
	}

	#endregion
}
