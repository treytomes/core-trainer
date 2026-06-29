namespace CoreTrainer.Core;

public enum OpCode
{
	/// <summary>
	/// Data - Kills the process
	/// </summary>
	DAT = 0,

	/// <summary>
	/// Move - Copies data from A to B
	/// </summary>
	MOV,

	/// <summary>
	/// Add - Adds A to B
	/// </summary>
	ADD,

	/// <summary>
	/// Subtract - Subtracts A from B
	/// </summary>
	SUB,

	/// <summary>
	/// Multiply - Multiplies B by A
	/// </summary>
	MUL,

	/// <summary>
	/// Divide - Divides B by A
	/// </summary>
	DIV,

	/// <summary>
	/// Modulo - B modulo A
	/// </summary>
	MOD,

	/// <summary>
	/// Jump - Transfers execution to A
	/// </summary>
	JMP,

	/// <summary>
	/// Jump if Zero - Jumps to A if B is zero
	/// </summary>
	JMZ,

	/// <summary>
	/// Jump if Not Zero - Jumps to A if B is not zero
	/// </summary>
	JMN,

	/// <summary>
	/// Decrement and Jump if Not Zero
	/// </summary>
	DJN,

	/// <summary>
	/// Compare - Skips next instruction if A equals B
	/// </summary>
	CMP,

	/// <summary>
	/// Skip if Less Than
	/// </summary>
	SLT,

	/// <summary>
	/// Split - Creates new process at A
	/// </summary>
	SPL,

	/// <summary>
	/// No Operation
	/// </summary>
	NOP
}
