namespace CoreTrainer.Core;

public enum AddressingMode
{
	/// <summary>
	/// # - The value itself
	/// </summary>
	Immediate = 0,

	/// <summary>
	/// $ - The address
	/// </summary>
	Direct,

	/// <summary>
	/// @ - The address pointed to by the value at the address
	/// </summary>
	Indirect,

	/// <summary>
	/// < - Decrement value at address, then use as pointer
	/// </summary>
	Decrement
}
