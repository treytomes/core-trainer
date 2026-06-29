namespace CoreTrainer.Core;

/// <summary>
/// Represents a warrior program with its instructions and starting address.
/// </summary>
public class Warrior(string name, Instruction[] instructions)
{
	public string Name { get; } = name;
	public Instruction[] Instructions { get; } = instructions;
	public int StartAddress { get; set; }
}
