namespace CoreTrainer.Core;

public class Warrior(string name, Instruction[] instructions)
{
	public string Name { get; } = name;
	public Instruction[] Instructions { get; } = instructions;
	public int StartAddress { get; set; }
}
