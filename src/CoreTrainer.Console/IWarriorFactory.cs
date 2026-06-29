using CoreTrainer.Core;

namespace CoreTrainer.Console;

/// <summary>
/// Factory interface for creating warrior programs.
/// </summary>
public interface IWarriorFactory
{
	/// <summary>
	/// Creates the Dwarf warrior - a bomber that writes DAT instructions every 4 locations.
	/// </summary>
	Warrior CreateDwarf();

	/// <summary>
	/// Creates the Imp warrior - a simple one-instruction warrior that copies itself forward.
	/// </summary>
	Warrior CreateImp();
}
