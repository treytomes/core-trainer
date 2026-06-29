using CoreTrainer.Core;

namespace CoreTrainer.Console;

/// <summary>
/// Factory interface for creating warrior programs.
/// </summary>
public interface IWarriorFactory
{
	/// <summary>
	/// Creates a warrior for the given archetype.
	/// </summary>
	/// <param name="archetype">Which built-in warrior to create.</param>
	/// <returns>A new warrior instance.</returns>
	Warrior Create(WarriorArchetype archetype);

	/// <summary>
	/// Creates the Dwarf warrior - a bomber that writes DAT instructions every 4 locations.
	/// </summary>
	Warrior CreateDwarf();

	/// <summary>
	/// Creates the Imp warrior - a simple one-instruction warrior that copies itself forward.
	/// </summary>
	Warrior CreateImp();
}
