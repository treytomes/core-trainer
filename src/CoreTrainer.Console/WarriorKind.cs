namespace CoreTrainer.Console;

/// <summary>
/// Discriminates the kinds of warrior a battle roster entry can describe. New kinds
/// (e.g. an assembly-language source file) can be added here without reshaping the
/// roster schema.
/// </summary>
public enum WarriorKind
{
	/// <summary>
	/// A built-in archetype program (Dwarf, Imp). Uses the entry's <c>Archetype</c> field.
	/// </summary>
	Archetype,

	/// <summary>
	/// A trained NAND-network warrior loaded from a genome file. Uses the entry's
	/// <c>GenomeFile</c> field.
	/// </summary>
	Nand
}
