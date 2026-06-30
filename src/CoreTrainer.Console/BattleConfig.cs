namespace CoreTrainer.Console;

/// <summary>
/// Defines the roster of warriors loaded into a battle. Loaded from a battle file
/// (battle.yaml by default, overridable via the --battle CLI option).
/// </summary>
public class BattleConfig
{
	/// <summary>
	/// The warriors to load, in order. Each becomes one combatant.
	/// </summary>
	public List<BattleWarrior> Warriors { get; set; } = [];
}

/// <summary>
/// A single warrior entry in a <see cref="BattleConfig"/>. The <see cref="Kind"/>
/// discriminator selects which other fields apply.
/// </summary>
public class BattleWarrior
{
	/// <summary>
	/// What kind of warrior this entry describes. Defaults to <see cref="WarriorKind.Archetype"/>
	/// so existing rosters that omit the field keep working.
	/// </summary>
	public WarriorKind Kind { get; set; } = WarriorKind.Archetype;

	/// <summary>
	/// Optional display name. When null or empty, a default name for the kind is used.
	/// </summary>
	public string? Name { get; set; }

	/// <summary>
	/// For <see cref="WarriorKind.Archetype"/>: which built-in program to load.
	/// </summary>
	public WarriorArchetype Archetype { get; set; } = WarriorArchetype.Imp;

	/// <summary>
	/// For <see cref="WarriorKind.Nand"/>: path to the trained genome file. Relative paths
	/// are resolved against the executable directory.
	/// </summary>
	public string? GenomeFile { get; set; }
}
