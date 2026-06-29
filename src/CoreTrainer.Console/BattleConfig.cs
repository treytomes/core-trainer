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
/// A single warrior entry in a <see cref="BattleConfig"/>.
/// </summary>
public class BattleWarrior
{
	/// <summary>
	/// Which built-in warrior program to load.
	/// </summary>
	public WarriorArchetype Archetype { get; set; } = WarriorArchetype.Imp;

	/// <summary>
	/// Optional display name. When null or empty, the archetype's default name is used.
	/// </summary>
	public string? Name { get; set; }
}
