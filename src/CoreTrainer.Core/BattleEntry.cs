namespace CoreTrainer.Core;

/// <summary>
/// A combatant in a battle: the warrior program to load, plus an optional brain that
/// drives it. A null brain means the warrior runs its in-memory program classically;
/// a non-null brain means it synthesizes its instruction each cycle. This pairing is
/// what lets standard and brain-driven warriors share one roster.
/// </summary>
/// <param name="Warrior">The warrior program to load into the core.</param>
/// <param name="Brain">Optional decision maker; null for a standard warrior.</param>
public sealed record BattleEntry(Warrior Warrior, IWarriorBrain? Brain = null);
