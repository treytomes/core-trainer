# Coding Standards Update - One Type Per File

## Summary

Added "One Type Per File" rule to the coding standards document.

## Rule Added

**Location:** [.claude/coding-standards.md](../coding-standards.md) - File Organization section

### Core Rule

**Each file should contain exactly one type (class, struct, record, interface, or enum).**

### Requirements

- File name must match type name exactly
- Example: `Button.cs` contains `Button` class
- Example: `IWarriorFactory.cs` contains `IWarriorFactory` interface

### Exceptions

Two acceptable exceptions to the rule:

1. **Nested types** - Can be in same file as containing type
   ```csharp
   // Button.cs
   public class Button
   {
       private class ButtonState { } // OK - nested type
   }
   ```

2. **Small helper types** - Internal types < 5 lines closely tied to main type
   ```csharp
   // SimulationController.cs
   public class SimulationController { }
   
   public enum SimulationState { Stopped, Paused, Running, Finished } // OK - small and coupled
   ```

### Rationale

**Benefits:**
- **Predictable navigation** - Type name = file name
- **Better IDE search** - "Go to file" matches type name
- **Clearer git history** - Changes isolated per type
- **Easier code reviews** - One type = one concern
- **Prevents "god files"** - Multiple related types stay separated

**Example:**
```
CoreTrainer.Graphics/UI/
├── Button.cs         → class Button
├── Toolbar.cs        → class Toolbar
└── Panel.cs          → class Panel
```

## Current Codebase Compliance

**Reviewed files:**

✅ **CoreTrainer.Core/**
- `OpCode.cs` → `OpCode` enum
- `AddressingMode.cs` → `AddressingMode` enum
- `Instruction.cs` → `Instruction` record
- `CoreMemory.cs` → `CoreMemory` class
- `Mars.cs` → `Mars` class
- `Warrior.cs` → `Warrior` class
- `ExecutionResult.cs` → `ExecutionResult` enum

✅ **CoreTrainer.Graphics/**
- `ColorScheme.cs` → `ColorScheme` static class
- `GridConfig.cs` → `GridConfig` class
- `UI/Button.cs` → `Button` class
- `UI/Toolbar.cs` → `Toolbar` class
- `Rendering/CoreRenderer.cs` → `CoreRenderer` class
- `Simulation/SimulationController.cs` → `SimulationController` class + `SimulationState` enum (exception: small, coupled)

✅ **CoreTrainer.Console/**
- `Program.cs` → `Program` static class
- `BattleService.cs` → `BattleService` class
- `WarriorFactory.cs` → `WarriorFactory` class
- `IWarriorFactory.cs` → `IWarriorFactory` interface
- `CoreWarsOptions.cs` → `CoreWarsOptions` class
- `WindowOptions.cs` → `WindowOptions` class

**Compliance Status:** ✅ 100% compliant

All files follow the one-type-per-file rule, with `SimulationController.cs` using the acceptable exception for a small, coupled enum.

## Decision: SimulationState Enum Placement

**Current:** `SimulationState` enum in `SimulationController.cs`

**Justification for keeping together:**
1. Small enum (4 values, ~20 lines with docs)
2. Only used by `SimulationController`
3. Tightly coupled - no other consumer
4. Splitting would add ceremony without benefit

**If it grows:** Move to separate `SimulationState.cs` file when:
- Enum exceeds 10 values
- Enum gets methods/extensions
- Other types need to reference it
- Documentation exceeds 50 lines

## Future Considerations

**When to split small enums:**
- Enum is referenced by multiple types
- Enum is part of public API used across assemblies
- Enum has significant documentation or methods
- Team preference for stricter separation

**Current philosophy:** Pragmatic compliance - follow the rule by default, use exceptions sparingly for small helper types.

## Related Standards

This rule complements existing standards:
- File-scoped namespaces (reduces nesting)
- Region directives (organize within single type)
- Primary constructors (keep constructors concise)

## Examples

### ✅ Good - Separate Files

```
UI/
├── Button.cs
│   └── public class Button
├── Toolbar.cs
│   └── public class Toolbar
└── Panel.cs
    └── public class Panel
```

### ✅ Good - Small Coupled Enum

```csharp
// SimulationController.cs
public class SimulationController
{
    public SimulationState State { get; }
}

public enum SimulationState
{
    Stopped,
    Paused,
    Running,
    Finished
}
```

### ❌ Bad - Multiple Unrelated Types

```csharp
// UIComponents.cs (DON'T DO THIS)
public class Button { }
public class Label { }
public class Panel { }
public enum ButtonState { }
```

Should be:
```
Button.cs → Button + ButtonState (if ButtonState is small and only used by Button)
Label.cs → Label
Panel.cs → Panel
```

### ✅ Good - Nested Type Exception

```csharp
// Button.cs
public class Button
{
    private class ButtonState  // OK - nested and private
    {
        public bool IsHovered { get; set; }
        public bool IsPressed { get; set; }
    }
}
```

## Documentation Updated

- ✅ Added "One Type Per File" section to coding-standards.md
- ✅ Placed before "Namespace and Using Statements" section
- ✅ Included rationale and examples
- ✅ Specified exceptions clearly
- ✅ Current codebase already compliant
