# Mars Step-Based Execution Refactoring

## Summary

Successfully refactored the Mars simulation engine to support step-based execution while maintaining backward compatibility with the existing `Run()` method.

## Changes Made

### New Files

**[ExecutionResult.cs](../../src/CoreTrainer.Core/ExecutionResult.cs)**
- Enum representing simulation state: `Continue`, `Warrior0Wins`, `Warrior1Wins`, `Draw`, `MaxCyclesReached`
- Provides clearer state than nullable int return value
- Enables better state management for visualizer

### Modified Files

**[Mars.cs](../../src/CoreTrainer.Core/Mars.cs)**

**New Properties:**
- `MaxCycles` - Exposes max cycle limit (was private)
- `WarriorCount` - Returns number of warriors (future-proofs for N warriors)

**New Methods:**

1. **`StepCycle()` → ExecutionResult**
   - Executes a single simulation cycle (one instruction per warrior)
   - Returns current battle state after the cycle
   - Core building block for visualizer frame loop
   - Enables pause/resume/variable speed

2. **`GetProcessQueue(int warriorIndex)` → IReadOnlyCollection<int>**
   - Returns read-only view of warrior's process queue
   - Enables visualizer to display process positions
   - Uses collection expression `[.. _processQueues[warriorIndex]]` for efficient copy

3. **`DetermineResult()` → ExecutionResult** (private helper)
   - Extracts win condition logic from Run()
   - Checks process queue states and returns appropriate result
   - Reused by both `StepCycle()` and `Run()`

**Refactored Methods:**

- **`Run()` → int?**
  - Now implemented using `StepCycle()` in a loop
  - Maps `ExecutionResult` to nullable int for backward compatibility
  - Zero code duplication - all execution logic in `StepCycle()`
  - Existing console app works unchanged

## Architecture Benefits

### Step-Based Control
```csharp
// Visualizer can now control execution frame-by-frame
while (mars.StepCycle() == ExecutionResult.Continue)
{
    // Render current state
    // Handle input (pause/speed change)
    // Sleep for frame timing
}
```

### Backward Compatibility
```csharp
// Existing console app continues to work
var winner = mars.Run(); // Unchanged API
```

### State Inspection
```csharp
// Visualizer can query simulation state
var dwarf Processes = mars.GetProcessQueue(0);
var impProcesses = mars.GetProcessQueue(1);
var cycle = mars.CurrentCycle;
```

## Code Quality

- ✅ **0 warnings, 0 errors** - Clean build
- ✅ **Full backward compatibility** - Existing code unchanged
- ✅ **No code duplication** - `Run()` delegates to `StepCycle()`
- ✅ **Clear state model** - `ExecutionResult` enum more explicit than nullable int
- ✅ **Read-only access** - `GetProcessQueue()` returns immutable snapshot
- ✅ **Modern C#** - Uses collection expressions `[..]`

## Testing

Verified with existing Dwarf vs Imp battle:
- Execution completes normally
- Draw detected at 80,000 cycles
- Spectre.Console UI displays correctly
- No behavior changes

## Next Steps

**Ready for visualizer implementation:**
1. ✅ Step-based execution available
2. ✅ State query methods available  
3. ✅ Clear result enumeration
4. ⬜ Create CoreTrainer.Visualizer project
5. ⬜ Integrate Raylib rendering
6. ⬜ Connect `StepCycle()` to render loop

## Design Decisions

**Why keep nullable int in `Run()`?**
- Maintains API compatibility with BattleService
- Console app doesn't need to change
- Visualizer uses `StepCycle()` directly

**Why collection expression `[..]` instead of `.ToArray()`?**
- More idiomatic in modern C#
- Slightly more efficient
- Matches language evolution direction

**Why extract `DetermineResult()` helper?**
- Eliminates duplication between `StepCycle()` and old `Run()` logic
- Single source of truth for win conditions
- Easier to extend for N-warrior battles

**Why `IReadOnlyCollection<int>` not `IReadOnlyList<int>`?**
- Queue doesn't guarantee indexed access semantics
- Collection is sufficient (visualizer just needs to iterate)
- More honest API contract

## Performance Impact

- **Negligible overhead** - `Run()` adds single switch statement per cycle
- **StepCycle()** has identical logic to original `Run()` loop body
- Collection expression is optimized by compiler
- No allocations except `GetProcessQueue()` snapshot (on-demand only)

## Breaking Changes

**None.** This is a purely additive refactoring.

All existing code continues to work without modification.
