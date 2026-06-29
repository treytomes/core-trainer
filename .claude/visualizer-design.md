# Core Wars Visualizer Design Document

## Implementation Status

**Phase 1: Core Library Extensions** ✅ COMPLETE  
**Phase 2: Raylib Project Setup** ✅ COMPLETE  
**Phase 3: Basic Rendering** ⬜ Not Started  
**Phase 4: Simulation Integration** ⬜ Not Started  
**Phase 5: Visual Polish** ⬜ Not Started  
**Phase 6: Advanced Features** ⬜ Future

**Current State:** Foundation complete. Ready to implement grid rendering and simulation control.

---

## Overview

Design a real-time graphical visualizer for Core Wars battles using Raylib-cs (C# bindings for Raylib). The visualizer will display the memory arena, warrior positions, execution flow, and battle statistics.

## Architecture Decisions

### Project Structure

```
CoreTrainer.Visualizer/          # New Raylib GUI project
├── Program.cs                   # Raylib window initialization
├── Scenes/
│   ├── IScene.cs               # Scene interface
│   ├── BattleScene.cs          # Main battle visualization
│   └── MenuScene.cs            # Warrior selection menu (future)
├── Rendering/
│   ├── CoreRenderer.cs         # Memory grid visualization
│   ├── StatisticsRenderer.cs   # Stats panel (cycles, processes)
│   └── ColorScheme.cs          # Color definitions per warrior
├── Simulation/
│   ├── SimulationController.cs # Controls Mars execution speed/pausing
│   └── SimulationState.cs      # Snapshot of battle state
└── Input/
    └── InputHandler.cs         # Keyboard/mouse input
```

### Key Design Patterns

**Observer Pattern**: Mars needs to notify visualizer of state changes
- Problem: Current `Mars.Run()` executes entire battle synchronously
- Solution: Add event hooks or make Mars step-by-step controllable

**State Snapshot**: Visualizer needs read-only view of simulation state
- Core memory contents
- Active process queues
- Current cycle count
- Last executed instructions

**Scene Management**: Different visualization modes
- Battle scene (main view)
- Statistics scene (post-battle analysis)
- Menu scene (warrior selection)

## Core Library Changes Needed

### Option 1: Event-Based (Observer Pattern)

**Pros:**
- Minimal changes to Mars
- Keeps simulation logic separate from presentation
- Easy to add multiple observers (logger, visualizer, metrics)

**Cons:**
- Event overhead on every instruction
- Hard to implement pause/resume
- Difficult to step backward

```csharp
// Add to Mars.cs
public event EventHandler<InstructionExecutedEventArgs>? InstructionExecuted;
public event EventHandler<CycleCompletedEventArgs>? CycleCompleted;

// In ExecuteInstruction()
InstructionExecuted?.Invoke(this, new InstructionExecutedEventArgs(
    warriorIndex, pc, instruction, nextPc));
```

### Option 2: Step-Based Execution (Recommended)

**Pros:**
- Full control over simulation speed
- Easy pause/resume/step-forward
- Can run at arbitrary speeds (slow-mo, fast-forward)
- Simpler state inspection

**Cons:**
- Breaks existing `Run()` API
- Need to refactor BattleService

```csharp
// Add to Mars.cs
public enum ExecutionResult
{
    Continue,
    Warrior0Wins,
    Warrior1Wins,
    Draw,
    MaxCyclesReached
}

public ExecutionResult StepCycle()
{
    // Execute one cycle (one instruction per warrior)
    // Return current battle state
}

public ExecutionResult StepInstruction(int warriorIndex)
{
    // Execute one instruction for specific warrior
    // For more granular control
}
```

### Option 3: Hybrid Approach (Best of Both)

Keep `Run()` for console app, add `StepCycle()` for visualizer:

```csharp
public class Mars
{
    // Existing synchronous API (unchanged)
    public int? Run() { /* existing code */ }
    
    // New step-based API for visualizer
    public ExecutionResult StepCycle()
    {
        if (_currentCycle >= _maxCycles)
            return ExecutionResult.MaxCyclesReached;
            
        // Execute one instruction per warrior
        for (var i = 0; i < _processQueues.Length; i++)
        {
            if (_processQueues[i].Count == 0) continue;
            var pc = _processQueues[i].Dequeue();
            var nextPc = ExecuteInstruction(i, pc);
            if (nextPc.HasValue)
                _processQueues[i].Enqueue(nextPc.Value);
        }
        
        _currentCycle++;
        
        // Check win conditions
        return DetermineResult();
    }
    
    // Read-only state access
    public IReadOnlyList<int> GetProcessQueue(int warriorIndex) { }
    public Instruction[] GetMemorySnapshot() { }
}
```

## Visualization Design

### Memory Grid Visualization

**Grid Layout:**
- Core size: 8000 cells
- Display as square grid: 89x90 (closest to square for 8000)
- Alternative: 100x80 for widescreen
- Cell size: 8x8 pixels = 712x720 window (add margins for stats)

**Color Coding:**
- Empty/DAT: Dark gray (#1a1a1a)
- Warrior 0 code: Red (#ff4444)
- Warrior 1 code: Blue (#4444ff)
- Current execution: Bright yellow (#ffff00)
- Recent writes: Fade from white → warrior color over 10 cycles
- Process positions: Small bright dots

**Optimizations:**
- Only redraw changed cells (dirty rectangle tracking)
- Use render texture for static grid
- Batch draw calls for cells

### Statistics Panel

**Left Panel (200px wide):**
```
╔═══════════════════╗
║ CORE WARS         ║
║ Cycle: 1234       ║
║ Speed: 60 Hz      ║
╠═══════════════════╣
║ DWARF             ║
║ Color: [■] Red    ║
║ Processes: 1      ║
║ Code Size: 4      ║
║ Start: 1234       ║
╠═══════════════════╣
║ IMP               ║
║ Color: [■] Blue   ║
║ Processes: 128    ║
║ Code Size: 1      ║
║ Start: 5678       ║
╠═══════════════════╣
║ [SPACE] Pause     ║
║ [+/-] Speed       ║
║ [R] Reset         ║
║ [ESC] Quit        ║
╚═══════════════════╝
```

### Controls

**Keyboard:**
- `SPACE` - Pause/Resume
- `+` / `-` - Increase/Decrease speed (1x, 2x, 5x, 10x, 60x, 600x)
- `S` - Step forward one cycle (when paused)
- `R` - Reset and restart battle
- `ESC` - Exit to menu
- `F11` - Fullscreen toggle

**Mouse:**
- Hover over cell - Show instruction details in tooltip
- Click cell - Highlight cell and show full instruction data

## Implementation Phases

### Phase 1: Core Library Extensions (1-2 hours) ✅ COMPLETE
- [x] Add `StepCycle()` to Mars
- [x] Add `ExecutionResult` enum
- [x] Add read-only state access methods (`GetProcessQueue()`, `MaxCycles`, `WarriorCount`)
- [x] Extract `DetermineResult()` helper
- [x] Add memory access via `Core` property → `CoreMemory.Read(address)` and `CoreMemory.Size`
- [x] Add `GetProcessQueue(int warriorIndex)` for visualization

**Implementation Notes:**
- `StepCycle()` executes one cycle and returns `ExecutionResult`
- `Run()` refactored to use `StepCycle()` internally (zero duplication)
- `GetProcessQueue()` returns `IReadOnlyCollection<int>` with process positions
- Memory accessible via `mars.Core.Read(address)` and `mars.Core.Size`
- See [refactoring-summary.md](refactoring-summary.md) for details

### Phase 2: Raylib Project Setup (1 hour) ✅ COMPLETE (integrated into Console project)
- [x] ~~Create CoreTrainer.Visualizer project~~ Integrated into CoreTrainer.Console
- [x] Add Raylib-cs package (7.0.0)
- [x] Create basic window with test rendering (1200x800, 60 FPS)
- [x] ~~Set up project structure (folders)~~ Will split when visualizer grows
- [x] ~~Add project reference to CoreTrainer.Core~~ Already referenced

**Implementation Notes:**
- Raylib integrated directly into CoreTrainer.Console project for simplicity
- Command-line flag `--headless` selects between GUI and console mode
- Basic render loop implemented in `RunGuiMode()` with placeholder text
- Window configuration: 1200x800, 60 FPS target, white background
- Can refactor to separate project later if needed
- See [command-line-implementation.md](command-line-implementation.md) for details

### Phase 3: Basic Rendering (2-3 hours)
- [ ] Implement CoreRenderer with grid drawing
- [ ] Color-code cells by owner
- [ ] Implement cell-to-screen coordinate conversion
- [ ] Add camera for pan/zoom (future)
- [ ] Test with static memory snapshot

### Phase 4: Simulation Integration (2-3 hours)
- [ ] Create SimulationController
- [ ] Connect Mars.StepCycle() to frame loop
- [ ] Implement pause/resume
- [ ] Implement variable speed control
- [ ] Track "dirty" cells for optimized rendering

### Phase 5: Visual Polish (2-3 hours)
- [ ] Add process position indicators (glowing dots)
- [ ] Implement write fade-out effect
- [ ] Add execution trail (last N executed instructions)
- [ ] Statistics panel rendering
- [ ] Input handling (keyboard/mouse)

### Phase 6: Advanced Features (future)
- [ ] Cell hover tooltip with instruction details
- [ ] Minimap for large cores
- [ ] Playback controls (scrub timeline)
- [ ] Save/load battle replays
- [ ] Export battle as video/GIF
- [ ] Multiple visualization modes (heatmap, activity graph)

## Technical Challenges

### Challenge 1: Performance at High Speed
**Problem:** 80,000 cycles might take seconds at 60 FPS (1333 seconds if 1 cycle/frame)
**Solutions:**
- Multiple cycles per frame at high speeds (10x = 10 cycles/frame)
- Skip rendering some frames entirely at max speed
- Provide speed presets: 1x, 10x, 100x, 1000x, "Max"

### Challenge 2: Memory Visualization at 8000 Cells
**Problem:** 8000 cells is large for detailed visualization
**Solutions:**
- Zoom/pan camera controls
- Minimap showing overall activity
- Focus mode that follows active warrior processes
- Heat map mode showing write frequency

### Challenge 3: Smooth Visual Transitions
**Problem:** Instant state changes look jarring
**Solutions:**
- Fade effects for cell color changes (write tracking)
- Particle effects for process execution
- Trail effect for moving processes
- Interpolation between frames (not needed for grid-based)

### Challenge 4: Keeping Core Library Clean
**Problem:** Don't want to pollute Mars with visualization code
**Solutions:**
- Minimal API surface: just StepCycle() and state access
- No visualization logic in Core library
- All rendering concerns in Visualizer project
- Use read-only interfaces for state access

## Raylib Package Requirements

```xml
<PackageReference Include="Raylib-cs" Version="6.2.1" />
```

## Window Configuration

```csharp
// Initial window setup
const int windowWidth = 1200;
const int windowHeight = 800;
const int targetFPS = 60;

Raylib.InitWindow(windowWidth, windowHeight, "Core Wars Visualizer");
Raylib.SetTargetFPS(targetFPS);

// Grid configuration
const int gridCols = 100;
const int gridRows = 80;
const int cellSize = 8;
const int gridOffsetX = 220; // After stats panel
const int gridOffsetY = 10;
```

## Color Scheme

```csharp
public static class ColorScheme
{
    // Background
    public static readonly Color Background = new(20, 20, 20, 255);
    public static readonly Color GridLine = new(40, 40, 40, 255);
    
    // Memory cells
    public static readonly Color Empty = new(26, 26, 26, 255);
    public static readonly Color Warrior0 = new(255, 68, 68, 255);    // Red
    public static readonly Color Warrior1 = new(68, 68, 255, 255);    // Blue
    
    // Execution
    public static readonly Color CurrentExecution = new(255, 255, 0, 255);  // Yellow
    public static readonly Color ProcessMarker = new(255, 255, 255, 200);   // White, semi-transparent
    
    // UI
    public static readonly Color PanelBackground = new(30, 30, 30, 255);
    public static readonly Color TextPrimary = new(220, 220, 220, 255);
    public static readonly Color TextSecondary = new(140, 140, 140, 255);
}
```

## State Tracking for Rendering

```csharp
public class SimulationState
{
    public int CurrentCycle { get; set; }
    public int[] CellOwners { get; set; }              // warriorIndex per cell (-1 = empty)
    public int[] CellWriteCycles { get; set; }         // Last write cycle per cell
    public List<int>[] ProcessQueues { get; set; }     // Process positions per warrior
    public (int warrior, int pc)? LastExecution { get; set; }
}
```

## Next Steps

1. **Decision Point:** Which execution model (Event, Step, or Hybrid)?
   - **Recommendation:** Hybrid - keep Run() intact, add StepCycle()

2. **MVP Scope:** What's the minimal viable visualizer?
   - Static grid with color-coded cells
   - Basic play/pause/speed controls
   - Cycle counter and process counts
   - No fade effects, no tooltips initially

3. **Testing Strategy:**
   - Unit tests for coordinate conversions
   - Integration tests for SimulationController
   - Manual testing for visual quality
   - Performance testing at different speeds

4. **Documentation:**
   - Update CLAUDE.md with visualizer architecture
   - Add keyboard controls reference
   - Document color scheme in README
   - Add screenshots/GIFs to showcase

## Questions to Resolve

1. Should Mars expose internal state directly or through read-only interfaces?
   - **Recommendation:** Add read-only properties/methods to avoid exposing internal collections

2. Should visualizer run in same process or separate?
   - **Recommendation:** Same process initially; separate for distributed battles later

3. Should we support replay/time-scrubbing in MVP?
   - **Recommendation:** No - add in Phase 6 after core visualization works

4. Grid size: 100x80 or 89x90?
   - **Recommendation:** 100x80 for widescreen, wastes 0 cells (100*80 = 8000 exactly!)

5. Should we support configurable core sizes or hardcode 8000?
   - **Recommendation:** Make grid dynamic based on Core.Size, but optimize for 8000
