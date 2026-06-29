# Command Line and GUI Mode Implementation

## Summary

Implemented System.CommandLine and Raylib-cs to support both headless (console) and GUI modes.

## Changes Made

### Package Additions

**[CoreTrainer.Console.csproj](../../src/CoreTrainer.Console/CoreTrainer.Console.csproj)**
- `Raylib-cs` (7.0.0) - Graphics library for visualizer
- `System.CommandLine` (2.0.0-beta4.22272.1) - Command-line parsing

### Program Structure Refactoring

**[Program.cs](../../src/CoreTrainer.Console/Program.cs)**

**New Command-Line Interface:**
```csharp
var rootCommand = new RootCommand("Core Wars Simulator - MARS");
var headlessOption = new Option<bool>("--headless", 
    description: "Run in headless mode (console battle without GUI)",
    getDefaultValue: () => false);
```

**Mode Selection:**
- No arguments → GUI mode (default)
- `--headless` → Console mode with Spectre.Console

**Refactored Methods:**

1. **`Main(string[] args)` → int**
   - Changed return type from void to int for proper exit codes
   - Sets up System.CommandLine root command
   - Delegates to `RunHeadlessBattle()` or `RunGuiMode()` based on arguments

2. **`RunHeadlessBattle(IServiceProvider)`** (renamed from `RunBattle`)
   - Runs console-based battle simulation
   - Uses Spectre.Console for rich output (FigletText, spinners, panels)
   - Fixed Color namespace conflicts with fully-qualified names

3. **`RunGuiMode(IServiceProvider)`** (new)
   - Initializes Raylib window (1200x800, 60 FPS)
   - Simple render loop with placeholder text
   - Waits for ESC key or window close
   - Logs initialization and shutdown

**Color Namespace Resolution:**
- Both Spectre.Console and Raylib_cs define `Color` type
- Fixed by fully qualifying: `Spectre.Console.Color.Red`
- Keeps code explicit and avoids ambiguity

## Usage Examples

### GUI Mode (Default)
```bash
# Launch visualizer window
dotnet run --project src/CoreTrainer.Console/CoreTrainer.Console.csproj

# Or using batch file
run.bat
```

Output:
- Opens 1200x800 Raylib window
- Shows placeholder text
- Waits for user to close window

### Headless Mode
```bash
# Run console simulation
dotnet run --project src/CoreTrainer.Console/CoreTrainer.Console.csproj -- --headless

# Or using batch file
run.bat --headless
```

Output:
- ASCII art "Core Wars" banner
- Animated spinner during battle
- Colored result panel showing winner/draw

### Help
```bash
dotnet run --project src/CoreTrainer.Console/CoreTrainer.Console.csproj -- --help
```

Output:
```
Description:
  Core Wars Simulator - MARS (Memory Array Redcode Simulator)

Usage:
  CoreTrainer.Console [options]

Options:
  --headless       Run in headless mode (console battle simulation without GUI) [default: False]
  --version        Show version information
  -?, -h, --help   Show help and usage information
```

## Technical Details

### Raylib Window Configuration
```csharp
const int windowWidth = 1200;
const int windowHeight = 800;
Raylib.InitWindow(windowWidth, windowHeight, "Core Wars Visualizer");
Raylib.SetTargetFPS(60);
```

### Render Loop
```csharp
while (!Raylib.WindowShouldClose())
{
    Raylib.BeginDrawing();
    Raylib.ClearBackground(Raylib_cs.Color.RayWhite);
    
    // Drawing code here
    
    Raylib.EndDrawing();
}
Raylib.CloseWindow();
```

### System.CommandLine Pattern
```csharp
rootCommand.SetHandler((headless) =>
{
    var host = CreateHostBuilder(args).Build();
    if (headless)
        RunHeadlessBattle(host.Services);
    else
        RunGuiMode(host.Services);
}, headlessOption);

return rootCommand.Invoke(args);
```

## Build Status

✅ **0 warnings, 0 errors** - Clean build  
✅ **Headless mode tested** - Battle runs correctly  
✅ **GUI mode tested** - Window opens and closes properly  
✅ **Exit codes** - Returns 0 on success, 1 on exception

## Dependencies Injected in Both Modes

The `CreateHostBuilder()` method configures services available to both modes:
- `IOptions<CoreWarsOptions>` - Configuration
- `IWarriorFactory` - Warrior creation
- `BattleService` - Battle orchestration
- `ILogger<T>` - Structured logging

GUI mode will use these services when visualizer is fully implemented.

## Next Steps

**Immediate:**
1. ✅ Refactor Mars for step-based execution
2. ✅ Add command-line interface
3. ✅ Create Raylib placeholder window
4. ⬜ Implement core memory grid renderer
5. ⬜ Connect StepCycle() to render loop

**Future Enhancements:**
- Add more command-line options (`--warrior1`, `--warrior2`, `--speed`)
- Load warriors from `.red` files via arguments
- Replay mode: `--replay battle-12345.json`
- Benchmark mode: `--benchmark --rounds 1000`
- Tournament mode: `--tournament warriors/*.red`

## Design Decisions

**Why System.CommandLine over other parsing libraries?**
- Official Microsoft library
- Strongly typed
- Built-in help generation
- Modern async/await support
- Good integration with dependency injection

**Why default to GUI mode instead of headless?**
- Visual feedback is the project's goal
- Users expect graphical output
- Headless is the "special case" (automation, CI/CD)
- Can still use `--headless` when needed

**Why Raylib-cs 7.0.0 instead of 6.2.1?**
- 6.2.1 not available in NuGet
- 7.0.0 is latest stable
- No breaking API changes affecting our usage
- Better .NET 10 compatibility

**Why keep everything in CoreTrainer.Console project?**
- Simpler architecture initially
- Both modes share same services
- Can split into CoreTrainer.Visualizer later if needed
- Reduces project overhead for MVP

## Logging Output

Both modes log to Serilog:

**Headless mode:**
```
[10:33:20 INF] Starting Core Wars Simulator
[10:33:20 INF] Running in headless mode
[10:33:20 INF] Initializing battle between Dwarf and Imp
[10:33:20 INF] Loading Dwarf at position 1919
[10:33:20 INF] Loading Imp at position 4133
[10:33:20 INF] Starting battle (max 80000 cycles, max 8000 processes per warrior)
[10:33:20 INF] Battle finished after 80000 cycles
[10:33:20 INF] Result: Draw or time limit reached
[10:33:20 INF] Core Wars Simulator completed successfully
```

**GUI mode:**
```
[10:33:28 INF] Starting Core Wars Simulator
[10:33:28 INF] Starting GUI mode
[10:33:29 INF] Raylib window initialized: 1200x800
[10:33:35 INF] Raylib window closed
[10:33:35 INF] Core Wars Simulator completed successfully
```

Raylib also outputs detailed initialization info to stdout (OpenGL version, display size, extensions, etc.)

## Code Quality

- **Separation of concerns** - GUI and headless logic separated
- **Dependency injection** - Both modes use same service container
- **Error handling** - Exceptions caught and logged with exit code 1
- **Clean shutdown** - Serilog flush, Raylib cleanup
- **Explicit namespacing** - Resolved Color type conflicts
