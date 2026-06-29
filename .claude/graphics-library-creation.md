# CoreTrainer.Graphics Library Creation

## Summary

Created a dedicated graphics library project to hold common visualization components shared between the console app and future visualizer implementations.

## Project Structure

```
src/CoreTrainer.Graphics/
├── CoreTrainer.Graphics.csproj    # Project file
├── ColorScheme.cs                 # Color definitions
├── GridConfig.cs                  # Grid layout and coordinate math
└── Rendering/
    └── CoreRenderer.cs            # Core memory grid renderer
```

## Files Created

### [CoreTrainer.Graphics.csproj](../../src/CoreTrainer.Graphics/CoreTrainer.Graphics.csproj)

**Package References:**
- `Raylib-cs` (7.0.0) - Graphics library
- Project reference to `CoreTrainer.Core`

**Properties:**
- Target Framework: .NET 10.0
- ImplicitUsings: Enabled
- Nullable: Enabled

### [ColorScheme.cs](../../src/CoreTrainer.Graphics/ColorScheme.cs)

Static class defining all colors for the visualizer:

**Background Colors:**
- `Background` - Main background (20, 20, 20)
- `GridLine` - Grid lines (40, 40, 40)

**Memory Cell Colors:**
- `Empty` - Empty/DAT cells (26, 26, 26)
- `Warrior0` - Warrior 0 code (255, 68, 68) - Red
- `Warrior1` - Warrior 1 code (68, 68, 255) - Blue

**Execution Indicators:**
- `CurrentExecution` - Active instruction (255, 255, 0) - Yellow
- `ProcessMarker` - Process positions (255, 255, 255, 200) - White, semi-transparent

**UI Colors:**
- `PanelBackground` - Side panels (30, 30, 30)
- `TextPrimary` - Main text (220, 220, 220)
- `TextSecondary` - Secondary text (140, 140, 140)

**Helper Methods:**
- `GetWarriorColor(int index)` - Returns color for warrior by index
- `Fade(Color color, float amount)` - Creates faded version with adjusted alpha

### [GridConfig.cs](../../src/CoreTrainer.Graphics/GridConfig.cs)

Configuration class for grid layout with coordinate conversion:

**Constants:**
- `DefaultColumns = 100` - Perfect fit for 8000 cells
- `DefaultRows = 80`
- `DefaultCellSize = 8` pixels
- `DefaultOffsetX = 220` - Leave space for stats panel
- `DefaultOffsetY = 10`

**Properties:**
- `Columns`, `Rows`, `CellSize`, `OffsetX`, `OffsetY`
- `TotalWidth`, `TotalHeight`, `TotalCells` - Computed properties

**Coordinate Conversion Methods:**

```csharp
// Address to grid coordinates
(int col, int row) AddressToGrid(int address)

// Grid coordinates to address
int GridToAddress(int col, int row)

// Grid coordinates to screen pixels
(int x, int y) GridToScreen(int col, int row)

// Address directly to screen pixels
(int x, int y) AddressToScreen(int address)

// Screen pixels to grid (mouse interaction)
(int col, int row)? ScreenToGrid(int screenX, int screenY)
```

### [Rendering/CoreRenderer.cs](../../src/CoreTrainer.Graphics/Rendering/CoreRenderer.cs)

Main renderer for core memory visualization:

**Constructor:**
```csharp
public CoreRenderer(GridConfig? gridConfig = null)
```
Optional grid config, uses defaults if not provided.

**Public Methods:**

1. **`Render(Mars mars, int warrior0Start, int warrior1Start)`**
   - Renders entire 8000-cell memory grid
   - Color-codes cells by ownership
   - Draws grid lines for visual separation
   
2. **`RenderProcessMarkers(Mars mars)`**
   - Draws bright dots at active process positions
   - Color-coded by warrior
   - Centered in cells with 1/4 cell radius

**Private Methods:**

1. **`DetermineCellColor(address, instruction, w0Start, w1Start)`**
   - Detects cell ownership based on proximity to warrior start positions
   - Returns appropriate color (Warrior0, Warrior1, or Empty)
   - TODO: Proper ownership tracking instead of proximity heuristic

2. **`CircularDistance(addr1, addr2, coreSize)`**
   - Calculates true distance in circular address space
   - Used for ownership detection

3. **`DrawGridLines()`**
   - Draws vertical and horizontal grid lines
   - Optional, can be disabled for performance

## Project References

**Console project updated:**
- Added `<ProjectReference Include="..\CoreTrainer.Graphics\CoreTrainer.Graphics.csproj" />`

## Design Decisions

### Why separate Graphics library?

1. **Separation of Concerns**
   - Core library: Simulation logic (no dependencies)
   - Graphics library: Visualization (depends on Core + Raylib)
   - Console app: Composition root

2. **Reusability**
   - Can be used by multiple visualization projects
   - Could support different rendering backends later
   - Shared by headless and GUI modes

3. **Testability**
   - Can unit test coordinate math independently
   - Can test color schemes without full app
   - Clear boundaries between layers

4. **Future-Proofing**
   - Easy to add more renderers (minimap, heatmap)
   - Can extract interfaces if needed
   - Natural place for visualization utilities

### Grid Configuration Design

**Why 100×80 grid?**
- 100 × 80 = 8000 cells exactly (perfect fit)
- Widescreen friendly (16:10 aspect ratio)
- 8×8 pixel cells = 800×640 grid (fits in 1200×800 window)

**Why init properties instead of readonly?**
- Allows customization via object initializer
- Maintains immutability after construction
- Clean syntax: `new GridConfig { CellSize = 10 }`

**Why computed properties for totals?**
- Always consistent with cell size/count
- No risk of stale cached values
- Minimal performance impact (simple multiplication)

### Color Scheme Design

**Why static class?**
- Colors are global constants
- No state to manage
- Easy access: `ColorScheme.Warrior0`
- No need for instantiation

**Why Raylib Color struct?**
- Native to Raylib API
- No conversion overhead
- RGBA byte values (0-255)

**Why helper methods?**
- `GetWarriorColor()` - Extensible to N warriors
- `Fade()` - Useful for write fade effects, hover states

### Renderer Design

**Why pass Mars directly?**
- Renderer needs full state access
- Avoids large DTO objects
- Trusts Mars API (read-only access)

**Why pass warrior start addresses?**
- Current ownership detection is proximity-based
- Need starting points for heuristic
- TODO: Better tracking in future (write history)

**Ownership Detection Limitations:**
- Current: Proximity to start address
- Problem: Doesn't track written cells
- Solution (future): Maintain owner array per cell

## Build Status

✅ **0 warnings, 0 errors**  
✅ **All projects build successfully**  
✅ **Project references resolved**

## Usage Example

```csharp
// In GUI mode
var gridConfig = new GridConfig();
var renderer = new CoreRenderer(gridConfig);

// In render loop
Raylib.BeginDrawing();
Raylib.ClearBackground(ColorScheme.Background);

renderer.Render(mars, warrior0Start, warrior1Start);
renderer.RenderProcessMarkers(mars);

Raylib.EndDrawing();
```

## Next Steps

**Immediate:**
1. Update `Program.cs` GUI mode to use `CoreRenderer`
2. Pass warrior start positions to renderer
3. Test static rendering with loaded warriors

**Short-term:**
4. Add proper cell ownership tracking (write history)
5. Implement write fade effect
6. Add statistics panel renderer
7. Add input handling for keyboard/mouse

**Future:**
8. Extract `IRenderer` interface for different render modes
9. Add heatmap renderer (write frequency)
10. Add minimap renderer (zoomed-out view)
11. Add replay renderer (saved battle data)

## Testing Considerations

**Unit Tests (future):**
- GridConfig coordinate conversions
- ColorScheme helper methods
- Ownership detection logic

**Integration Tests:**
- Renderer with Mars simulation
- Full render loop performance
- Grid size variations

## Documentation Updates Needed

- [x] Update CLAUDE.md with Graphics project structure
- [ ] Update README.md with architecture diagram
- [ ] Add code examples to visualizer-design.md
- [ ] Document color scheme choices

## Performance Considerations

**Current Implementation:**
- Renders all 8000 cells every frame
- Grid lines drawn every frame
- No optimization (dirty rectangles, culling)

**Expected Performance:**
- 8000 small rectangles + 181 lines per frame
- Should achieve 60 FPS easily on modern hardware
- Raylib batches draw calls efficiently

**Future Optimizations (if needed):**
- Dirty rectangle tracking (only redraw changed cells)
- Grid lines as render texture (draw once, reuse)
- Culling (skip off-screen cells if zoomed)
- Texture atlas for cell patterns
