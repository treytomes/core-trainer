# Core Trainer - C# Coding Standards

## General Principles

- **Clarity over cleverness** - Write code that is easy to understand and maintain
- **Consistency** - Follow established patterns in the codebase
- **Immutability preferred** - Use `record` types and `readonly` where possible
- **Explicit is better than implicit** - Be clear about intent

## File Organization

### One Type Per File

**Each file should contain exactly one type (class, struct, record, interface, or enum).**

- File name must match the type name exactly (e.g., `Button.cs` contains `Button` class)
- Exception: Nested types can be in the same file as their containing type
- Exception: Small helper types (e.g., internal records < 5 lines) closely tied to the main type
- Enums related to a single class can be in separate files or together based on size

**Example - Correct:**
```
Button.cs         → class Button
Toolbar.cs        → class Toolbar
SimulationState.cs → enum SimulationState (if large)
```

**Example - Acceptable:**
```
SimulationController.cs → class SimulationController + enum SimulationState (if enum is small and only used by controller)
```

**Rationale:**
- Easier to find types (predictable file names)
- Better IDE navigation and search
- Clearer git history per type
- Easier to review changes
- Prevents "god files" with multiple related types

### Namespace and Using Statements

```csharp
namespace CoreTrainer.Core;

public class Example
{
    // Class content
}
```

- Use file-scoped namespaces (no braces)
- Place using statements inside namespace when needed for local scope
- Order: System namespaces first, then third-party, then local

### Class Structure

Order members in this sequence:

1. **Constructors** (when using primary constructors, this is on the type declaration line)
2. **Fields** (private, readonly preferred)
   - Group in `#region Fields` ... `#endregion`
3. **Properties** (public, then internal, then private)
   - Group in `#region Properties` ... `#endregion` when you have 3+
4. **Public methods**
   - Group in `#region Methods` ... `#endregion` when you have 5+
5. **Internal methods**
6. **Private methods**

#### Using #region Directives

Use `#region` / `#endregion` to group related members for better organization:

- **Always use regions for**: Fields (if any), Properties (when 3+), Constructors (when 3+), Methods (when 5+)
- **Region naming**: Use plural nouns (`Fields`, `Properties`, `Constructors`, `Methods`)
- **Region benefits**: Collapsible sections in IDE, clear visual separation, easier navigation

```csharp
public class Example
{
    #region Fields
    
    private readonly int _size;
    private readonly string _name;
    
    #endregion
    
    #region Properties
    
    public int Size => _size;
    public string Name => _name;
    
    #endregion
    
    #region Constructors
    
    public Example(int size, string name)
    {
        _size = size;
        _name = name;
    }
    
    #endregion
    
    #region Methods
    
    public void PublicMethod() { }
    
    private void PrivateMethod() { }
    
    #endregion
}
```

#### Primary Constructors (C# 12+)

Use primary constructors for simple classes/records with straightforward initialization:

```csharp
// Good - primary constructor for simple initialization
public class Warrior(string name, Instruction[] instructions)
{
    public string Name { get; } = name;
    public Instruction[] Instructions { get; } = instructions;
    public int StartAddress { get; set; }
}

// Also good - primary constructor with direct property assignment
public record Point(int X, int Y);

// Avoid primary constructors when:
// - You need validation logic
// - You have multiple constructors
// - Initialization is complex

// Better with traditional constructor when you need validation
public class CoreMemory
{
    #region Fields
    
    private readonly Instruction[] _memory;
    private readonly int _size;
    
    #endregion
    
    public int Size => _size;
    
    public CoreMemory(int size = 8000)
    {
        if (size <= 0)
            throw new ArgumentException("Size must be positive", nameof(size));
            
        _size = size;
        _memory = new Instruction[size];
        
        // Complex initialization
        for (int i = 0; i < size; i++)
        {
            _memory[i] = new Instruction();
        }
    }
}
```

**Primary Constructor Guidelines:**
- ✅ Use for simple property assignment
- ✅ Use for records with positional parameters
- ✅ Use when no validation is needed
- ❌ Avoid when you have multiple constructors
- ❌ Avoid when initialization requires logic beyond simple assignment
- ❌ Avoid when validation is needed

## Naming Conventions

### General Rules

- **PascalCase**: Classes, methods, properties, enums, enum values
- **camelCase**: Parameters, local variables
- **_camelCase**: Private fields (with underscore prefix)
- **ALL_CAPS**: Constants (sparingly - prefer `const` or `static readonly`)

### Examples

```csharp
public class CoreMemory                    // PascalCase
{
    private readonly int _size;            // _camelCase for fields
    
    public int Size => _size;              // PascalCase for properties
    
    public void Write(int address)         // PascalCase for methods, camelCase for params
    {
        int normalizedAddr = Normalize(address);  // camelCase for locals
    }
}

public enum OpCode                         // PascalCase
{
    DAT,                                   // PascalCase
    MOV,
    ADD
}
```

## Type Choices

### Use Records for Immutable Data

Use `record` types for data transfer objects and value-like types:

```csharp
public record Instruction
{
    public OpCode OpCode { get; init; }
    public AddressingMode AMode { get; init; }
    public int AValue { get; init; }
    
    public Instruction() { }
    
    public Instruction(OpCode opCode, AddressingMode aMode, int aValue)
    {
        OpCode = opCode;
        AMode = aMode;
        AValue = aValue;
    }
}
```

Benefits:
- Immutable by default (`init` properties)
- Value-based equality
- `with` expressions for non-destructive mutations
- Concise syntax

### Use Classes for Behavior-Heavy Types

Use `class` for types with complex behavior, mutable state, or reference semantics:

```csharp
public class Mars
{
    private readonly CoreMemory _core;
    private int _currentCycle;
    
    public void Run() { /* complex logic */ }
}
```

### Use Enums for Fixed Sets

```csharp
public enum OpCode
{
    /// <summary>
    /// Data - Kills the process
    /// </summary>
    DAT = 0,
    
    /// <summary>
    /// Move - Copies data from A to B
    /// </summary>
    MOV,
}
```

- Always document enum values with XML comments
- Explicitly set first value to 0 when order matters
- Use singular names (OpCode not OpCodes)

## Region Directive Usage

### When to Use Regions

Use `#region` directives to organize and group related code members:

| Member Type | Use Region When | Region Name |
|------------|-----------------|-------------|
| Fields | Always (if any exist) | `Fields` |
| Properties | 3 or more properties | `Properties` |
| Constructors | 2 or more constructors | `Constructors` |
| Methods | 5 or more methods | `Methods` |
| Nested Types | Any nested classes/enums | Descriptive name |

### Region Organization Examples

#### Record with Regions (from Instruction.cs)

```csharp
public record Instruction
{
    #region Constructors
    
    public Instruction()
    {
        OpCode = OpCode.DAT;
        AMode = AddressingMode.Direct;
        AValue = 0;
        BMode = AddressingMode.Direct;
        BValue = 0;
    }
    
    public Instruction(OpCode opCode, AddressingMode aMode, int aValue, AddressingMode bMode, int bValue)
    {
        OpCode = opCode;
        AMode = aMode;
        AValue = aValue;
        BMode = bMode;
        BValue = bValue;
    }
    
    #endregion
    
    #region Properties
    
    public OpCode OpCode { get; init; }
    public AddressingMode AMode { get; init; }
    public int AValue { get; init; }
    public AddressingMode BMode { get; init; }
    public int BValue { get; init; }
    
    #endregion
    
    #region Methods
    
    public override string ToString()
    {
        // ... implementation
    }
    
    #endregion
}
```

#### Class with Regions (from CoreMemory.cs)

```csharp
public class CoreMemory
{
    #region Fields
    
    private readonly Instruction[] _memory;
    private readonly int _size;
    
    #endregion
    
    public int Size => _size;  // Single property - no region needed
    
    public CoreMemory(int size = 8000)
    {
        _size = size;
        _memory = new Instruction[size];
        
        for (int i = 0; i < size; i++)
        {
            _memory[i] = new Instruction();
        }
    }
    
    public Instruction Read(int address) { }
    public void Write(int address, Instruction instruction) { }
    public int Normalize(int address) { }
    public void Load(int startAddress, IEnumerable<Instruction> instructions) { }
    
    // 4 public methods - no region needed (threshold is 5+)
}
```

### Region Best Practices

**DO:**
- Use descriptive, plural names (`Fields`, `Properties`, `Methods`)
- Keep regions flat (avoid nested regions)
- Use blank lines before and after region content
- Collapse regions when not actively editing them

**DON'T:**
- Create single-item regions (unless it's a field region)
- Use overly specific region names (`Public Methods`, `Private Methods` - just use `Methods`)
- Nest regions inside other regions
- Create regions for unrelated code groupings

### Small Classes Don't Need Regions

If your class is small enough to fit on one screen (< 50 lines), skip regions entirely:

```csharp
public class Warrior
{
    public string Name { get; }
    public Instruction[] Instructions { get; }
    public int StartAddress { get; set; }
    
    public Warrior(string name, Instruction[] instructions)
    {
        Name = name;
        Instructions = instructions;
    }
}
```

## Documentation

### XML Documentation Comments

**Required for:**
- Public classes, methods, properties
- Enums and enum values
- Non-obvious behavior

```csharp
/// <summary>
/// Represents the core memory arena where warriors battle.
/// </summary>
public class CoreMemory
{
    /// <summary>
    /// Normalizes an address to wrap around the core memory boundaries.
    /// </summary>
    /// <param name="address">The address to normalize.</param>
    /// <returns>An address between 0 and Size-1.</returns>
    public int Normalize(int address)
    {
        return ((address % _size) + _size) % _size;
    }
}
```

**Not required for:**
- Private members (unless complex)
- Self-explanatory getters/setters
- Override methods where behavior is inherited

### Inline Comments

Use sparingly and only when code cannot be made self-documenting:

```csharp
// Wrap around the core memory
return ((address % _size) + _size) % _size;
```

**When to comment:**
- Non-obvious algorithms
- Workarounds for bugs or limitations
- Performance-critical sections
- Complex business logic

**Don't comment:**
- What the code does (make it self-evident)
- Commented-out code (use version control)
- Redundant explanations

## Code Style

### Type Inference with `var`

**Use `var` for local variables whenever the type is obvious from the right-hand side.**

This improves maintainability and makes refactoring easier. The IDE shows the type on hover, so readability is not lost.

```csharp
// ✅ Good - type is obvious
var memory = new CoreMemory(8000);
var instructions = new List<Instruction>();
var winner = mars.Run();
var normalizedAddr = Normalize(address);
var source = _core.Read(aAddr);

// ✅ Also good - LINQ and factory methods
var activeWarriors = warriors.Where(w => w.IsActive).ToList();
var config = ConfigBuilder.Create();

// ❌ Avoid - explicit type is redundant
CoreMemory memory = new CoreMemory(8000);
List<Instruction> instructions = new List<Instruction>();

// ⚠️ Use explicit type when clarity is needed
int count = GetCount();           // Not obvious what GetCount() returns
IEnumerable<int> numbers = source; // Interface vs concrete type matters
```

**Benefits of using `var`:**
- Reduces visual noise and line length
- Easier refactoring (change type on right side, left side updates automatically)
- Focuses attention on variable name and initialization
- Consistent with modern C# practices

**When to use explicit types:**
- When the type is not clear from the right-hand side
- When working with interfaces and the specific interface matters
- When dealing with numeric literals where precision matters (`decimal` vs `double`)
- In public method signatures (never use `var` in method signatures)

### Properties

Use expression-bodied members for simple properties:

```csharp
// Good
public int Size => _size;

// Avoid
public int Size { get { return _size; } }
```

### Methods

Expression-bodied for single expressions:

```csharp
public int Normalize(int address) => ((address % _size) + _size) % _size;
```

Block body for multiple statements:

```csharp
public void Load(int startAddress, IEnumerable<Instruction> instructions)
{
    int address = Normalize(startAddress);
    foreach (var instruction in instructions)
    {
        _memory[address] = instruction;
        address = Normalize(address + 1);
    }
}
```

### Switch Expressions vs Statements

Prefer switch expressions for mapping values:

```csharp
// Good - switch expression
var modeStr = AMode switch
{
    AddressingMode.Immediate => "#",
    AddressingMode.Indirect => "@",
    AddressingMode.Decrement => "<",
    _ => "$"
};

// Use switch statement for complex logic with side effects
switch (instruction.OpCode)
{
    case OpCode.MOV:
        {
            int aAddr = ResolveAddress(pc, instruction.AMode, instruction.AValue);
            _core.Write(aAddr, source);
            return _core.Normalize(pc + 1);
        }
}
```

### Braces in Switch Statements

Use braces for case blocks with local variables:

```csharp
case OpCode.MOV:
    {
        int aAddr = ResolveAddress(pc, instruction.AMode, instruction.AValue);
        int bAddr = ResolveAddress(pc, instruction.BMode, instruction.BValue);
        // ...
        return _core.Normalize(pc + 1);
    }
```

### Null Handling

Use nullable reference types:

```csharp
public int? Run()  // Can return null
{
    if (condition)
        return null;
    return 0;
}
```

### String Interpolation

Prefer string interpolation over concatenation:

```csharp
// Good
Console.WriteLine($"Loading {dwarf.Name} at position {warrior1Pos}");

// Avoid
Console.WriteLine("Loading " + dwarf.Name + " at position " + warrior1Pos);
```

## Testing Standards

### Test Project Structure

- Mirror source structure: `CoreTrainer.Core` → `CoreTrainer.Core.Tests`
- Use xUnit framework
- Name test classes: `{ClassUnderTest}Tests`
- Name test methods: `{MethodName}_{Scenario}_{ExpectedResult}`

### Test Method Pattern

```csharp
[Fact]
public void Normalize_NegativeAddress_WrapsToPositive()
{
    // Arrange
    var memory = new CoreMemory(size: 100);
    
    // Act
    int result = memory.Normalize(-5);
    
    // Assert
    Assert.Equal(95, result);
}
```

**AAA Pattern:**
1. **Arrange** - Set up test data and dependencies
2. **Act** - Execute the method under test
3. **Assert** - Verify the expected outcome

## Project Standards

### Target Framework

- Use latest stable .NET version (currently .NET 10)
- No multi-targeting unless required

### Dependencies

- Minimize external dependencies
- Prefer framework features over third-party libraries
- Document reason for each dependency

### Project Structure

```
core-trainer/
├── src/
│   ├── CoreTrainer.Core/         # Core library
│   └── CoreTrainer.Console/      # Console application
├── tests/
│   ├── CoreTrainer.Core.Tests/
│   └── CoreTrainer.Console.Tests/
├── .claude/
│   ├── journal/
│   ├── skills/
│   └── coding-standards.md
├── build.bat
├── run.bat
└── README.md
```

## Git Commit Standards

### Commit Message Format

```
<type>: <subject>

<optional body>

<optional footer>
```

**Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

**Examples:**
```
feat: add SLT opcode implementation

Implements the Skip if Less Than instruction for Core Wars simulator.

fix: correct indirect addressing calculation in Mars

The indirect addressing mode was not properly dereferencing pointers.
```

## Code Review Checklist

Before committing code, verify:

- [ ] All public APIs have XML documentation
- [ ] No compiler warnings
- [ ] Code follows naming conventions
- [ ] No commented-out code
- [ ] Tests added for new functionality
- [ ] README updated if needed
- [ ] Region tags used appropriately for grouping

## Performance Guidelines

- Use `readonly` for fields that won't change after construction
- Prefer `for` over `foreach` for performance-critical loops with arrays
- Use `stackalloc` for small, temporary buffers (careful with size)
- Profile before optimizing - clarity first, performance when needed

## When in Doubt

1. Look at existing code in the project for patterns
2. Prioritize readability and maintainability
3. Ask for review on architectural decisions
4. Update this document when new patterns emerge
