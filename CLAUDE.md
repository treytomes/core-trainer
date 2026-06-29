# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Core Trainer is a .NET 10 Core Wars (Redcode) simulator that allows warrior programs to battle in a shared memory arena. The project implements the MARS (Memory Array Redcode Simulator) engine with support for classic Redcode instructions and addressing modes.

**Current Features:**
- Full MARS execution engine with 16 opcodes (DAT, MOV, ADD, SUB, MUL, DIV, MOD, JMP, JMZ, JMN, DJN, CMP, SNE, SLT, SPL, NOP)
- Four addressing modes (Immediate, Direct, Indirect, Decrement)
- Process queue management for multi-threaded warriors
- Demo console application with classic warriors (Dwarf vs Imp)
- Microsoft.Extensions.Hosting with dependency injection
- Serilog structured logging (console + rolling files)
- YAML-based configuration

## Architecture

### Project Structure

```
core-trainer/
├── src/
│   ├── CoreTrainer.Core/         # Core simulator library (MARS engine)
│   └── CoreTrainer.Console/      # Console demo application
├── tests/                         # (Future) Unit tests
├── .claude/
│   ├── journal/                   # Development journal entries
│   ├── skills/                    # Custom Claude skills
│   └── coding-standards.md        # Detailed coding standards
└── README.md
```

### Key Components

**Core Library (CoreTrainer.Core):**
- **[Instruction](src/CoreTrainer.Core/Instruction.cs)** - Immutable record representing a Redcode instruction
- **[CoreMemory](src/CoreTrainer.Core/CoreMemory.cs)** - 8000-cell memory arena with wraparound addressing
- **[Mars](src/CoreTrainer.Core/Mars.cs)** - Execution engine managing process queues and instruction execution
- **[Warrior](src/CoreTrainer.Core/Warrior.cs)** - Container for warrior programs
- **[OpCode](src/CoreTrainer.Core/OpCode.cs)** - Enum of supported Redcode operations
- **[AddressingMode](src/CoreTrainer.Core/AddressingMode.cs)** - Enum of addressing modes

**Console Application (CoreTrainer.Console):**
- **[BattleService](src/CoreTrainer.Console/BattleService.cs)** - Orchestrates battles with logging
- **[IWarriorFactory](src/CoreTrainer.Console/IWarriorFactory.cs)** / **[WarriorFactory](src/CoreTrainer.Console/WarriorFactory.cs)** - Creates warrior instances
- **[CoreWarsOptions](src/CoreTrainer.Console/CoreWarsOptions.cs)** - Strongly-typed configuration
- **[appsettings.yaml](src/CoreTrainer.Console/appsettings.yaml)** - YAML configuration file

### Design Patterns

- **Immutable data**: Use `record` types with `init` properties for value-like data (e.g., `Instruction`)
- **Behavioral classes**: Use `class` for stateful simulation objects (e.g., `Mars`, `CoreMemory`)
- **Value-based modifications**: Use `with` expressions to create modified copies of records
- **Queue-based execution**: Each warrior maintains a process queue for handling splits (SPL instruction)

## Development Commands

### Windows (Batch Scripts)

```batch
# Build solution
build.bat

# Run simulator
run.bat

# Build and run
build-and-run.bat
```

### Cross-platform (.NET CLI)

```bash
# Build
dotnet build core-trainer.slnx

# Run
dotnet run --project src/CoreTrainer.Console/CoreTrainer.Console.csproj

# Test (when tests exist)
dotnet test
```

## Coding Standards

**See [.claude/coding-standards.md](.claude/coding-standards.md) for complete standards.**

### Quick Reference

- **Namespaces**: Use file-scoped (`namespace CoreTrainer.Core;`)
- **Naming**: PascalCase for public members, _camelCase for fields
- **Type inference**: Use `var` for local variables when type is obvious
- **Documentation**: XML comments required for public APIs and enums
- **Immutability**: Prefer `record` for data, `readonly` for fields
- **Regions**: Group fields (always), properties (3+), constructors (2+), methods (5+)
- **Primary constructors**: Use for simple initialization without validation
- **Expression bodies**: Use for simple properties/methods

### Code Organization

```csharp
namespace CoreTrainer.Core;

public class Example
{
    #region Fields
    private readonly int _size;
    #endregion
    
    #region Properties
    public int Size => _size;
    #endregion
    
    public Example(int size) => _size = size;
    
    public void PublicMethod() { }
    private void PrivateMethod() { }
}
```

## Testing Guidelines

- Use xUnit framework
- Follow AAA pattern (Arrange, Act, Assert)
- Test project naming: `{ProjectName}.Tests`
- Mirror source structure in test projects

## Core Wars Implementation Notes

### Memory Model

- Core size: 8000 cells (configurable)
- Wraparound addressing: `((address % size) + size) % size`
- All cells initialized to `DAT #0, #0`

### Execution Model

- Round-robin: Each warrior executes one instruction per cycle
- Process death: Executing DAT or having empty process queue
- Max cycles: 80,000 (configurable)
- Win condition: Opponent has no active processes

### Addressing Modes

- **Immediate (#)**: Value used directly, no memory access
- **Direct ($)**: Offset from current instruction
- **Indirect (@)**: Follow pointer from offset location
- **Decrement (<)**: Decrement pointer, then follow it

### Instruction Execution

Most instructions:
1. Resolve operand addresses using addressing modes
2. Perform operation
3. Write result if needed
4. Return next program counter (usually `pc + 1`)

Jump instructions return the target address instead of `pc + 1`.

SPL (split) enqueues a new process and continues current process.

## Current Limitations

- No validation of warrior placement (could overlap)
- Warriors must be created programmatically (no Redcode parser)
- DIV/MOD by zero kills the executing process (ICWS-94 behavior, by design)

## Next Steps / Future Features

See [.claude/journal/2026-06-29.md](.claude/journal/2026-06-29.md) for detailed next steps.

**Priority items:**
1. Add Redcode parser for loading warriors from text files
2. Expand the warrior library beyond Dwarf and Imp
3. Explore the trainable NAND-network warrior (CartesianGP) in `CoreTrainer.Networking.Nand`

## Resources

- [Core Wars Official Site](https://corewar.co.uk/)
- [ICWS '94 Redcode Standard](https://corewar.co.uk/icws94.txt)
- [Redcode Tutorial](https://vyznev.net/corewar/guide.html)

## Project Context

This is a learning project to understand:
- Core Wars gameplay and strategy
- Low-level programming concepts (memory, addressing, processes)
- Simulator architecture and virtual machines
- Clean code organization in C#

Keep implementations simple and clear. Avoid over-engineering. Document non-obvious decisions.
