# Core Wars Simulator

A minimally-viable Core Wars (Redcode) simulator built in .NET 10.

## What is Core Wars?

Core Wars is a programming game where two or more programs (called "warriors") battle in a shared memory space (the "core"). Warriors are written in a simple assembly-like language called Redcode. The goal is to cause the opponent's processes to terminate while keeping your own running.

## Project Structure

- **CoreTrainer.Core** - Core library containing the MARS engine and Redcode implementation
- **CoreTrainer.Console** - Console application demonstrating two warriors battling

## Core Components

### Instruction Set (OpCodes)

The simulator implements these Redcode opcodes:

- **DAT** - Data (kills the executing process)
- **MOV** - Copy from source to destination
- **ADD** - Add source to destination
- **SUB** - Subtract source from destination
- **JMP** - Jump to address
- **JMZ** - Jump if zero
- **JMN** - Jump if not zero
- **DJN** - Decrement and jump if not zero
- **CMP** - Compare and skip next if equal
- **SPL** - Split (create new process)
- **NOP** - No operation

### Addressing Modes

- **Immediate (#)** - Use the value directly
- **Direct ($)** - Use the address offset
- **Indirect (@)** - Use the value at address as pointer
- **Decrement (<)** - Decrement, then use as pointer

### MARS (Memory Array Redcode Simulator)

The execution engine that:
- Manages an 8000-cell core memory
- Maintains process queues for each warrior
- Executes instructions in round-robin fashion
- Determines winners when opponent processes die

## Running the Demo

### Windows (Batch Scripts)

```batch
# Build only
build.bat

# Run only (builds automatically if needed)
run.bat

# Build and run
build-and-run.bat
```

### Cross-platform (.NET CLI)

```bash
# Build the solution
dotnet build

# Run the simulator
dotnet run --project src/CoreTrainer.Console/CoreTrainer.Console.csproj
```

The demo pits two classic warriors against each other:

1. **Dwarf** - A "bomber" that writes DAT instructions every 4 locations
2. **Imp** - A one-instruction warrior that copies itself forward continuously

## Example Warriors

### Imp
```redcode
MOV $0, $1  ; Copy self to next location and execute it
```

### Dwarf
```redcode
ADD #4, $3    ; Add 4 to bombing pointer
MOV $2, @$2   ; Copy DAT bomb to target
JMP $-2       ; Loop back
DAT #0, #0    ; The bomb
```

## Building from Source

### Windows
```batch
build.bat
```

### Cross-platform
```bash
dotnet build

# Run tests (when added)
dotnet test
```

## Next Steps

This is a minimal implementation. Possible enhancements:

- Parse Redcode from text files
- Add visualization of core memory
- Implement more opcodes (MUL, DIV, MOD, SLT)
- Support more than 2 warriors
- Add tournament mode
- Create a warrior library with classic programs
- Add step-by-step debugging

## References

- [Core Wars Official Site](https://corewar.co.uk/)
- [Redcode Reference](https://corewar.co.uk/icws94.txt)
