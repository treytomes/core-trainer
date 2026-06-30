# Core Wars Simulator

A Core Wars (Redcode) simulator built in .NET 10, featuring a real-time graphical
visualizer and **trainable NAND-network warriors** evolved with Cartesian Genetic
Programming.

## What is Core Wars?

Core Wars is a programming game where two or more programs (called "warriors") battle in
a shared memory space (the "core"). Warriors are written in a simple assembly-like
language called Redcode. The goal is to cause the opponents' processes to terminate while
keeping your own running.

This project extends the classic game with a twist: alongside hand-written warriors, you
can **train neural warriors** whose "brain" is an evolved network of NAND gates. Standard
and trained warriors fight on the same field.

## Project Structure

- **CoreTrainer.Core** — the MARS engine, Redcode instructions, warrior placement, the
  warrior-brain hook, and the instruction/observation codecs. No external dependencies.
- **CoreTrainer.Graphics** — the Raylib-based visualizer: core grid renderer, themeable
  UI controls (visual tree), and the simulation controller.
- **CoreTrainer.Networking.Nand** — the trainable warrior: NAND gates, a Cartesian
  Genetic Programming genome, network evaluation, the CGP evolution strategy, battle-based
  fitness, and genome serialization.
- **CoreTrainer.Console** — the application: command-line interface, GUI and headless
  battle modes, training mode, and configuration.
- **CoreTrainer.Tests** — xUnit test suite covering all of the above.

## Core Components

### Instruction Set (16 OpCodes)

- **DAT** — Data (kills the executing process)
- **MOV** — Copy from source to destination
- **ADD** — Add source to destination
- **SUB** — Subtract source from destination
- **MUL** — Multiply destination by source
- **DIV** — Divide destination by source (division by zero kills the process)
- **MOD** — Destination modulo source (modulo by zero kills the process)
- **JMP** — Jump to address
- **JMZ** — Jump if zero
- **JMN** — Jump if not zero
- **DJN** — Decrement and jump if not zero
- **CMP** — Compare and skip next if equal
- **SNE** — Skip next if not equal (ICWS-94)
- **SLT** — Skip next if less than
- **SPL** — Split (create new process)
- **NOP** — No operation

The 16 opcodes fill a 4-bit field exactly, which matters for NAND warriors (see below).

### Addressing Modes

- **Immediate (#)** — Use the value directly
- **Direct ($)** — Use the address offset
- **Indirect (@)** — Use the value at address as pointer
- **Decrement (<)** — Decrement, then use as pointer

### MARS (Memory Array Redcode Simulator)

The execution engine:
- Manages an 8000-cell circular core memory (configurable)
- Supports **any number of warriors** in one battle
- Maintains a process queue per warrior, executed round-robin
- Tracks **cell ownership** — which warrior last wrote each cell (drives the visualizer's
  coloring)
- Determines the winner when only one warrior's processes survive (a draw if all die, or
  the cycle limit is reached)

## Architecture

The application uses Microsoft.Extensions.Hosting with dependency injection and Serilog
for structured logging.

### Key Technologies

- **Microsoft.Extensions.Hosting** — application host with dependency injection
- **Serilog** — structured logging to console and rolling files
- **Spectre.Console** — rich console UI (headless and training modes)
- **Raylib-cs** — 2D graphics library (GUI mode)
- **System.CommandLine** — command-line argument parsing
- **System.Reactive** — observable UI events (button clicks)

### Configuration

Settings live in [appsettings.yaml](src/CoreTrainer.Console/appsettings.yaml):

```yaml
CoreWars:
  CoreSize: 8000
  MaxCycles: 80000
  MaxProcesses: 8000
  MinWarriorSeparation: 100

Window:
  Width: 1200
  Height: 800
  Title: "Core Wars Visualizer"
  TargetFPS: 60

Training:
  WindowRadius: 1        # cells observed on each side of the program counter
  NodeCount: 200         # NAND gates per candidate network
  Generations: 200
  Lambda: 4              # offspring per generation (1+lambda strategy)
  MutationRate: 0.03
  BattlesPerOpponent: 3
  MaxCycles: 8000        # per training battle (smaller = faster training)
```

Colors are themeable via [colorscheme.yaml](src/CoreTrainer.Console/colorscheme.yaml),
and the battle roster is defined in [battle.yaml](src/CoreTrainer.Console/battle.yaml)
(see below).

### Logging

Serilog logs to:
- **Console** — structured output with timestamps
- **Rolling files** — `logs/corewar-YYYYMMDD.log` (retained for 7 days)

## Building and Running

### Build

```bash
# Windows
build.bat

# Cross-platform
dotnet build core-trainer.slnx
```

### Test

```bash
# Windows
test.bat

# Cross-platform
dotnet test core-trainer.slnx
```

### GUI Mode (default)

Launches the Raylib visualizer: a live grid of the core colored by cell ownership, a
control toolbar (Step, Run/Pause, Reset, speed adjustment), and per-warrior status.

```bash
# Windows
run.bat

# Cross-platform
dotnet run --project src/CoreTrainer.Console/CoreTrainer.Console.csproj
```

### Headless Mode

Runs a battle in the console with Spectre.Console output (no window):

```bash
# Windows
run.bat --headless

# Cross-platform
dotnet run --project src/CoreTrainer.Console/CoreTrainer.Console.csproj -- --headless
```

## Defining a Battle Roster

The warriors in a battle are listed in [battle.yaml](src/CoreTrainer.Console/battle.yaml).
Each entry has a `Kind` discriminator selecting which fields apply. Standard and NAND
warriors can be mixed freely.

```yaml
Battle:
  Warriors:
    - Kind: Archetype      # a built-in program
      Archetype: Dwarf     # Dwarf or Imp
      Name: Classic Dwarf

    - Kind: Nand           # a trained NAND warrior
      GenomeFile: trained-warrior.nand   # path relative to the executable
      Name: Evolved Brain

    - Kind: Archetype
      Archetype: Imp
      Name: Classic Imp
```

Use a different roster file with `--battle`:

```bash
dotnet run --project src/CoreTrainer.Console/CoreTrainer.Console.csproj -- --battle my-battle.yaml
# or headless:
dotnet run --project src/CoreTrainer.Console/CoreTrainer.Console.csproj -- --headless --battle my-battle.yaml
```

## Training a NAND Warrior

A NAND warrior has no fixed program. Instead, each cycle its **brain** — an evolved
network of NAND gates — observes the cells around its program counter plus its own state
(program counter, process count) and emits the instruction to execute. The instruction
encoding is designed so that *any* network output is a valid instruction, so an evolved
warrior can never crash the simulator.

The brain is evolved with **Cartesian Genetic Programming** using a 1+λ evolution
strategy. Fitness comes from running real battles against the built-in warriors (Dwarf
and Imp), with a shaped signal: a win scores highest, but survival time and territory
held also count, so even early, losing genomes have a gradient to climb.

### Train

```bash
# Trains using the Training settings in appsettings.yaml, saving the result.
dotnet run --project src/CoreTrainer.Console/CoreTrainer.Console.csproj -- --train

# Choose where to save the trained genome:
dotnet run --project src/CoreTrainer.Console/CoreTrainer.Console.csproj -- --train --output my-warrior.nand
```

Training shows a live progress bar with the best fitness per generation, then writes the
winning genome to the output file (default `trained-warrior.nand` beside the executable).
A fitness above ~1000 means the warrior is winning battles.

Tune the run via the `Training` section of `appsettings.yaml` — more generations and a
larger `NodeCount` give stronger warriors at the cost of time.

### Use the trained warrior

Reference the saved genome from a battle roster with `Kind: Nand`, then run a battle:

```yaml
Battle:
  Warriors:
    - Kind: Nand
      GenomeFile: trained-warrior.nand
      Name: My Champion
    - Kind: Archetype
      Archetype: Dwarf
```

```bash
dotnet run --project src/CoreTrainer.Console/CoreTrainer.Console.csproj -- --headless --battle my-battle.yaml
```

> **Note:** a NAND genome is tied to the core parameters it was trained with (core size,
> max processes, window radius). Train and run with the same `CoreWars` settings, or
> loading the warrior will report a mismatch.

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

## How NAND Warriors Work (in brief)

1. **NandGate** — a single-bit NAND, functionally complete (any boolean function can be
   built from NANDs).
2. **NandGenome** — a Cartesian GP genome: a grid of NAND gates wired with backwards-only
   addressing, which makes cycles structurally impossible. Inactive gates drift under
   mutation, enabling the neutral search that helps CGP escape local optima.
3. **NandNetwork** — decodes a genome and evaluates it for a given input, computing only
   the gates that feed an output.
4. **Codec** — packs a warrior's observation (window of cells + state) into the network's
   input bits, and decodes the output bits into a guaranteed-valid instruction.
5. **NandBrain** — implements the engine's brain hook, tying the network and codec to a
   warrior in MARS.
6. **CgpEvolution + BattleFitness** — the 1+λ search, scored by real battles.

Trained genomes are saved in a compact, versioned text format and can be reloaded into
any battle.

## References

- [Core Wars Official Site](https://corewar.co.uk/)
- [ICWS '94 Redcode Standard](https://corewar.co.uk/icws94.txt)
- [Cartesian Genetic Programming](https://www.cartesiangp.com/)
