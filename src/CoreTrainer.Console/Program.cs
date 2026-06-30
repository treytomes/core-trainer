using System.CommandLine;
using System.Reactive.Linq;
using CoreTrainer.Core;
using CoreTrainer.Graphics;
using CoreTrainer.Graphics.Rendering;
using CoreTrainer.Graphics.Simulation;
using CoreTrainer.Graphics.UI;
using CoreTrainer.Networking.Nand;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;
using NetEscapades.Configuration.Yaml;
using Raylib_cs;
using Serilog;
using Spectre.Console;

namespace CoreTrainer.Console;

public static class Program
{
	public static int Main(string[] args)
	{
		// Configure Serilog early for startup logging
		Log.Logger = new LoggerConfiguration()
			.WriteTo.Console()
			.CreateBootstrapLogger();

		try
		{
			Log.Information("Starting Core Wars Simulator");

			// Create command line interface
			var rootCommand = new RootCommand("Core Wars Simulator - MARS (Memory Array Redcode Simulator)");

			var headlessOption = new Option<bool>(
				name: "--headless",
				description: "Run in headless mode (console battle simulation without GUI)",
				getDefaultValue: () => false);

			var battleOption = new Option<FileInfo?>(
				name: "--battle",
				description: "Path to a battle roster file (defaults to battle.yaml beside the executable)",
				getDefaultValue: () => null);

			var trainOption = new Option<bool>(
				name: "--train",
				description: "Train a NAND-network warrior against the built-in opponents and save it",
				getDefaultValue: () => false);

			var outputOption = new Option<FileInfo?>(
				name: "--output",
				description: "Where to save the trained genome (defaults to trained-warrior.nand)",
				getDefaultValue: () => null);

			rootCommand.AddOption(headlessOption);
			rootCommand.AddOption(battleOption);
			rootCommand.AddOption(trainOption);
			rootCommand.AddOption(outputOption);

			rootCommand.SetHandler((headless, battleFile, train, outputFile) =>
			{
				var host = CreateHostBuilder(args, battleFile).Build();

				if (train)
				{
					Log.Information("Running in training mode");
					RunTraining(host.Services, outputFile);
				}
				else if (headless)
				{
					Log.Information("Running in headless mode");
					RunHeadlessBattle(host.Services);
				}
				else
				{
					Log.Information("Starting GUI mode");
					RunGuiMode(host.Services);
				}

				Log.Information("Core Wars Simulator completed successfully");
			}, headlessOption, battleOption, trainOption, outputOption);

			return rootCommand.Invoke(args);
		}
		catch (Exception ex)
		{
			Log.Fatal(ex, "Application terminated unexpectedly");
			return 1;
		}
		finally
		{
			Log.CloseAndFlush();
		}
	}

	private static IHostBuilder CreateHostBuilder(string[] args, FileInfo? battleFile) =>
		Host.CreateDefaultBuilder(args)
			.UseSerilog((context, services, configuration) => configuration
				.ReadFrom.Configuration(context.Configuration)
				.ReadFrom.Services(services)
				.Enrich.FromLogContext())
			.ConfigureAppConfiguration((context, config) =>
			{
				config.SetBasePath(AppContext.BaseDirectory);
				config.AddYamlFile("appsettings.yaml", optional: false, reloadOnChange: true);
				config.AddYamlFile("colorscheme.yaml", optional: true, reloadOnChange: true);

				// The battle roster: the user-supplied file if given (required to exist),
				// otherwise the default battle.yaml beside the executable.
				if (battleFile is not null)
				{
					config.AddYamlFile(battleFile.FullName, optional: false, reloadOnChange: false);
				}
				else
				{
					config.AddYamlFile("battle.yaml", optional: true, reloadOnChange: true);
				}
			})
			.ConfigureServices((context, services) =>
			{
				// Configure options from appsettings.yaml
				services.Configure<CoreWarsOptions>(context.Configuration.GetSection("CoreWars"));
				services.Configure<WindowOptions>(context.Configuration.GetSection("Window"));
				services.Configure<TrainingOptions>(context.Configuration.GetSection("Training"));

				// Bind the color scheme (colorscheme.yaml is optional; falls back to defaults)
				var colorSchemeData = context.Configuration.GetSection("ColorScheme").Get<ColorSchemeData>() ?? new ColorSchemeData();
				services.AddSingleton(new ColorScheme(colorSchemeData));

				// Bind the battle roster from its section.
				var battleConfig = context.Configuration.GetSection("Battle").Get<BattleConfig>() ?? new BattleConfig();
				services.AddSingleton(battleConfig);

				// Register services
				services.AddSingleton<IWarriorFactory, WarriorFactory>();
				services.AddSingleton<BattleService>();
				services.AddSingleton<INandGenomeSerializer, TextNandGenomeSerializer>();
			});

	/// <summary>
	/// Builds the combatant roster from the battle configuration, dispatching on each
	/// entry's <see cref="WarriorKind"/>. Falls back to a default Dwarf-vs-Imp matchup
	/// when the roster is empty.
	/// </summary>
	private static List<BattleEntry> BuildRoster(
		BattleConfig battleConfig,
		IWarriorFactory factory,
		INandGenomeSerializer genomeSerializer,
		CoreWarsOptions coreWarsOptions)
	{
		var entries = battleConfig.Warriors;
		if (entries.Count == 0)
		{
			Log.Warning("Battle roster is empty; falling back to a default Dwarf vs Imp matchup");
			return [new BattleEntry(factory.CreateDwarf()), new BattleEntry(factory.CreateImp())];
		}

		var roster = new List<BattleEntry>(entries.Count);
		foreach (var entry in entries)
		{
			roster.Add(entry.Kind switch
			{
				WarriorKind.Archetype => BuildArchetypeEntry(entry, factory),
				WarriorKind.Nand => BuildNandEntry(entry, genomeSerializer, coreWarsOptions),
				_ => throw new InvalidOperationException($"Unknown warrior kind '{entry.Kind}'.")
			});
		}

		return roster;
	}

	/// <summary>
	/// Builds a standard archetype combatant (no brain).
	/// </summary>
	private static BattleEntry BuildArchetypeEntry(BattleWarrior entry, IWarriorFactory factory)
	{
		var warrior = factory.Create(entry.Archetype);
		if (!string.IsNullOrWhiteSpace(entry.Name))
		{
			warrior = new Warrior(entry.Name, warrior.Instructions);
		}
		return new BattleEntry(warrior);
	}

	/// <summary>
	/// Builds a NAND-network combatant: loads the genome and wraps it in a brain whose
	/// observation codec matches the current core parameters. The genome's input width
	/// must match this codec (it does when trained with the same settings); a mismatch
	/// surfaces as a clear error from <see cref="NandBrain"/>.
	/// </summary>
	private static BattleEntry BuildNandEntry(
		BattleWarrior entry,
		INandGenomeSerializer genomeSerializer,
		CoreWarsOptions coreWarsOptions)
	{
		if (string.IsNullOrWhiteSpace(entry.GenomeFile))
			throw new InvalidOperationException("A NAND warrior entry must specify a GenomeFile.");

		var path = Path.IsPathRooted(entry.GenomeFile)
			? entry.GenomeFile
			: Path.Combine(AppContext.BaseDirectory, entry.GenomeFile);

		var genome = genomeSerializer.LoadFromFile(path);
		var codec = new WarriorObservationCodec(
			InferWindowRadius(genome, coreWarsOptions),
			coreWarsOptions.CoreSize,
			coreWarsOptions.MaxProcesses);
		var brain = new NandBrain(new NandNetwork(genome), codec);

		// A brain-driven warrior is seeded with a single NOP; the brain overwrites it each step.
		var name = string.IsNullOrWhiteSpace(entry.Name) ? "Nand" : entry.Name;
		var warrior = new Warrior(name, [new Instruction(OpCode.NOP, AddressingMode.Direct, 0, AddressingMode.Direct, 0)]);
		return new BattleEntry(warrior, brain);
	}

	/// <summary>
	/// Recovers the window radius a genome was trained with from its input width and the
	/// current core parameters, so the battle-time codec matches the training-time codec.
	/// </summary>
	private static int InferWindowRadius(NandGenome genome, CoreWarsOptions coreWarsOptions)
	{
		// Try increasing radii until the codec's input width matches the genome's. This is
		// robust to the exact PC/process-count field sizing without duplicating that math.
		for (var radius = 0; radius <= 64; radius++)
		{
			var codec = new WarriorObservationCodec(radius, coreWarsOptions.CoreSize, coreWarsOptions.MaxProcesses);
			if (codec.InputBitCount == genome.InputCount)
				return radius;
		}

		throw new InvalidOperationException(
			$"Could not match genome input width ({genome.InputCount}) to any window radius for the " +
			$"current core (CoreSize={coreWarsOptions.CoreSize}, MaxProcesses={coreWarsOptions.MaxProcesses}). " +
			"The genome was likely trained with different core settings.");
	}

	private static void RunHeadlessBattle(IServiceProvider services)
	{
		AnsiConsole.Write(
			new FigletText("Core Wars")
				.LeftJustified()
				.Color(Spectre.Console.Color.Red));

		AnsiConsole.MarkupLine("[bold cyan]MARS Simulator[/] [dim]v1.0[/]\n");

		var battleService = services.GetRequiredService<BattleService>();
		var warriorFactory = services.GetRequiredService<IWarriorFactory>();
		var battleConfig = services.GetRequiredService<BattleConfig>();
		var genomeSerializer = services.GetRequiredService<INandGenomeSerializer>();
		var coreWarsOptions = services.GetRequiredService<IOptions<CoreWarsOptions>>().Value;

		// Build the roster from the battle configuration
		var roster = BuildRoster(battleConfig, warriorFactory, genomeSerializer, coreWarsOptions);
		AnsiConsole.MarkupLine($"[dim]Roster ({roster.Count}):[/] {string.Join(", ", roster.Select(e => e.Warrior.Name))}\n");

		// Run the battle with visual feedback
		var winner = AnsiConsole.Status()
			.Spinner(Spinner.Known.Dots)
			.SpinnerStyle(Style.Parse("yellow"))
			.Start("[yellow]Initializing battle...[/]", ctx =>
			{
				ctx.Status("[yellow]Loading warriors...[/]");
				ctx.Status("[yellow]Starting battle simulation...[/]");
				return battleService.RunBattle(roster);
			});

		// Display result with styling
		AnsiConsole.WriteLine();
		var resultPanel = winner.HasValue
			? new Panel($"[bold green]Winner:[/] [yellow]{roster[winner.Value].Warrior.Name}[/]")
				.Border(BoxBorder.Double)
				.BorderColor(Spectre.Console.Color.Green)
			: new Panel("[bold yellow]Result: Draw[/]")
				.Border(BoxBorder.Double)
				.BorderColor(Spectre.Console.Color.Yellow);

		AnsiConsole.Write(resultPanel);
	}

	private static void RunTraining(IServiceProvider services, FileInfo? outputFile)
	{
		var coreWarsOptions = services.GetRequiredService<IOptions<CoreWarsOptions>>().Value;
		var trainingOptions = services.GetRequiredService<IOptions<TrainingOptions>>().Value;
		var warriorFactory = services.GetRequiredService<IWarriorFactory>();
		var serializer = services.GetRequiredService<INandGenomeSerializer>();

		AnsiConsole.Write(
			new FigletText("Train")
				.LeftJustified()
				.Color(Spectre.Console.Color.Aqua));

		// Deterministic opponents the warrior is scored against.
		var opponents = new[] { warriorFactory.CreateDwarf(), warriorFactory.CreateImp() };

		// Codec sizes the network's input/output; the genome must match.
		var codec = new WarriorObservationCodec(
			trainingOptions.WindowRadius, coreWarsOptions.CoreSize, coreWarsOptions.MaxProcesses);

		var random = new Random();
		var fitness = new BattleFitness(
			opponents, codec, random,
			coreWarsOptions.CoreSize, trainingOptions.MaxCycles, coreWarsOptions.MaxProcesses,
			trainingOptions.BattlesPerOpponent);

		var initial = NandGenome.CreateRandom(
			codec.InputBitCount, trainingOptions.NodeCount, codec.OutputBitCount, random);

		AnsiConsole.MarkupLine(
			$"[dim]Network:[/] {codec.InputBitCount} inputs, {trainingOptions.NodeCount} nodes, {codec.OutputBitCount} outputs");
		AnsiConsole.MarkupLine(
			$"[dim]Strategy:[/] 1+{trainingOptions.Lambda}, {trainingOptions.Generations} generations, " +
			$"mutation {trainingOptions.MutationRate:P1}, {trainingOptions.BattlesPerOpponent} battles/opponent\n");

		var evolution = new CgpEvolution(
			fitness.Evaluate, trainingOptions.Lambda, trainingOptions.MutationRate, random);

		EvolutionResult result = null!;
		AnsiConsole.Progress()
			.Columns(
				new TaskDescriptionColumn(),
				new ProgressBarColumn(),
				new PercentageColumn(),
				new ElapsedTimeColumn())
			.Start(ctx =>
			{
				var task = ctx.AddTask("[green]Evolving[/]", maxValue: trainingOptions.Generations);
				result = evolution.Evolve(
					initial,
					trainingOptions.Generations,
					onGeneration: (generation, bestFitness) =>
					{
						task.Increment(1);
						task.Description = $"[green]Evolving[/] [dim]gen {generation + 1}, fitness {bestFitness:F1}[/]";
					});
			});

		// Persist the winner.
		var path = outputFile?.FullName ?? Path.Combine(AppContext.BaseDirectory, "trained-warrior.nand");
		serializer.SaveToFile(result.Best, path);

		AnsiConsole.WriteLine();
		AnsiConsole.Write(new Panel(
				$"[bold green]Training complete[/]\nFinal fitness: [yellow]{result.Fitness:F1}[/]\nSaved to: [aqua]{path}[/]")
			.Border(BoxBorder.Double)
			.BorderColor(Spectre.Console.Color.Green));

		Log.Information("Training complete: fitness {Fitness:F1}, saved to {Path}", result.Fitness, path);
	}

	private static void RunGuiMode(IServiceProvider services)
	{
		var windowOptions = services.GetRequiredService<IOptions<WindowOptions>>().Value;
		var coreWarsOptions = services.GetRequiredService<IOptions<CoreWarsOptions>>().Value;
		var colorScheme = services.GetRequiredService<ColorScheme>();
		var warriorFactory = services.GetRequiredService<IWarriorFactory>();
		var battleConfig = services.GetRequiredService<BattleConfig>();
		var genomeSerializer = services.GetRequiredService<INandGenomeSerializer>();

		// Build the roster from the battle configuration and a matching color palette.
		var roster = BuildRoster(battleConfig, warriorFactory, genomeSerializer, coreWarsOptions);
		var random = new Random();
		var palette = new WarriorPalette(roster.Count, random);

		Raylib.InitWindow(windowOptions.Width, windowOptions.Height, windowOptions.Title);
		Raylib.SetTargetFPS(windowOptions.TargetFPS);
		Raylib.SetExitKey(KeyboardKey.Null); // Disable ESC to exit

		Log.Information("Raylib window initialized: {Width}x{Height}", windowOptions.Width, windowOptions.Height);

		// Create simulation controller
		var simController = new SimulationController();

		// Create the core grid renderer, sized to the configured core so no address
		// can map outside the grid, and positioned below the toolbar.
		var gridConfig = GridConfig.ForCoreSize(coreWarsOptions.CoreSize, offsetY: 80);
		var coreRenderer = new CoreRenderer(colorScheme, palette, gridConfig);

		// Loads the roster at freshly-chosen separated positions. Reused by Reset.
		void InitializeBattle()
		{
			var positions = WarriorPlacement.GetSeparatedPositions(
				coreWarsOptions.CoreSize, roster.Count, coreWarsOptions.MinWarriorSeparation, random);
			simController.Initialize(
				roster, positions,
				coreWarsOptions.CoreSize, coreWarsOptions.MaxCycles, coreWarsOptions.MaxProcesses);
			Log.Information("Battle initialized with {Count} warriors: {Names}",
				roster.Count, string.Join(", ", roster.Select(e => e.Warrior.Name)));
		}

		// Build the UI tree. The root carries the theme; all descendants inherit it.
		var root = new RootContainer(windowOptions.Width, windowOptions.Height, colorScheme);

		// Control toolbar
		var toolbar = root.Add(new Toolbar(10, 10, windowOptions.Width - 130, 60));
		var stepButton = toolbar.AddButton("Step");
		var runPauseButton = toolbar.AddButton("Run");
		var resetButton = toolbar.AddButton("Reset");
		var speedDownButton = toolbar.AddButton("<<", 50);
		var speedLabel = toolbar.AddLabel("1/s", 70);
		// Give the speed readout a panel look so it reads as a field, not floating text.
		speedLabel.BackgroundColor = colorScheme.PanelBackground;
		speedLabel.BorderColor = colorScheme.GridLine;
		var speedUpButton = toolbar.AddButton(">>", 50);

		// Close button in top-right corner
		var closeButton = root.Add(new Button(
			x: windowOptions.Width - 110,
			y: 10,
			width: 100,
			height: 40,
			text: "Close"
		));

		var shouldClose = false;

		// Initialize the battle from the roster
		InitializeBattle();

		// Subscribe to button click events
		var subscriptions = new List<IDisposable>
		{
			stepButton.Clicked.Subscribe(_ =>
			{
				Log.Information("Step button clicked");
				simController.Step();
			}),

			runPauseButton.Clicked.Subscribe(_ =>
			{
				simController.TogglePause();
				runPauseButton.Text = simController.State == SimulationState.Running ? "Pause" : "Run";
				Log.Information("{Action} button clicked", runPauseButton.Text);
			}),

			resetButton.Clicked.Subscribe(_ =>
			{
				Log.Information("Reset button clicked");
				simController.Reset();
				InitializeBattle();
				runPauseButton.Text = "Run";
			}),

			speedDownButton.Clicked.Subscribe(_ =>
			{
				simController.DecreaseSpeed();
				speedLabel.Text = simController.GetSpeedDisplay();
				Log.Information("Speed decreased to {Speed}", speedLabel.Text);
			}),

			speedUpButton.Clicked.Subscribe(_ =>
			{
				simController.IncreaseSpeed();
				speedLabel.Text = simController.GetSpeedDisplay();
				Log.Information("Speed increased to {Speed}", speedLabel.Text);
			}),

			closeButton.Clicked.Subscribe(_ =>
			{
				Log.Information("Close button clicked");
				shouldClose = true;
			})
		};

		while (!shouldClose && !Raylib.WindowShouldClose())
		{
			var deltaTime = Raylib.GetFrameTime();

			// Update simulation
			simController.Update(deltaTime);

			// Update the UI tree (detects clicks and emits events)
			root.Update();

			// Rendering
			Raylib.BeginDrawing();
			Raylib.ClearBackground(colorScheme.Background);

			// Draw the core memory grid (below the UI, outside the control tree)
			if (simController.Mars != null)
			{
				coreRenderer.Render(simController.Mars);
				coreRenderer.RenderProcessMarkers(simController.Mars);
			}

			// Draw status information
			var statusY = 90;
			var statusColor = Raylib_cs.Color.White;

			if (simController.Mars != null)
			{
				var mars = simController.Mars;
				Raylib.DrawText($"Cycle: {mars.CurrentCycle}", 20, statusY, 20, statusColor);
				Raylib.DrawText($"State: {simController.State}", 20, statusY + 30, 20, statusColor);

				// Per-warrior process counts, each labeled in its palette color.
				var lineY = statusY + 60;
				for (var i = 0; i < roster.Count; i++)
				{
					var processes = mars.GetProcessQueue(i).Count;
					Raylib.DrawText($"{roster[i].Warrior.Name}: {processes}", 20, lineY, 18, palette.GetColor(i));
					lineY += 24;
				}

				if (simController.State == SimulationState.Finished)
				{
					var result = simController.LastResult switch
					{
						ExecutionResult.Win => $"{roster[mars.Winner].Warrior.Name} Wins!",
						ExecutionResult.Draw => "Draw",
						ExecutionResult.MaxCyclesReached => "Time Limit Reached",
						_ => "Unknown"
					};
					Raylib.DrawText($"Result: {result}", 20, lineY + 10, 24, Raylib_cs.Color.Yellow);
				}
			}

			// Render the UI tree on top of the grid
			root.Render();

			Raylib.EndDrawing();
		}

		// Clean up subscriptions
		foreach (var subscription in subscriptions)
		{
			subscription.Dispose();
		}

		Raylib.CloseWindow();
		Log.Information("Raylib window closed");
	}
}
