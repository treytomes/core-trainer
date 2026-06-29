using System.CommandLine;
using System.Reactive.Linq;
using CoreTrainer.Graphics.Simulation;
using CoreTrainer.Graphics.UI;
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

			rootCommand.AddOption(headlessOption);

			rootCommand.SetHandler((headless) =>
			{
				var host = CreateHostBuilder(args).Build();

				if (headless)
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
			}, headlessOption);

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

	private static IHostBuilder CreateHostBuilder(string[] args) =>
		Host.CreateDefaultBuilder(args)
			.UseSerilog((context, services, configuration) => configuration
				.ReadFrom.Configuration(context.Configuration)
				.ReadFrom.Services(services)
				.Enrich.FromLogContext())
			.ConfigureAppConfiguration((context, config) =>
			{
				config.SetBasePath(AppContext.BaseDirectory);
				config.AddYamlFile("appsettings.yaml", optional: false, reloadOnChange: true);
			})
			.ConfigureServices((context, services) =>
			{
				// Configure options from appsettings.yaml
				services.Configure<CoreWarsOptions>(context.Configuration.GetSection("CoreWars"));
				services.Configure<WindowOptions>(context.Configuration.GetSection("Window"));

				// Register services
				services.AddSingleton<IWarriorFactory, WarriorFactory>();
				services.AddSingleton<BattleService>();
			});

	private static void RunHeadlessBattle(IServiceProvider services)
	{
		AnsiConsole.Write(
			new FigletText("Core Wars")
				.LeftJustified()
				.Color(Spectre.Console.Color.Red));

		AnsiConsole.MarkupLine("[bold cyan]MARS Simulator[/] [dim]v1.0[/]\n");

		var battleService = services.GetRequiredService<BattleService>();
		var warriorFactory = services.GetRequiredService<IWarriorFactory>();

		// Create warriors using factory
		var dwarf = warriorFactory.CreateDwarf();
		var imp = warriorFactory.CreateImp();

		// Run the battle with visual feedback
		var winner = AnsiConsole.Status()
			.Spinner(Spinner.Known.Dots)
			.SpinnerStyle(Style.Parse("yellow"))
			.Start("[yellow]Initializing battle...[/]", ctx =>
			{
				ctx.Status("[yellow]Loading warriors...[/]");
				ctx.Status("[yellow]Starting battle simulation...[/]");
				return battleService.RunBattle(dwarf, imp);
			});

		// Display result with styling
		AnsiConsole.WriteLine();
		var resultPanel = winner.HasValue
			? new Panel($"[bold green]Winner:[/] [yellow]{(winner.Value == 0 ? dwarf.Name : imp.Name)}[/]")
				.Border(BoxBorder.Double)
				.BorderColor(Spectre.Console.Color.Green)
			: new Panel("[bold yellow]Result: Draw[/]")
				.Border(BoxBorder.Double)
				.BorderColor(Spectre.Console.Color.Yellow);

		AnsiConsole.Write(resultPanel);
	}

	private static void RunGuiMode(IServiceProvider services)
	{
		var windowOptions = services.GetRequiredService<IOptions<WindowOptions>>().Value;
		var warriorFactory = services.GetRequiredService<IWarriorFactory>();

		Raylib.InitWindow(windowOptions.Width, windowOptions.Height, windowOptions.Title);
		Raylib.SetTargetFPS(windowOptions.TargetFPS);
		Raylib.SetExitKey(KeyboardKey.Null); // Disable ESC to exit

		Log.Information("Raylib window initialized: {Width}x{Height}", windowOptions.Width, windowOptions.Height);

		// Create simulation controller
		var simController = new SimulationController();

		// Create control toolbar
		var toolbar = new Toolbar(10, 10, windowOptions.Width - 130, 60);
		var stepButton = toolbar.AddButton("Step");
		var runPauseButton = toolbar.AddButton("Run");
		var resetButton = toolbar.AddButton("Reset");
		var speedDownButton = toolbar.AddButton("<<", 50);
		var speedLabel = toolbar.AddLabel("1/s", 70);
		speedLabel.BackgroundColor = Graphics.ColorScheme.Default.PanelBackground;
		speedLabel.BorderColor = Graphics.ColorScheme.Default.GridLine;
		var speedUpButton = toolbar.AddButton(">>", 50);

		// Create close button in top-right corner
		var closeButton = new Button(
			x: windowOptions.Width - 110,
			y: 10,
			width: 100,
			height: 40,
			text: "Close"
		);

		var shouldClose = false;

		// Initialize a battle
		var dwarf = warriorFactory.CreateDwarf();
		var imp = warriorFactory.CreateImp();
		var random = new Random();
		var warrior0Start = random.Next(8000);
		var warrior1Start = (warrior0Start + 4000) % 8000; // Separated

		simController.Initialize(dwarf, imp, warrior0Start, warrior1Start);
		Log.Information("Simulation initialized: Dwarf at {W0}, Imp at {W1}", warrior0Start, warrior1Start);

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
				simController.Initialize(dwarf, imp, warrior0Start, warrior1Start);
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

			// Update all buttons (to detect clicks and emit events)
			toolbar.Update();
			closeButton.Update();

			// Rendering
			Raylib.BeginDrawing();
			Raylib.ClearBackground(Graphics.ColorScheme.Default.Background);

			// Draw status information
			var statusY = 90;
			var statusColor = Raylib_cs.Color.White;

			if (simController.Mars != null)
			{
				Raylib.DrawText($"Cycle: {simController.Mars.CurrentCycle}", 20, statusY, 20, statusColor);
				Raylib.DrawText($"State: {simController.State}", 20, statusY + 30, 20, statusColor);

				var warrior0Processes = simController.Mars.GetProcessQueue(0).Count;
				var warrior1Processes = simController.Mars.GetProcessQueue(1).Count;
				Raylib.DrawText($"Dwarf Processes: {warrior0Processes}", 20, statusY + 60, 20, statusColor);
				Raylib.DrawText($"Imp Processes: {warrior1Processes}", 20, statusY + 90, 20, statusColor);

				if (simController.State == SimulationState.Finished)
				{
					var result = simController.LastResult switch
					{
						Core.ExecutionResult.Warrior0Wins => "Dwarf Wins!",
						Core.ExecutionResult.Warrior1Wins => "Imp Wins!",
						Core.ExecutionResult.Draw => "Draw",
						Core.ExecutionResult.MaxCyclesReached => "Time Limit Reached",
						_ => "Unknown"
					};
					Raylib.DrawText($"Result: {result}", 20, statusY + 120, 24, Raylib_cs.Color.Yellow);
				}
			}

			// Render UI
			toolbar.Render();
			closeButton.Render();

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
