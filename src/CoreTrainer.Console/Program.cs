using System.CommandLine;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
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
		const int windowWidth = 1200;
		const int windowHeight = 800;

		Raylib.InitWindow(windowWidth, windowHeight, "Core Wars Visualizer");
		Raylib.SetTargetFPS(60);

		Log.Information("Raylib window initialized: {Width}x{Height}", windowWidth, windowHeight);

		while (!Raylib.WindowShouldClose())
		{
			Raylib.BeginDrawing();
			Raylib.ClearBackground(Raylib_cs.Color.RayWhite);

			// Draw simple placeholder text
			Raylib.DrawText("Core Wars Visualizer", 400, 300, 40, Raylib_cs.Color.Black);
			Raylib.DrawText("Press ESC to exit", 480, 360, 20, Raylib_cs.Color.DarkGray);
			Raylib.DrawText("Visualization coming soon...", 430, 400, 20, Raylib_cs.Color.Gray);

			Raylib.EndDrawing();
		}

		Raylib.CloseWindow();
		Log.Information("Raylib window closed");
	}
}
