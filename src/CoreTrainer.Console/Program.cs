using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using NetEscapades.Configuration.Yaml;
using Serilog;
using Spectre.Console;

namespace CoreTrainer.Console;

public static class Program
{
	public static void Main(string[] args)
	{
		// Configure Serilog early for startup logging
		Log.Logger = new LoggerConfiguration()
			.WriteTo.Console()
			.CreateBootstrapLogger();

		try
		{
			Log.Information("Starting Core Wars Simulator");

			var host = CreateHostBuilder(args).Build();

			// Run the battle
			RunBattle(host.Services);

			Log.Information("Core Wars Simulator completed successfully");
		}
		catch (Exception ex)
		{
			Log.Fatal(ex, "Application terminated unexpectedly");
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

	private static void RunBattle(IServiceProvider services)
	{
		AnsiConsole.Write(
			new FigletText("Core Wars")
				.LeftJustified()
				.Color(Color.Red));

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
				.BorderColor(Color.Green)
			: new Panel("[bold yellow]Result: Draw[/]")
				.Border(BoxBorder.Double)
				.BorderColor(Color.Yellow);

		AnsiConsole.Write(resultPanel);
	}
}
