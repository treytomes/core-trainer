using CoreTrainer.Examples.LogicMlp;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Serilog;
using Serilog.Extensions.Logging;
using static TorchSharp.torch;
// Disambiguate ILogger (both MS and Serilog define one); the example logs via the MS abstraction.
using ILogger = Microsoft.Extensions.Logging.ILogger;

// ---------------------------------------------------------------------------------------
// LogicMlp — a differentiable LOGIC GATE network, the dual of CoreTrainer.Examples.DiffNand.
//
// DiffNand fixes the gate (always NAND) and learns the WIRING. This fixes the wiring (random,
// frozen) and learns each node's GATE FUNCTION — which of the 16 two-input Boolean gates it is —
// via a softmax over gate types, annealed and hardened with a straight-through estimator. Signals
// are BIPOLAR {-1,+1}, where every gate is an exact multilinear polynomial and a signal can carry a
// sign (argue against a feature), which [0,1] logic cannot. See LogicGateNet for the mechanism.
//
// Runs a DIFFICULTY LADDER of Boolean tasks (see LogicTask), not just the 4-bit adder: a single hard
// task is a poor proof of mechanism, since the adder's carry chain is hard for every method here (CGP
// walls on it) — so its failure alone could not tell "mechanism broken" from "task just hard." The
// ladder starts at the floor (bitwise: depth-1 gate-pick) and climbs, so any failure is diagnostic.
// ---------------------------------------------------------------------------------------

const int EPOCHS = 20000;
const double LEARNING_RATE = 0.01;
const int SOFT_WARMUP = 2000;          // pure-soft epochs before switching on straight-through
const double TEMP_START = 2.0;
const double TEMP_END = 0.05;          // final softmax temperature (sharp, near one-gate)
const int LOG_EVERY = 1000;
const int SEED = 1;

using var loggerFactory = CreateLoggerFactory();
var log = loggerFactory.CreateLogger("LogicMlp");

// The ladder, floor first. Each entry pairs a task with a network depth/width sized to it: bitwise is
// depth-1 logic (shallow suffices), parity is an XOR tree (log-depth), the adder needs a deep carry
// chain. Width gives the fixed random wiring enough pairings to both compute and route signals.
var ladder = new (LogicTask Task, int[] Hidden)[]
{
	(LogicTask.Bitwise("XOR", 4), new[] { 32, 32 }),
	(LogicTask.Parity(4),         new[] { 32, 32, 32 }),
	(LogicTask.Parity(8),         new[] { 64, 64, 64, 64 }),
	(LogicTask.Adder(2),          new[] { 64, 64, 64, 64 }),
	(LogicTask.Adder(4),          new[] { 128, 128, 128, 128, 128, 128, 128, 128 }),
};

// The two architectures under test, each a factory (inputCount, outputCount, hidden, generator) -> net:
//   - "gate-only": LogicGateNet — learns gate function, fixed random wiring (the ladder found routing
//      was its wall).
//   - "wire+gate": JointLogicNet — learns wiring AND gate function (the fusion meant to lift that wall).
// Running the SAME ladder for both makes the comparison direct, rung by rung.
var architectures = new (string Label, Func<int, int, int[], Generator, ILogicNet> Factory)[]
{
	("gate-only", (i, o, h, g) => new LogicGateNet(i, o, h, g)),
	("wire+gate", (i, o, h, g) => new JointLogicNet(i, o, h, g)),
};

var results = new List<(string Arch, string Name, double BitAcc, int Exact, int Rows)>();

foreach (var (label, factory) in architectures)
{
	log.LogInformation("########## Architecture: {Label} ##########", label);
	foreach (var (task, hidden) in ladder)
	{
		var (bitAcc, exact) = TrainTask(label, factory, task, hidden, log);
		results.Add((label, task.Name, bitAcc, exact, task.Rows));
	}
}

// Side-by-side ladder summary: for each task, both architectures' results on one line, so the lift from
// adding learnable wiring is visible rung by rung.
log.LogInformation("=== Ladder summary (both architectures) ===");
foreach (var (task, _) in ladder)
{
	var line = $"  {task.Name,-18}";
	foreach (var (label, _) in architectures)
	{
		var r = results.First(x => x.Arch == label && x.Name == task.Name);
		var verdict = r.Exact == r.Rows ? "SOLVED " : "partial";
		line += $"   {label}: {r.BitAcc,7:P2} {r.Exact,3}/{r.Rows,-3} {verdict}";
	}
	log.LogInformation("{Line}", line);
}

return;

// === Training ==========================================================================

// Trains one task's network end to end (soft warmup -> annealed straight-through), logging progress,
// and returns the final hardened-netlist accuracy. Also prints the cullability report per task.
static (double BitAcc, int Exact) TrainTask(
	string arch, Func<int, int, int[], Generator, ILogicNet> factory, LogicTask task, int[] hidden, ILogger log)
{
	// Fresh, deterministic generator per task so the ladder is reproducible and tasks don't couple.
	manual_seed(SEED);
	var generator = random.manual_seed(SEED);

	log.LogInformation("=== [{Arch}] Task: {Name} ===", arch, task.Name);
	log.LogInformation("Network:  {Inputs} inputs, layers [{Layers}], {Outputs} outputs",
		task.InputCount, string.Join(", ", hidden), task.OutputCount);
	log.LogInformation("Train:    {Rows} rows (full truth table), Adam lr {Lr}, {Epochs} epochs", task.Rows, LEARNING_RATE, EPOCHS);

	var net = factory(task.InputCount, task.OutputCount, hidden, generator);
	var optimizer = optim.Adam(net.Parameters, lr: LEARNING_RATE);

	// Track TWO separate questions across the run (they can diverge, and the divergence is the point):
	//   peakSoft — the SOFT (differentiable) net's best accuracy: does wiring+gate learning mechanically
	//              work at all? A high peakSoft with a low final hardened acc means "the mechanism learns
	//              but STE hardening destroys it" — still salvageable to bring back to the transformer.
	//   peakHard — the hardened discrete circuit's best accuracy: does the CIRCUIT actually train?
	var peakSoftBitAcc = 0.0;
	var peakHardBitAcc = 0.0;

	for (var epoch = 0; epoch < EPOCHS; epoch++)
	{
		// Anneal temperature geometrically from TEMP_START to TEMP_END across training.
		var progress = (double)epoch / (EPOCHS - 1);
		var temperature = TEMP_START * Math.Pow(TEMP_END / TEMP_START, progress);

		// Pure-soft warmup lets the gate blends settle on smooth gradients before the hard forward kicks in.
		var hard = epoch >= SOFT_WARMUP;

		optimizer.zero_grad();
		var output = net.Forward(task.Inputs, temperature, hard);
		// MSE on {0,1} targets: with the STE hard forward this equals the fraction of wrong output bits —
		// exactly the exhaustive bit error — and avoids BCE's log(0) on crisp 0/1 outputs.
		var loss = (output - task.Targets).pow(2).mean();
		loss.backward();
		optimizer.step();

		if (epoch == 0 || (epoch + 1) % LOG_EVERY == 0)
		{
			// Both metrics every log line: the soft net's accuracy (mechanism) and the hardened circuit's
			// (deliverable), so their gap is visible throughout, not just at the end.
			var (softAcc, softExact) = Evaluate(net, task, hard: false);
			var (hardAcc, hardExact) = Evaluate(net, task, hard: true);
			peakSoftBitAcc = Math.Max(peakSoftBitAcc, softAcc);
			peakHardBitAcc = Math.Max(peakHardBitAcc, hardAcc);
			log.LogInformation(
				"  epoch {Epoch,6}/{Total}   temp {Temp:0.000}   {Mode}   loss {Loss:0.0000}   soft {SoftAcc:P2}   hard {HardAcc:P2} (exact {Exact}/{Rows})",
				epoch + 1, EPOCHS, temperature, hard ? "hard" : "soft", loss.item<float>(), softAcc, hardAcc, hardExact, task.Rows);
		}
	}

	var (finalBitAcc, finalExact) = Evaluate(net, task, hard: true);
	log.LogInformation(
		"Result: peak SOFT {PeakSoft:P2} (mechanism), peak HARD {PeakHard:P2}, final hardened {Final:P2} exact {Exact}/{Rows}",
		peakSoftBitAcc, peakHardBitAcc, finalBitAcc, finalExact, task.Rows);
	if (finalExact == task.Rows)
		log.LogInformation("SUCCESS: the extracted circuit computes {Name} exactly.", task.Name);
	else if (peakSoftBitAcc > 0.99)
		log.LogInformation("NOTE: the soft mechanism learned {Name} (peak {PeakSoft:P2}) but hardening did not preserve it.", task.Name, peakSoftBitAcc);
	ReportNetlist(net, log);

	return (finalBitAcc, finalExact);
}

// Scores the hardened circuit (hard forward pass) over all rows: fraction of correct output bits, and
// the count of rows that are entirely correct.
// Scores the network over all rows: fraction of correct output bits and count of fully-correct rows.
// hard=true scores the discrete circuit (round via STE); hard=false scores the soft net, rounding its
// bipolar-derived {0,1} output at 0.5 so the "does the mechanism learn" metric is comparable.
static (double BitAcc, int ExactRows) Evaluate(ILogicNet net, LogicTask task, bool hard)
{
	using var noGrad = no_grad();
	var output = net.Forward(task.Inputs, temperature: 1.0, hard);
	var bits = hard ? output : (output > 0.5).to_type(ScalarType.Float32); // soft net: threshold to bits
	var correct = bits.eq(task.Targets);                    // [rows, outputCount] bool
	var bitAcc = correct.to_type(ScalarType.Float32).mean().item<float>();
	var exactRows = correct.all(dim: 1).sum().item<long>();
	return (bitAcc, (int)exactRows);
}

// Summarizes the extracted netlist: distribution of chosen gate functions, and what fraction of gates
// are constants (dead, prunable) or passthroughs (wires, not logic) — the cullability metric that ties
// this back to the "weights trained to exactly zero" story of the ternary-weight work.
static void ReportNetlist(ILogicNet net, ILogger log)
{
	var netlist = net.ExtractNetlist();
	var counts = new int[16];
	int total = 0, constants = 0, passthroughs = 0;
	foreach (var layer in netlist)
		foreach (var gate in layer)
		{
			counts[gate.Function]++;
			total++;
			if (gate.IsConstant) constants++;
			if (gate.IsPassthrough) passthroughs++;
		}

	log.LogInformation("  Netlist: {Total} gates — constants {Const} ({CPct:P0}), passthroughs {Pass} ({PPct:P0}), logic {Logic} ({LPct:P0})",
		total, constants, (double)constants / total, passthroughs, (double)passthroughs / total,
		total - constants - passthroughs, (double)(total - constants - passthroughs) / total);

	var order = Enumerable.Range(0, 16).Where(k => counts[k] > 0).OrderByDescending(k => counts[k]);
	log.LogInformation("  Gate mix: {Mix}", string.Join(", ", order.Select(k => $"{ExtractedGate.FunctionNames[k]}={counts[k]}")));
}

static ILoggerFactory CreateLoggerFactory()
{
	var configuration = new ConfigurationBuilder()
		.SetBasePath(AppContext.BaseDirectory)
		.AddYamlFile("appsettings.yaml", optional: false, reloadOnChange: false)
		.Build();

	// Console sink from YAML; file sink added here with a per-run timestamped filename (one log per
	// run), applying the lesson from the MiniGptScratch experiments where day-rolling logs stacked.
	var runStamp = DateTime.Now.ToString("yyyyMMdd-HHmmss");
	var logPath = Path.Combine(AppContext.BaseDirectory, "logs", $"logic-mlp-{runStamp}.log");

	var serilogLogger = new Serilog.LoggerConfiguration()
		.ReadFrom.Configuration(configuration)
		.WriteTo.File(
			path: logPath,
			outputTemplate: "{Timestamp:yyyy-MM-dd HH:mm:ss.fff zzz} [{Level:u3}] {Message:lj}{NewLine}{Exception}")
		.CreateLogger();

	return new SerilogLoggerFactory(serilogLogger, dispose: true);
}
