using CoreTrainer.Examples.DiffNand;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Serilog;
using Serilog.Extensions.Logging;
using static TorchSharp.torch;
// Disambiguate ILogger (both MS and Serilog define one); the example logs via the MS abstraction.
using ILogger = Microsoft.Extensions.Logging.ILogger;

// ---------------------------------------------------------------------------------------
// Differentiable NAND — the gradient-trained counterpart to CoreTrainer.Examples.NandArithmetic.
//
// Instead of evolving a NAND circuit with CGP, this builds a LAYERED network of soft-NAND gates
// (1 - a·b, which is exactly NAND on crisp {0,1} inputs) and trains it with backpropagation. The
// only learnable freedom is connectivity — which two earlier outputs each gate reads, chosen by a
// softmax that is annealed toward a hard one-hot pick. A straight-through estimator makes the
// forward pass the exact discrete circuit, so the training accuracy IS the extracted netlist's
// exhaustive accuracy.
//
// Same problem and same interleaved sum/carry output layout as the CGP example, so the two are
// directly comparable: can gradient descent + a fixed layered topology find the carry chain that
// the evolutionary search walls on?
// ---------------------------------------------------------------------------------------

const int BIT_WIDTH = 4;

// Input layout:  [A bits] [B bits] [opSelect].  Output layout: interleaved sum/carry, identical to
// the CGP example: sum0, c1, sum1, c2, ..., sum_{W-1}, c_W, sum_W.
var inputCount = 2 * BIT_WIDTH + 1;
var outputCount = 2 * BIT_WIDTH + 1;

// Hidden depth: a ripple carry of XORs is several NAND-levels deep per bit, so give it generous
// depth. The cumulative skip pool means signals carry forward without being re-derived.
var hiddenWidths = new[] { 64, 64, 64, 64, 64, 64, 64, 64 };

const int EPOCHS = 20000;
const double LEARNING_RATE = 0.01;
const int SOFT_WARMUP = 2000;          // pure-soft epochs before switching on straight-through
const double TEMP_START = 2.0;
const double TEMP_END = 0.05;          // final softmax temperature (sharp, near one-hot)
const int LOG_EVERY = 500;
const int SEED = 1;

using var loggerFactory = CreateLoggerFactory();
var log = loggerFactory.CreateLogger("DiffNand");

manual_seed(SEED);
var generator = random.manual_seed(SEED);

// Build the full add/sub truth table once: every (A, B, op) combination is a training row. With
// only 2^(2W+1) rows there is no train/test split — we fit the entire function, and "accuracy"
// over this set is exhaustive by definition.
var (inputs, targets, rowCount) = BuildTruthTable(BIT_WIDTH, inputCount, outputCount);

log.LogInformation("=== Differentiable NAND Trainer ===");
log.LogInformation("Goal:     train a {BitWidth}-bit adder/subtractor as a layered soft-NAND net", BIT_WIDTH);
log.LogInformation("Network:  {Inputs} inputs, layers [{Layers}], {Outputs} outputs",
	inputCount, string.Join(", ", hiddenWidths), outputCount);
log.LogInformation("Train:    {Rows} rows (full truth table), Adam lr {Lr}, {Epochs} epochs", rowCount, LEARNING_RATE, EPOCHS);
log.LogInformation("Harden:   soft warmup {Warmup}, then straight-through; temp {T0}->{T1}", SOFT_WARMUP, TEMP_START, TEMP_END);

var net = new DiffLogicNet(inputCount, outputCount, hiddenWidths, generator);
var optimizer = optim.Adam(net.Parameters, lr: LEARNING_RATE);

for (var epoch = 0; epoch < EPOCHS; epoch++)
{
	// Anneal temperature geometrically from TEMP_START to TEMP_END across training.
	var progress = (double)epoch / (EPOCHS - 1);
	var temperature = TEMP_START * Math.Pow(TEMP_END / TEMP_START, progress);

	// Pure-soft warmup lets wires settle on smooth gradients before the hard forward pass kicks in.
	var hard = epoch >= SOFT_WARMUP;

	optimizer.zero_grad();
	var output = net.Forward(inputs, temperature, hard);
	// MSE on {0,1} targets: with the STE hard forward this equals the fraction of wrong output bits,
	// which is exactly the exhaustive bit error — and avoids BCE's log(0) on crisp 0/1 outputs.
	var loss = (output - targets).pow(2).mean();
	loss.backward();
	optimizer.step();

	if (epoch == 0 || (epoch + 1) % LOG_EVERY == 0)
	{
		// Report the hardened circuit's exhaustive accuracy: a hard forward pass scores the exact
		// discrete netlist over all rows, so this is the truthful metric, not the soft loss.
		var (bitAcc, exactRows) = Evaluate(net, inputs, targets);
		log.LogInformation(
			"  epoch {Epoch,6}/{Total}   temp {Temp:0.000}   {Mode}   loss {Loss:0.0000}   bit-acc {BitAcc:P2}   exact rows {Exact}/{Rows}",
			epoch + 1, EPOCHS, temperature, hard ? "hard" : "soft", loss.item<float>(), bitAcc, exactRows, rowCount);
	}
}

// Final verdict on the hardened netlist.
var (finalBitAcc, finalExact) = Evaluate(net, inputs, targets);
log.LogInformation("Final hardened netlist: bit-acc {BitAcc:P2}, fully-correct rows {Exact}/{Rows}",
	finalBitAcc, finalExact, rowCount);
if (finalExact == rowCount)
	log.LogInformation("SUCCESS: the extracted circuit is a fully correct {BitWidth}-bit adder/subtractor.", BIT_WIDTH);

return;

// === Helpers ===========================================================================

// Builds the full truth table as tensors: inputs [rows, inputCount] in {0,1}, and the interleaved
// sum/carry targets [rows, outputCount]. Mirrors the CGP example's encoding exactly.
static (Tensor Inputs, Tensor Targets, int Rows) BuildTruthTable(int bitWidth, int inputCount, int outputCount)
{
	var range = 1 << bitWidth;
	var rows = range * range * 2;
	var inputData = new float[rows, inputCount];
	var targetData = new float[rows, outputCount];

	var row = 0;
	for (var a = 0; a < range; a++)
	{
		for (var b = 0; b < range; b++)
		{
			foreach (var subtract in new[] { false, true })
			{
				for (var bit = 0; bit < bitWidth; bit++)
				{
					inputData[row, bit] = (a >> bit) & 1;
					inputData[row, bitWidth + bit] = (b >> bit) & 1;
				}
				inputData[row, 2 * bitWidth] = subtract ? 1f : 0f;

				var packed = Expected(a, b, subtract, bitWidth);
				for (var bit = 0; bit < outputCount; bit++)
					targetData[row, bit] = (packed >> bit) & 1;

				row++;
			}
		}
	}

	return (tensor(inputData), tensor(targetData), rows);
}

// The correct interleaved sum/carry output vector for (a, b, op), packed LSB-first — identical to
// the CGP example's Expected: a ripple-carry adder, two's-complement subtraction (a + ~b + 1).
static int Expected(int a, int b, bool subtract, int bitWidth)
{
	var packed = 0;
	var carry = subtract ? 1 : 0;
	for (var k = 0; k < bitWidth; k++)
	{
		var ak = (a >> k) & 1;
		var bk = (b >> k) & 1;
		var op2 = subtract ? bk ^ 1 : bk;
		var full = ak + op2 + carry;

		packed |= (full & 1) << (2 * k);
		carry = full >> 1;
		packed |= carry << (2 * k + 1);
	}
	packed |= ((subtract ? 1 : 0) ^ carry) << (2 * bitWidth);
	return packed;
}

// Scores the hardened circuit (hard forward pass) over all rows: fraction of correct output bits,
// and the count of rows that are entirely correct.
static (double BitAcc, int ExactRows) Evaluate(DiffLogicNet net, Tensor inputs, Tensor targets)
{
	using var noGrad = no_grad();
	var output = net.Forward(inputs, temperature: 1.0, hard: true);
	var correct = output.eq(targets);                       // [rows, outputCount] bool
	var bitAcc = correct.to_type(ScalarType.Float32).mean().item<float>();
	var exactRows = correct.all(dim: 1).sum().item<long>();
	return (bitAcc, (int)exactRows);
}

static ILoggerFactory CreateLoggerFactory()
{
	var configuration = new ConfigurationBuilder()
		.SetBasePath(AppContext.BaseDirectory)
		.AddYamlFile("appsettings.yaml", optional: false, reloadOnChange: false)
		.Build();

	var serilogLogger = new Serilog.LoggerConfiguration()
		.ReadFrom.Configuration(configuration)
		.CreateLogger();

	return new SerilogLoggerFactory(serilogLogger, dispose: true);
}
