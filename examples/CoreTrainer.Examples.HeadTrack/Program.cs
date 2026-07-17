using CoreTrainer.Examples.HeadTrack;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Serilog;
using Serilog.Extensions.Logging;
using static TorchSharp.torch;
using ILogger = Microsoft.Extensions.Logging.ILogger;

// ---------------------------------------------------------------------------------------
// HeadTrack — measure the reliable head-tracking distance D, and how it scales with model size.
//
// The scrollable-tape Turing-machine design (transformer = control, external buffer = tape, window ≤ D)
// rests on ONE primitive: within a window, track a head through L/R moves and read the cell under it.
// This probe verifies that primitive and MEASURES D — the move count at which per-step tracking accuracy
// crosses a threshold. Run across model sizes to get D(params): the scaling law that decides whether the
// scrollable-tape machine is practical (D grows usefully with size) or not (D ~ constant / log).
//
// Output includes machine-readable `DATA,params,moves,accuracy` lines for direct plotting.
// ---------------------------------------------------------------------------------------

var configuration = new ConfigurationBuilder()
	.SetBasePath(AppContext.BaseDirectory)
	.AddYamlFile("appsettings.yaml", optional: false, reloadOnChange: false)
	.Build();
var training = configuration.GetSection("Training");

var BLOCK_SIZE = training.GetValue<int>("BlockSize");
var HEADS = training.GetValue<int>("Heads");
var PINNED_EMBED = training.GetValue<int>("EmbedDim"); // used when sweeping depth
var PINNED_LAYERS = training.GetValue<int>("Layers");  // used when sweeping width
var SWEEP_AXIS = (training.GetValue<string>("SweepAxis") ?? "width").ToLowerInvariant();
var SWEEP_VALUES = training.GetSection("SweepValues").Get<int[]>() ?? throw new InvalidOperationException("SweepValues missing.");
if (SWEEP_AXIS != "width" && SWEEP_AXIS != "depth")
	throw new InvalidOperationException($"SweepAxis must be 'width' or 'depth', got '{SWEEP_AXIS}'.");
var EPOCHS = training.GetValue<int>("Epochs");
var LEARNING_RATE = training.GetValue<double>("LearningRate");
var SEED = training.GetValue<int>("Seed");
var BATCH_SIZE = training.GetValue<int>("BatchSize");
var LOG_EVERY = training.GetValue<int>("LogEvery");
var EVAL_EVERY = training.GetValue<int>("EvalEvery");
var TRAIN_MOVES = training.GetSection("TrainMoveCounts").Get<int[]>() ?? throw new InvalidOperationException("TrainMoveCounts missing.");
var EVAL_MOVES = training.GetSection("EvalMoveCounts").Get<int[]>() ?? throw new InvalidOperationException("EvalMoveCounts missing.");
var PER_COUNT = training.GetValue<int>("ExamplesPerCount");
var D_THRESHOLD = training.GetValue<double>("DThreshold");
var CHECKPOINT_EVERY = training.GetValue<int>("CheckpointEvery");
var CHECKPOINT_PATH = training.GetValue<string>("CheckpointPath") ?? "checkpoints/head-track";
var RESUME_IF_PRESENT = training.GetValue<bool>("ResumeIfPresent");

using var loggerFactory = CreateLoggerFactory(configuration);
var log = loggerFactory.CreateLogger("HeadTrack");

manual_seed(SEED);
set_num_threads(Environment.ProcessorCount);

var vocabSample = HeadTrackTask.Render("0110", "RRLR") + HeadTrackTask.Render("1", "L");
var tokenizer = new CharTokenizer(vocabSample);
var modelVocab = tokenizer.VocabSize + 1;

// Self-check: every rendered probe must have a trace that matches the head-walk oracle.
foreach (var (t, m) in new[] { ("01101", "RRLR"), ("1101", "RRRR"), ("101", "LLR") })
{
	var expected = HeadTrackTask.Trace(t, m);
	if (!HeadTrackTask.Render(t, m).Contains($">{expected}."))
		throw new InvalidOperationException($"Self-check failed for tape={t} moves={m}.");
}
log.LogInformation("Format self-check passed.");

// Datasets are shared across all widths (task and length split are fixed; only the model changes).
var trainSet = HeadTrackDataset.Build(tokenizer, TRAIN_MOVES, PER_COUNT, BLOCK_SIZE, seed: SEED);
var evalSet = HeadTrackDataset.Build(tokenizer, EVAL_MOVES, PER_COUNT, BLOCK_SIZE, seed: SEED + 1);

log.LogInformation("=== HeadTrack {Axis} sweep (D & meanAcc vs {Axis2}) ===", SWEEP_AXIS, SWEEP_AXIS);
log.LogInformation("Fixed:    {Heads} heads, block {Block}, NoPE, {Epochs} epochs, lr {Lr}; pinned {Pin}",
	HEADS, BLOCK_SIZE, EPOCHS, LEARNING_RATE, SWEEP_AXIS == "width" ? $"Layers={PINNED_LAYERS}" : $"EmbedDim={PINNED_EMBED}");
log.LogInformation("Sweep:    {Axis} {Values}", SWEEP_AXIS, string.Join(",", SWEEP_VALUES));
log.LogInformation("Train:    move counts [{Moves}], {N} examples", string.Join(",", TRAIN_MOVES), trainSet.Examples.Count);
log.LogInformation("Eval:     move counts [{Moves}], {N} examples, D threshold {T:P0}", string.Join(",", EVAL_MOVES), evalSet.Examples.Count, D_THRESHOLD);

// One fresh model per swept value, trained to convergence and measured, all other axes fixed. Per-point
// checkpoints (stem includes axis+value) so an interrupted run resumes the exact point it was on and
// finished points are skipped — the whole sweep is a single unattended, restartable run.
var results = new List<(int Value, long Params, int D, double MeanAcc)>();
foreach (var value in SWEEP_VALUES)
{
	// Resolve this point's (embedDim, layers) from the swept axis; the other stays pinned.
	var embedDim = SWEEP_AXIS == "width" ? value : PINNED_EMBED;
	var layers = SWEEP_AXIS == "depth" ? value : PINNED_LAYERS;
	if (embedDim % HEADS != 0)
		throw new InvalidOperationException($"EmbedDim {embedDim} not divisible by {HEADS} heads.");

	var stem = $"{Path.Combine(AppContext.BaseDirectory, CHECKPOINT_PATH)}-{SWEEP_AXIS[0]}{value}";
	Directory.CreateDirectory(Path.GetDirectoryName(stem)!);

	// Isolate each point: log any failure INTO the file (unhandled exceptions otherwise go only to stderr
	// and vanish from an unattended run's log) and continue the sweep, so one bad point doesn't lose the
	// others. The point's checkpoint remains, so it can be retried later by re-running.
	try
	{
		var (paramCount, d, meanAcc) = RunPoint(embedDim, layers, stem);
		results.Add((value, paramCount, d, meanAcc));
		log.LogInformation("### {Axis} {Value} done: params {Params:N0}, D={D} moves, meanAcc {Mean:P1}", SWEEP_AXIS, value, paramCount, d, meanAcc);
	}
	catch (Exception ex)
	{
		log.LogError(ex, "### {Axis} {Value} FAILED — skipping to next.", SWEEP_AXIS, value);
	}
}

// Final plot-ready summary: one row per swept value. Grep 'SWEEP,' for the scaling curve. meanAcc is the
// primary metric (whole-curve, robust); D@threshold kept for continuity.
log.LogInformation("=== {Axis} sweep complete: D & meanAcc vs params ===", SWEEP_AXIS);
log.LogInformation("  SWEEP,{Axis},params,D,meanAcc", SWEEP_AXIS);
foreach (var (v, p, d, mean) in results)
	log.LogInformation("  SWEEP,{Value},{Params},{D},{Mean:0.0000}", v, p, d, mean);

return;

// Builds, trains to convergence, and measures D for one (embedDim, layers) point. Resumes from this
// point's checkpoint if present (so the sweep is restartable); returns (param count, D, meanAcc).
(long Params, int D, double MeanAcc) RunPoint(int embedDim, int layers, string ckptStem)
{
	manual_seed(SEED); // same init draw per point so the sweep is reproducible
	var model = new MiniGpt(modelVocab, BLOCK_SIZE, embedDim, HEADS, layers);
	var paramCount = model.parameters().Sum(p => p.numel());
	var optimizer = optim.Adam(model.parameters(), lr: LEARNING_RATE);
	var lossFn = nn.CrossEntropyLoss(ignore_index: trainSet.PadId);

	log.LogInformation("--- dim {Dim}, layers {Layers}, params {Params:N0} ({M:0.000} M) ---", embedDim, layers, paramCount, paramCount / 1e6);

	var startEpoch = 0;
	if (RESUME_IF_PRESENT && File.Exists($"{ckptStem}.model") && File.Exists($"{ckptStem}.meta"))
	{
		model.load($"{ckptStem}.model");
		if (File.Exists($"{ckptStem}.optim"))
			optimizer.load_state_dict($"{ckptStem}.optim");
		startEpoch = int.Parse(File.ReadAllText($"{ckptStem}.meta").Trim());
		log.LogInformation("    resumed (dim {Dim}, layers {Layers}) at epoch {Epoch}.", embedDim, layers, startEpoch);
	}

	var exampleN = trainSet.Inputs.shape[0];
	for (var epoch = startEpoch; epoch < EPOCHS; epoch++)
	{
		model.train();
		var perm = randperm(exampleN);
		var epochLoss = 0.0;
		for (long start = 0; start < exampleN; start += BATCH_SIZE)
		{
			var count = Math.Min(BATCH_SIZE, exampleN - start);
			var idx = perm.narrow(0, start, count);
			optimizer.zero_grad();
			var logits = model.forward(trainSet.Inputs.index_select(0, idx));
			var loss = lossFn.forward(logits.reshape(-1, modelVocab), trainSet.Targets.index_select(0, idx).reshape(-1));
			loss.backward();
			optimizer.step();
			epochLoss += loss.item<float>() * count;
		}

		if (epoch == 0 || (epoch + 1) % LOG_EVERY == 0)
			log.LogInformation("  d{Dim}L{Layers} epoch {Epoch,5}/{Total}   loss {Loss:0.0000}", embedDim, layers, epoch + 1, EPOCHS, epochLoss / exampleN);

		if (epoch == 0 || (epoch + 1) % EVAL_EVERY == 0)
			ReportD(model, evalSet, tokenizer, EVAL_MOVES, BLOCK_SIZE, D_THRESHOLD, paramCount, epoch + 1, log);

		if (CHECKPOINT_EVERY > 0 && (epoch + 1) % CHECKPOINT_EVERY == 0)
			SaveCheckpoint(model, optimizer, ckptStem, epoch + 1, log);
	}

	SaveCheckpoint(model, optimizer, ckptStem, EPOCHS, log);
	var (finalD, finalMean) = ReportD(model, evalSet, tokenizer, EVAL_MOVES, BLOCK_SIZE, D_THRESHOLD, paramCount, EPOCHS, log);
	model.Dispose();
	optimizer.Dispose();
	return (paramCount, finalD, finalMean);
}

// Measures per-step trace accuracy at each eval move count, reports the D-curve and D (the largest move
// count whose accuracy >= threshold), and emits machine-readable `DATA,params,moves,accuracy` rows for
// plotting D vs model size across a sweep of runs.
static (int D, double MeanAcc) ReportD(MiniGpt model, HeadTrackDataset data, CharTokenizer tok, int[] evalMoves, int blockSize,
	double threshold, long paramCount, int epoch, ILogger log)
{
	using var noGrad = no_grad();
	model.eval();

	var curve = new SortedDictionary<int, double>();
	foreach (var group in data.Examples.GroupBy(e => e.Moves))
	{
		var exs = group.ToList();
		var promptLen = exs[0].PromptLen;
		var promptData = new long[exs.Count, promptLen];
		for (var r = 0; r < exs.Count; r++)
			for (var c = 0; c < promptLen; c++)
				promptData[r, c] = exs[r].Ids[c];
		var prompts = tensor(promptData);

		var budget = group.Key + 2; // trace length = move count, plus terminator slack
		var generated = model.GenerateBatch(prompts, Math.Min(blockSize - promptLen - 1, budget));

		// Per-STEP accuracy: fraction of trace symbols correct across all examples at this move count
		// (not whole-sequence — this locates where the head drifts and is the right resolution for D).
		long correctSteps = 0, totalSteps = 0;
		for (var r = 0; r < exs.Count; r++)
		{
			var got = Trace(tok.Decode(generated[r].data<long>().ToArray()));
			var exp = tok.Decode(exs[r].Ids); exp = Trace(exp);
			var n = Math.Min(got.Length, exp.Length);
			for (var i = 0; i < exp.Length; i++)
			{
				totalSteps++;
				if (i < got.Length && got[i] == exp[i])
					correctSteps++;
			}
		}
		curve[group.Key] = totalSteps == 0 ? 0 : (double)correctSteps / totalSteps;
	}

	// D = largest move count with per-step accuracy >= threshold (0 if even the shortest fails it).
	var d = 0;
	foreach (var kv in curve)
		if (kv.Value >= threshold)
			d = kv.Key;

	// Mean per-step accuracy across all eval move counts — the AUC-style metric. Unlike D@threshold, it
	// uses the WHOLE curve, so it is monotonic-friendly and not knife-edged by one crossing point (the
	// width sweep showed D@90% flip 12→8 between widths whose curves were actually near-identical; mean
	// accuracy correctly showed them equal). This is the primary scaling metric; D is kept for continuity.
	var meanAcc = curve.Count == 0 ? 0 : curve.Values.Average();

	log.LogInformation("  [epoch {Epoch}] meanAcc {Mean:P1}  D={D} moves (thr {T:P0})   curve: {Curve}",
		epoch, meanAcc, d, threshold, string.Join("  ", curve.Select(kv => $"{kv.Key}:{kv.Value:P0}")));
	// Machine-readable rows for plotting across a sweep. Grep 'DATA,' for curves, 'DPOINT,' for per-eval
	// summary (params, D, meanAcc, epoch).
	foreach (var kv in curve)
		log.LogInformation("  DATA,{Params},{Moves},{Acc:0.0000},{Epoch}", paramCount, kv.Key, kv.Value, epoch);
	log.LogInformation("  DPOINT,{Params},{D},{Mean:0.0000},{Epoch}", paramCount, d, meanAcc, epoch);
	return (d, meanAcc);
}

// The trace between the '>' marker and the '.' terminator (the model's first tape; ignores overflow).
static string Trace(string text)
{
	var gt = text.IndexOf('>');
	if (gt < 0) return "";
	var rest = text[(gt + 1)..];
	var dot = rest.IndexOf('.');
	return (dot < 0 ? rest : rest[..dot]).Replace("\n", "");
}

static void SaveCheckpoint(MiniGpt model, TorchSharp.Modules.Adam optimizer, string stem, int nextEpoch, ILogger log)
{
	model.save($"{stem}.model");
	optimizer.save_state_dict($"{stem}.optim");
	File.WriteAllText($"{stem}.meta", nextEpoch.ToString());
	log.LogInformation("           [checkpoint saved @ epoch {Epoch}]", nextEpoch);
}

static ILoggerFactory CreateLoggerFactory(IConfiguration configuration)
{
	var runStamp = DateTime.Now.ToString("yyyyMMdd-HHmmss");
	var logPath = Path.Combine(AppContext.BaseDirectory, "logs", $"head-track-{runStamp}.log");
	var serilogLogger = new Serilog.LoggerConfiguration()
		.ReadFrom.Configuration(configuration)
		.WriteTo.File(logPath, outputTemplate: "{Timestamp:yyyy-MM-dd HH:mm:ss.fff zzz} [{Level:u3}] {Message:lj}{NewLine}{Exception}")
		.CreateLogger();
	return new SerilogLoggerFactory(serilogLogger, dispose: true);
}
