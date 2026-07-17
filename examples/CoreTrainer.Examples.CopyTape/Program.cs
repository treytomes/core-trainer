using CoreTrainer.Examples.CopyTape;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Serilog;
using Serilog.Extensions.Logging;
using static TorchSharp.torch;
using ILogger = Microsoft.Extensions.Logging.ILogger;

// ---------------------------------------------------------------------------------------
// CopyTape — rung 0 of the "context as Turing tape" ladder: isolate HEAD-TRACKING from computation.
//
// The binary adder (CoreTrainer.Examples.AdderTape) cliffed on held-out length with STRUCTURAL failures
// (dropped record slots, lost alignment) — not arithmetic errors. That pointed at head-tracking ("which
// cell am I on?") as the real wall, which a transformer has no explicit register for. The copy machine
// strips away all computation: emit the input (plain), reversed, or two inputs interleaved (zip). A
// length-generalization cliff here can ONLY be head-tracking. Discriminator: if plain generalizes but
// reverse/zip cliff, the model faked head-tracking with an induction head on the easy case.
// ---------------------------------------------------------------------------------------

var configuration = new ConfigurationBuilder()
	.SetBasePath(AppContext.BaseDirectory)
	.AddYamlFile("appsettings.yaml", optional: false, reloadOnChange: false)
	.Build();
var training = configuration.GetSection("Training");

var MODE = Enum.Parse<CopyMode>(training.GetValue<string>("Mode") ?? "Plain", ignoreCase: true);
var BLOCK_SIZE = training.GetValue<int>("BlockSize");
var EMBED_DIM = training.GetValue<int>("EmbedDim");
var HEADS = training.GetValue<int>("Heads");
var LAYERS = training.GetValue<int>("Layers");
var EPOCHS = training.GetValue<int>("Epochs");
var LEARNING_RATE = training.GetValue<double>("LearningRate");
var SEED = training.GetValue<int>("Seed");
var BATCH_SIZE = training.GetValue<int>("BatchSize");
var LOG_EVERY = training.GetValue<int>("LogEvery");
var EVAL_EVERY = training.GetValue<int>("EvalEvery");
var TRAIN_BITS = training.GetSection("TrainBits").Get<int[]>() ?? throw new InvalidOperationException("TrainBits missing.");
var HELDOUT_BITS = training.GetSection("HeldOutBits").Get<int[]>() ?? throw new InvalidOperationException("HeldOutBits missing.");
var MAX_PAIRS = training.GetValue<int>("MaxPairsPerWidth");
var CHECKPOINT_EVERY = training.GetValue<int>("CheckpointEvery");
var CHECKPOINT_PATH = training.GetValue<string>("CheckpointPath") ?? "checkpoints/copy-tape";
var RESUME_IF_PRESENT = training.GetValue<bool>("ResumeIfPresent");

using var loggerFactory = CreateLoggerFactory(configuration);
var log = loggerFactory.CreateLogger("CopyTape");

manual_seed(SEED);
set_num_threads(Environment.ProcessorCount);

// Fixed alphabet: bits, the two control tokens, terminator. Build the tokenizer from a sample containing
// all of them (zip includes '|'), so ids are stable across modes/lengths.
var vocabSample = CopyTask.Render(CopyMode.Zip, 4, 5, 10);
var tokenizer = new CharTokenizer(vocabSample);
var modelVocab = tokenizer.VocabSize + 1;

// Self-check: every rendered tape's output must equal the intended transform, or the metric is meaningless.
foreach (var (cb, ca, cbb) in new[] { (4, 5, 10), (8, 151, 99), (12, 4095, 2730) })
{
	var expected = CopyTask.Output(MODE, cb, ca, cbb);
	var rendered = CopyTask.Render(MODE, cb, ca, cbb);
	if (!rendered.Contains($">{expected}."))
		throw new InvalidOperationException($"Self-check failed for {MODE} bits={cb}: output '{expected}' not in '{rendered.Replace("\n", "/")}'.");
}
log.LogInformation("Format self-check passed ({Mode}).", MODE);

var trainSet = CopyDataset.Build(MODE, tokenizer, TRAIN_BITS, MAX_PAIRS, BLOCK_SIZE, seed: SEED);
var heldOutSet = CopyDataset.Build(MODE, tokenizer, HELDOUT_BITS, MAX_PAIRS, BLOCK_SIZE, seed: SEED + 1);

log.LogInformation("=== CopyTape ({Mode}) head-tracking probe ===", MODE);
log.LogInformation("Vocab:    {Vocab} tokens: {Chars}", tokenizer.VocabSize, string.Join("", tokenizer.Decode(Enumerable.Range(0, tokenizer.VocabSize).Select(i => (long)i)).Select(c => c == '\n' ? '/' : c)));
log.LogInformation("Model:    dim {Dim}, {Heads} heads, {Layers} layers, block {Block}, NoPE", EMBED_DIM, HEADS, LAYERS, BLOCK_SIZE);
log.LogInformation("Train:    bits [{Bits}], {N} examples (longest tape {Max} tok)", string.Join(",", TRAIN_BITS), trainSet.Examples.Count, trainSet.MaxLen);
log.LogInformation("Held-out: bits [{Bits}], {N} examples (longest tape {Max} tok)", string.Join(",", HELDOUT_BITS), heldOutSet.Examples.Count, heldOutSet.MaxLen);

var model = new MiniGpt(modelVocab, BLOCK_SIZE, EMBED_DIM, HEADS, LAYERS);
long paramCount = model.parameters().Sum(p => p.numel());
log.LogInformation("Params:   {Count:N0} ({M:0.00} M), Adam lr {Lr}, {Epochs} epochs, batch {Batch}", paramCount, paramCount / 1e6, LEARNING_RATE, EPOCHS, BATCH_SIZE);

var optimizer = optim.Adam(model.parameters(), lr: LEARNING_RATE);
var lossFn = nn.CrossEntropyLoss(ignore_index: trainSet.PadId);

var ckptDir = Path.Combine(AppContext.BaseDirectory, Path.GetDirectoryName(CHECKPOINT_PATH) ?? ".");
var ckptStem = Path.Combine(AppContext.BaseDirectory, CHECKPOINT_PATH);
Directory.CreateDirectory(ckptDir);
var startEpoch = 0;
if (RESUME_IF_PRESENT && File.Exists($"{ckptStem}.model") && File.Exists($"{ckptStem}.meta"))
{
	model.load($"{ckptStem}.model");
	if (File.Exists($"{ckptStem}.optim"))
		optimizer.load_state_dict($"{ckptStem}.optim");
	startEpoch = int.Parse(File.ReadAllText($"{ckptStem}.meta").Trim());
	log.LogInformation("Resumed from checkpoint at epoch {Epoch}.", startEpoch);
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
	{
		// Cheap per-log feedback: one held-out sample generation, so progress is visible qualitatively
		// between the expensive full evals. Shows the actual generated output vs expected.
		var sample = SampleOne(model, heldOutSet, tokenizer, BLOCK_SIZE);
		log.LogInformation("  epoch {Epoch,5}/{Total}   loss {Loss:0.0000}   sample[{Bits}b {Ok}] {Got}",
			epoch + 1, EPOCHS, epochLoss / exampleN, sample.Bits, sample.Ok ? "OK " : "BAD", sample.Line);
	}

	var isEvalEpoch = epoch == 0 || (epoch + 1) % EVAL_EVERY == 0;
	if (isEvalEpoch)
	{
		var (trainAcc, _) = EvalExecAccuracy(model, trainSet, tokenizer, MODE, BLOCK_SIZE, sample: 128);
		var (heldAcc, perBits) = EvalExecAccuracy(model, heldOutSet, tokenizer, MODE, BLOCK_SIZE, sample: 128, dumpLog: log, dumpCount: 3);
		log.LogInformation("           exec-acc  train-len {Train:P1}   held-out-len {Held:P1}", trainAcc, heldAcc);
		log.LogInformation("           held-out by length: {PerBits}", PerBitsStr(perBits));
	}

	if (CHECKPOINT_EVERY > 0 && (epoch + 1) % CHECKPOINT_EVERY == 0)
		SaveCheckpoint(model, optimizer, ckptStem, epoch + 1, log);
}

SaveCheckpoint(model, optimizer, ckptStem, EPOCHS, log);
log.LogInformation("=== Final ===");
LogFinal(model, trainSet, "train-lengths", tokenizer, MODE, BLOCK_SIZE, log);
LogFinal(model, heldOutSet, "held-out-lengths", tokenizer, MODE, BLOCK_SIZE, log);

return;

// Generates one held-out example (the first) and returns its correctness + a compact display line.
static (int Bits, bool Ok, string Line) SampleOne(MiniGpt model, CopyDataset data, CharTokenizer tok, int blockSize)
{
	var ex = data.Examples[0];
	// Rebuild the exact prompt from the stored ids up to PromptLen (mode-agnostic, avoids re-deriving).
	var promptIds = ex.Ids.Take(ex.PromptLen).ToArray();
	var prompt = tensor(promptIds, dtype: ScalarType.Int64).reshape(1, promptIds.Length);
	var budget = Math.Min(blockSize - promptIds.Length - 1, ex.Ids.Length - ex.PromptLen + 2);
	var gen = model.GenerateBatch(prompt, budget)[0].data<long>().ToArray();
	var text = tok.Decode(gen);
	var expected = tok.Decode(ex.Ids);
	var got = FirstTape(text);
	return (ex.Bits, got == FirstTape(expected), got.Replace("\n", "/"));
}

// Free-generation execution accuracy, per bit-length. Parses the model's output (up to the '.' terminator)
// and checks it equals the intended transform. Batched by bit-width (equal prompt lengths).
static (double Overall, SortedDictionary<int, double> PerBits) EvalExecAccuracy(
	MiniGpt model, CopyDataset data, CharTokenizer tok, CopyMode mode, int blockSize, int sample,
	ILogger? dumpLog = null, int dumpCount = 0)
{
	int correct = 0, total = 0;
	var perBits = new SortedDictionary<int, double>();
	var failures = new List<(int Bits, int A, int B, string Got, string Exp)>();

	foreach (var group in data.Examples.GroupBy(e => e.Bits))
	{
		int gCorrect = 0, gTotal = 0;
		var exs = group.ToList();
		var step = sample > 0 && exs.Count > sample ? exs.Count / sample : 1;
		var picked = new List<CopyDataset.Example>();
		for (var i = 0; i < exs.Count; i += step)
			picked.Add(exs[i]);
		if (picked.Count == 0)
			continue;

		var promptLen = picked[0].PromptLen;
		var promptData = new long[picked.Count, promptLen];
		for (var r = 0; r < picked.Count; r++)
			for (var c = 0; c < promptLen; c++)
				promptData[r, c] = picked[r].Ids[c];
		var prompts = tensor(promptData);

		var budget = Math.Min(blockSize - promptLen - 1, picked[0].Ids.Length - promptLen + 4);
		var generated = model.GenerateBatch(prompts, budget);

		for (var r = 0; r < picked.Count; r++)
		{
			var ex = picked[r];
			var got = FirstTape(tok.Decode(generated[r].data<long>().ToArray()));
			var exp = CopyTask.Output(mode, ex.Bits, ex.A, ex.B);
			var ok = got == exp;
			if (ok) { correct++; gCorrect++; }
			total++; gTotal++;
			if (dumpLog is not null && !ok && failures.Count < 64)
				failures.Add((ex.Bits, ex.A, ex.B, got, exp));
		}
		perBits[group.Key] = gTotal == 0 ? 0 : (double)gCorrect / gTotal;
	}

	if (dumpLog is not null && failures.Count > 0)
	{
		dumpLog.LogInformation("    {Fail} failures; longest:", failures.Count);
		foreach (var f in failures.OrderByDescending(f => f.Bits).Take(dumpCount))
			dumpLog.LogInformation("    [{Bits}b] got {Got}  exp {Exp}", f.Bits, f.Got, f.Exp);
	}

	return (total == 0 ? 0 : (double)correct / total, perBits);
}

// The model's output between the '>' marker and the '.' terminator (its first tape; ignores overflow).
static string FirstTape(string text)
{
	var gt = text.IndexOf('>');
	if (gt < 0) return "";
	var rest = text[(gt + 1)..];
	var dot = rest.IndexOf('.');
	return dot < 0 ? rest.Replace("\n", "") : rest[..dot].Replace("\n", "");
}

static string PerBitsStr(SortedDictionary<int, double> perBits) =>
	string.Join("  ", perBits.Select(kv => $"{kv.Key}b:{kv.Value:P0}"));

static void LogFinal(MiniGpt model, CopyDataset data, string label, CharTokenizer tok, CopyMode mode, int blockSize, ILogger log)
{
	var (overall, perBits) = EvalExecAccuracy(model, data, tok, mode, blockSize, sample: 0);
	log.LogInformation("  {Label,-18} exec-accuracy {Acc:P2}  (over {N})  by length: {PerBits}", label, overall, data.Examples.Count, PerBitsStr(perBits));
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
	var logPath = Path.Combine(AppContext.BaseDirectory, "logs", $"copy-tape-{runStamp}.log");
	var serilogLogger = new Serilog.LoggerConfiguration()
		.ReadFrom.Configuration(configuration)
		.WriteTo.File(logPath, outputTemplate: "{Timestamp:yyyy-MM-dd HH:mm:ss.fff zzz} [{Level:u3}] {Message:lj}{NewLine}{Exception}")
		.CreateLogger();
	return new SerilogLoggerFactory(serilogLogger, dispose: true);
}
