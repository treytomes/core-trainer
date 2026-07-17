using CoreTrainer.Examples.AdderTape;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Serilog;
using Serilog.Extensions.Logging;
using static TorchSharp.torch;
using ILogger = Microsoft.Extensions.Logging.ILogger;

// ---------------------------------------------------------------------------------------
// AdderTape — "context window as Turing tape" experiment.
//
// A fixed-depth transformer can't fold a binary adder's carry chain into constant depth (that's why the
// parallel adder is hard to train). But the carry chain is sequential, and autoregressive generation is
// an unbounded-depth sequential computer: each emitted token is a fresh forward pass that reads the whole
// scratchpad so far. So we train the model to write the ripple-carry derivation step by step, and test
// whether it EXECUTES the algorithm — by holding out LENGTH: train on 1-4 bit additions, test on unseen
// 6-8 bit ones. Generalizing to longer lengths = it learned the (position-free) step function and feeds
// its own carry forward; failing = it memorized a bounded-depth approximation (how real LLMs fail at long
// addition). NoPE (no positional embeddings) keeps the test valid — no untrained positions at test length.
// ---------------------------------------------------------------------------------------

var configuration = new ConfigurationBuilder()
	.SetBasePath(AppContext.BaseDirectory)
	.AddYamlFile("appsettings.yaml", optional: false, reloadOnChange: false)
	.Build();
var training = configuration.GetSection("Training");

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
var CHECKPOINT_EVERY = training.GetValue<int>("CheckpointEvery");
var CHECKPOINT_PATH = training.GetValue<string>("CheckpointPath") ?? "checkpoints/adder-tape";
var RESUME_IF_PRESENT = training.GetValue<bool>("ResumeIfPresent");
var TRAIN_BITS = training.GetSection("TrainBits").Get<int[]>() ?? throw new InvalidOperationException("TrainBits missing.");
var HELDOUT_BITS = training.GetSection("HeldOutBits").Get<int[]>() ?? throw new InvalidOperationException("HeldOutBits missing.");
var MAX_PAIRS = training.GetValue<int>("MaxPairsPerWidth");

using var loggerFactory = CreateLoggerFactory(configuration);
var log = loggerFactory.CreateLogger("AdderTape");

manual_seed(SEED);
set_num_threads(Environment.ProcessorCount);

// Vocabulary: build from a representative tape that contains every token the format can produce (all
// digit/letter markers). One wide example is enough since the alphabet is tiny and fixed.
var vocabSample = AdderTape.Render(0, 0, 8) + AdderTape.Render(255, 255, 8);
var tokenizer = new CharTokenizer(vocabSample);
var modelVocab = tokenizer.VocabSize + 1; // + reserved pad id

// Self-check: the ground-truth tape (AdderTape.Render) must parse back (ParseSum) to a+b for every case,
// or the whole exec-accuracy metric is meaningless. Cheap insurance against a Render/ParseSum mismatch
// before spending hours training. Covers a spread of widths incl. the all-ones worst case.
foreach (var (ca, cb, cbits) in new[] { (0, 0, 1), (3, 1, 3), (6, 7, 4), (151, 127, 8), (255, 255, 8), (170, 85, 8) })
{
	if (!(ParseSum(AdderTape.Render(ca, cb, cbits), out var got) && got == ca + cb))
		throw new InvalidOperationException($"Format self-check failed: Render({ca},{cb},{cbits}) parsed to {got}, expected {ca + cb}.");
}
log.LogInformation("Format self-check passed (Render round-trips through ParseSum).");

// Train and held-out splits use DISJOINT bit-widths — the length-generalization test.
var trainSet = TapeDataset.Build(tokenizer, TRAIN_BITS, MAX_PAIRS, BLOCK_SIZE, seed: SEED);
var heldOutSet = TapeDataset.Build(tokenizer, HELDOUT_BITS, MAX_PAIRS, BLOCK_SIZE, seed: SEED + 1);

log.LogInformation("=== AdderTape (context-as-tape) Trainer ===");
log.LogInformation("Vocab:    {Vocab} tokens: {Chars}", tokenizer.VocabSize, string.Join("", tokenizer.Decode(Enumerable.Range(0, tokenizer.VocabSize).Select(i => (long)i)).Select(c => c == '\n' ? '/' : c)));
log.LogInformation("Model:    dim {Dim}, {Heads} heads, {Layers} layers, block {Block}, NoPE", EMBED_DIM, HEADS, LAYERS, BLOCK_SIZE);
log.LogInformation("Train:    bits [{Bits}], {N} examples (longest tape {Max} tok)", string.Join(",", TRAIN_BITS), trainSet.Examples.Count, trainSet.MaxLen);
log.LogInformation("Held-out: bits [{Bits}], {N} examples (longest tape {Max} tok)", string.Join(",", HELDOUT_BITS), heldOutSet.Examples.Count, heldOutSet.MaxLen);

var model = new MiniGpt(modelVocab, BLOCK_SIZE, EMBED_DIM, HEADS, LAYERS);
long paramCount = model.parameters().Sum(p => p.numel());
log.LogInformation("Params:   {Count:N0} ({M:0.00} M), Adam lr {Lr}, {Epochs} epochs, batch {Batch}", paramCount, paramCount / 1e6, LEARNING_RATE, EPOCHS, BATCH_SIZE);

var optimizer = optim.Adam(model.parameters(), lr: LEARNING_RATE);
var lossFn = nn.CrossEntropyLoss(ignore_index: trainSet.PadId);

// Resume from checkpoint if one exists: restores model weights, Adam moments, and the epoch counter so a
// stopped run continues rather than restarting. Absolute path so it's independent of the working dir.
var ckptDir = Path.Combine(AppContext.BaseDirectory, Path.GetDirectoryName(CHECKPOINT_PATH) ?? ".");
var ckptStem = Path.Combine(AppContext.BaseDirectory, CHECKPOINT_PATH);
Directory.CreateDirectory(ckptDir);
var startEpoch = 0;
if (RESUME_IF_PRESENT && File.Exists($"{ckptStem}.model") && File.Exists($"{ckptStem}.meta"))
{
	model.load($"{ckptStem}.model");
	if (File.Exists($"{ckptStem}.optim"))
		optimizer.load_state_dict($"{ckptStem}.optim"); // restore Adam moments so momentum/variance continue
	startEpoch = int.Parse(File.ReadAllText($"{ckptStem}.meta").Trim());
	log.LogInformation("Resumed from checkpoint at epoch {Epoch} ({Path}).", startEpoch, CHECKPOINT_PATH);

	// Evaluate the resumed model immediately, so the loaded checkpoint's exec-accuracy (and failure dump)
	// is visible on startup rather than only at the next EvalEvery boundary — useful when re-running with
	// an improved eval/dump against an already-trained checkpoint.
	var t0 = EvalExecAccuracy(model, trainSet, tokenizer, BLOCK_SIZE, sample: 128);
	var h0 = EvalExecAccuracy(model, heldOutSet, tokenizer, BLOCK_SIZE, sample: 128, dumpLog: log, dumpCount: 6);
	log.LogInformation("  [resume eval @ {Epoch}]  exec-acc  train-len {Train:P1}   held-out-len {Held:P1}", startEpoch, t0.Overall, h0.Overall);
	log.LogInformation("           held-out by length: {PerBits}", PerBitsStr(h0.PerBits));
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
		var batchInputs = trainSet.Inputs.index_select(0, idx);
		var batchTargets = trainSet.Targets.index_select(0, idx);

		optimizer.zero_grad();
		var logits = model.forward(batchInputs);
		var loss = lossFn.forward(logits.reshape(-1, modelVocab), batchTargets.reshape(-1));
		loss.backward();
		optimizer.step();
		epochLoss += loss.item<float>() * count;
	}

	var isLogEpoch = epoch == 0 || (epoch + 1) % LOG_EVERY == 0;
	var isEvalEpoch = epoch == 0 || (epoch + 1) % EVAL_EVERY == 0;

	if (isLogEpoch)
	{
		// Cheap signal every LogEvery: teacher-forced held-out validation loss — one batched forward pass,
		// essentially free. Tracks whether the model is fitting the held-out LENGTHS' next-token structure
		// (the generalization signal) without paying for autoregressive generation.
		var valLoss = TeacherForcedLoss(model, heldOutSet, lossFn, modelVocab);
		log.LogInformation("  epoch {Epoch,5}/{Total}   train-loss {Train:0.0000}   val-loss(held-len) {Val:0.0000}",
			epoch + 1, EPOCHS, epochLoss / exampleN, valLoss);
	}

	if (isEvalEpoch)
	{
		// The real thesis measurement, on a coarser cadence because it is autoregressive (one forward
		// pass per generated token): free-generation execution accuracy, train lengths vs held-out lengths.
		// The gap is the experiment — matched = the tape executes the algorithm; train-only = memorized depth.
		var trainAcc = EvalExecAccuracy(model, trainSet, tokenizer, BLOCK_SIZE, sample: 128);
		// Dump a few held-out generated tapes so the exec-acc metric can be verified against the model's
		// actual output (this is how we caught the ParseSum last-vs-first-sum bug).
		var heldAcc = EvalExecAccuracy(model, heldOutSet, tokenizer, BLOCK_SIZE, sample: 128, dumpLog: log, dumpCount: 3);
		log.LogInformation("           exec-acc  train-len {Train:P1}   held-out-len {Held:P1}", trainAcc.Overall, heldAcc.Overall);
		log.LogInformation("           held-out by length: {PerBits}", PerBitsStr(heldAcc.PerBits));
	}

	// Checkpoint periodically so a stopped run resumes without losing progress. meta stores the NEXT
	// epoch to run (epoch+1), so resume picks up after the saved one.
	if (CHECKPOINT_EVERY > 0 && (epoch + 1) % CHECKPOINT_EVERY == 0)
		SaveCheckpoint(model, optimizer, ckptStem, epoch + 1, log);
}

// Final checkpoint so the fully-trained model is resumable/inspectable.
SaveCheckpoint(model, optimizer, ckptStem, EPOCHS, log);

log.LogInformation("=== Final ===");
LogFinal(model, trainSet, "train-lengths", tokenizer, BLOCK_SIZE, log);
LogFinal(model, heldOutSet, "held-out-lengths", tokenizer, BLOCK_SIZE, log);

return;

// Writes a checkpoint: model weights (.model), Adam moments (.optim), and the next epoch to run (.meta),
// so a stopped run can resume exactly where it left off (weights + optimizer momentum + epoch counter).
static void SaveCheckpoint(MiniGpt model, TorchSharp.Modules.Adam optimizer, string stem, int nextEpoch, ILogger log)
{
	model.save($"{stem}.model");
	optimizer.save_state_dict($"{stem}.optim");
	File.WriteAllText($"{stem}.meta", nextEpoch.ToString());
	log.LogInformation("           [checkpoint saved @ epoch {Epoch}]", nextEpoch);
}

// Teacher-forced validation loss over a dataset: a single batched forward pass, with the same
// continuation-only masking as training (the dataset's targets already ignore prompt + padding). Cheap
// enough to run every log epoch. Chunked over examples to bound memory on the larger held-out set.
static double TeacherForcedLoss(MiniGpt model, TapeDataset data, TorchSharp.Modules.CrossEntropyLoss lossFn, int vocab)
{
	using var noGrad = no_grad();
	model.eval();

	var n = data.Inputs.shape[0];
	const long chunk = 256;
	double weighted = 0;
	long counted = 0;
	for (long start = 0; start < n; start += chunk)
	{
		var count = Math.Min(chunk, n - start);
		var inp = data.Inputs.narrow(0, start, count);
		var tgt = data.Targets.narrow(0, start, count);
		var logits = model.forward(inp);
		// Reduction='mean' ignores pad targets; weight by the number of non-pad targets so the chunked
		// mean equals the true per-token mean over the continuation tokens.
		var nonPad = tgt.ne(data.PadId).sum().item<long>();
		if (nonPad == 0)
			continue;
		var loss = lossFn.forward(logits.reshape(-1, vocab), tgt.reshape(-1)).item<float>();
		weighted += loss * nonPad;
		counted += nonPad;
	}
	return counted == 0 ? 0 : weighted / counted;
}

// Free-generation execution accuracy: from each example's PROMPT, let the model generate the whole tape
// on its own (greedy, no teacher forcing), parse the final `sum` line, and check it equals a+b. This is
// the honest test — the model must ripple carries through its own generated scratchpad.
//
// Batched by bit-width: examples of the same width have equal-length prompts, so each width-group runs as
// one batched generation (one forward pass per step over the whole group) instead of one-sequence-at-a-
// time — the difference between a ~15-minute eval and a sub-minute one. `sample` caps examples per width.
static (double Overall, SortedDictionary<int, double> PerBits) EvalExecAccuracy(
	MiniGpt model, TapeDataset data, CharTokenizer tok, int blockSize, int sample,
	ILogger? dumpLog = null, int dumpCount = 0)
{
	int correct = 0, total = 0;
	var perBits = new SortedDictionary<int, double>(); // bit-width -> exec accuracy (length-decay signal)
	// Collect FAILURES (with how many carry-propagation steps the case needs) so the dump shows where
	// generation actually breaks. carryCount = number of bit positions whose addition produces a carry-out.
	var failures = new List<(int Bits, int A, int B, int CarryCount, bool Parsed, int Pred, string Tape)>();

	foreach (var group in data.Examples.GroupBy(e => e.Bits))
	{
		int groupCorrect = 0, groupTotal = 0;
		var exs = group.ToList();
		var step = sample > 0 && exs.Count > sample ? exs.Count / sample : 1;
		var picked = new List<TapeDataset.Example>();
		for (var i = 0; i < exs.Count; i += step)
			picked.Add(exs[i]);
		if (picked.Count == 0)
			continue;

		var bits = group.Key;
		// Every prompt in a width-group is the same length, so they stack into one [batch, promptLen] tensor.
		var promptIdsList = picked.Select(e => tok.Encode(AdderTape.RenderPrompt(e.A, e.B, e.Bits))).ToList();
		var promptLen = promptIdsList[0].Length;
		var promptData = new long[picked.Count, promptLen];
		for (var r = 0; r < picked.Count; r++)
			for (var c = 0; c < promptLen; c++)
				promptData[r, c] = promptIdsList[r][c];
		var prompts = tensor(promptData);

		// Budget: a b-bit tape's continuation is ~b work-lines (~20 tok each) + a sum line. Cap by block.
		var budget = Math.Min(blockSize - promptLen - 1, 22 * (bits + 1));
		var generated = model.GenerateBatch(prompts, budget); // [batch, promptLen + budget]

		for (var r = 0; r < picked.Count; r++)
		{
			var row = generated[r].data<long>().ToArray();
			var text = tok.Decode(row);
			var parsed = ParseSum(text, out var predicted);
			var ex = picked[r];
			var ok = parsed && predicted == ex.A + ex.B;
			if (ok)
			{
				correct++;
				groupCorrect++;
			}
			total++;
			groupTotal++;

			if (dumpLog is not null && !ok)
				failures.Add((ex.Bits, ex.A, ex.B, CarryCount(ex.A, ex.B, ex.Bits), parsed, parsed ? predicted : -1, text.Replace('\n', '/')));
		}

		perBits[bits] = groupTotal == 0 ? 0 : (double)groupCorrect / groupTotal;
	}

	// Dump the most carry-heavy failures — the ones needing the most carry propagation, which is exactly
	// the sequential-depth operation the tape is meant to enable. Shows WHERE generation breaks.
	if (dumpLog is not null && failures.Count > 0)
	{
		dumpLog.LogInformation("    {Fail} failures; most carry-heavy:", failures.Count);
		foreach (var f in failures.OrderByDescending(f => f.CarryCount).ThenByDescending(f => f.Bits).Take(dumpCount))
			dumpLog.LogInformation("    [{Bits}b carries={C}] a={A} b={B} expect={Exp} parsed={Ok}:{Pred}  tape: {Tape}",
				f.Bits, f.CarryCount, f.A, f.B, f.A + f.B, f.Parsed, f.Pred, f.Tape);
	}

	return (total == 0 ? 0 : (double)correct / total, perBits);
}

// Counts bit positions whose column sum (a_k + b_k + carry_in) produces a carry-out — i.e. how many
// sequential carry-propagation steps the addition genuinely requires. 0 = trivial (no carrying).
static int CarryCount(int a, int b, int bits)
{
	int carry = 0, count = 0;
	for (var k = 0; k < bits; k++)
	{
		var full = ((a >> k) & 1) + ((b >> k) & 1) + carry;
		carry = full >> 1;
		if (carry == 1)
			count++;
	}
	return count;
}

// Parses the sum straight from the generated completion's per-step `s` digits (LSB-first), plus the final
// carry from the `end <carry>` record — the streaming format has no separate sum line to gather. Reads
// records in order from the first work record up to (and including) `end`, stopping at the `.` terminator,
// so it takes the model's FIRST tape and ignores any overflow generation past `.`.
//
// A record is well-formed as `a b c<n> s<d> k<n>`; we pull the `s` digit from each, and `end <carry>` adds
// the high bit. Returns false if the tape never reaches a valid `end` record (e.g. ran out of budget).
static bool ParseSum(string tape, out int value)
{
	value = 0;
	var acc = 0;
	var bit = 0;
	foreach (var raw in tape.Split('\n'))
	{
		var line = raw.Trim();
		if (line.Length == 0)
			continue;
		if (line == ".")
			break;

		if (line.StartsWith("end ", StringComparison.Ordinal))
		{
			// Final carry-out is the top sum bit; then the answer is complete.
			var c = line["end ".Length..].Trim();
			if (c != "0" && c != "1")
				return false;
			if (c == "1")
				acc |= 1 << bit;
			value = acc;
			return true;
		}

		// A work record: extract the s<d> token. Skip anything that isn't a well-formed record (e.g. the
		// prompt's bare "a b" pairs or "go" if they appear in the decoded text before the work records).
		var sTok = line.Split(' ', StringSplitOptions.RemoveEmptyEntries).FirstOrDefault(t => t.Length == 2 && t[0] == 's');
		if (sTok is null)
			continue;
		if (sTok[1] == '1')
			acc |= 1 << bit;
		else if (sTok[1] != '0')
			return false;
		bit++;
	}
	return false; // never saw a valid `end` record
}

static void LogFinal(MiniGpt model, TapeDataset data, string label, CharTokenizer tok, int blockSize, ILogger log)
{
	var (overall, perBits) = EvalExecAccuracy(model, data, tok, blockSize, sample: 0); // full set
	log.LogInformation("  {Label,-18} exec-accuracy {Acc:P2}  (over {N} examples)  by length: {PerBits}",
		label, overall, data.Examples.Count, PerBitsStr(perBits));
}

// Formats a bit-width -> accuracy map as "5b:88% 6b:41% ..." — the length-decay curve at a glance.
static string PerBitsStr(SortedDictionary<int, double> perBits) =>
	string.Join("  ", perBits.Select(kv => $"{kv.Key}b:{kv.Value:P0}"));

static ILoggerFactory CreateLoggerFactory(IConfiguration configuration)
{
	var runStamp = DateTime.Now.ToString("yyyyMMdd-HHmmss");
	var logPath = Path.Combine(AppContext.BaseDirectory, "logs", $"adder-tape-{runStamp}.log");
	var serilogLogger = new Serilog.LoggerConfiguration()
		.ReadFrom.Configuration(configuration)
		.WriteTo.File(logPath, outputTemplate: "{Timestamp:yyyy-MM-dd HH:mm:ss.fff zzz} [{Level:u3}] {Message:lj}{NewLine}{Exception}")
		.CreateLogger();
	return new SerilogLoggerFactory(serilogLogger, dispose: true);
}
