using CoreTrainer.Examples.MiniGpt;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Serilog;
using Serilog.Extensions.Logging;
using TorchSharp;
using static TorchSharp.torch;
// Disambiguate ILogger (both MS and Serilog define one); the example logs via the MS abstraction.
using ILogger = Microsoft.Extensions.Logging.ILogger;

// ---------------------------------------------------------------------------------------
// Mini-GPT — a small GPT-2-style transformer trained to memorize a handful of sentences.
//
// This example builds a decoder-only transformer (token + positional embeddings, causal
// self-attention blocks, an LM head) with TorchSharp, trains it with next-token prediction on a
// tiny fixed corpus, and then tests that it actually learned the sentences by prompting with each
// one's opening words and checking it reproduces the rest verbatim.
//
// Unlike a real language model, the GOAL here is to overfit: with a fixed handful of sentences,
// success means the model memorizes them exactly (training loss -> ~0, greedy decoding reproduces
// each sentence). Dropout is off and we train until the loss collapses.
// ---------------------------------------------------------------------------------------

// Configuration (model shape, optimization, logging) comes from appsettings.yaml; the corpus (one
// sentence per line) comes from corpus.txt. Both are copied next to the binary at build.
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
var LOG_EVERY = training.GetValue<int>("LogEvery");
var SEED = training.GetValue<int>("Seed");
var BATCH_SIZE = training.GetValue<int>("BatchSize");
// Logic-MLP temperature anneal (ignored by the GELU baseline): hot start = smooth gate/wire blends and
// easy gradients; cold end = selections near one-hot so the soft net approaches its hard circuit.
var TEMP_START = training.GetValue<double>("TempStart");
var TEMP_END = training.GetValue<double>("TempEnd");
// All linears + embeddings become BitNet ternary {-1,0,+1} weights (aggressive: includes the head and
// embedding tables that BitNet normally leaves full precision).
var USE_BITLINEAR = training.GetValue<bool>("UseBitLinear");
// Also ternarize the embeddings + LM head (false = standard BitNet recipe with full-precision I/O).
var USE_BITLINEAR_IO = training.GetValue<bool>("UseBitLinearIO");

using var loggerFactory = CreateLoggerFactory(configuration);
var log = loggerFactory.CreateLogger("MiniGpt");

manual_seed(SEED);

// Use all available cores for libtorch's intra-op matmul threading (pure speed, no effect on training
// dynamics). Without this, the CPU backend can default to a conservative thread count.
set_num_threads(Environment.ProcessorCount);

// The sentences to memorize, one per line of corpus.txt. Blank lines are ignored; each sentence gets a
// newline "stop" token when built into an example.
var sentences = File.ReadAllLines(Path.Combine(AppContext.BaseDirectory, "corpus.txt"))
	.Where(line => line.Trim().Length > 0)
	.ToArray();
if (sentences.Length == 0)
	throw new InvalidOperationException("corpus.txt is empty — add at least one sentence to memorize.");
var corpus = string.Join("\n", sentences) + "\n";

var tokenizer = new CharTokenizer(corpus);

// Build one training example per sentence (each terminated by a newline "stop" token), rather than
// sliding fixed windows across the concatenated corpus. Per-sentence examples ensure every
// sentence — including the last — is learned as a first-class, equally-weighted target; sliding
// windows under-represent spans near the end of the corpus (there is no full window starting there),
// which leaves the final sentence under-trained.
var (inputs, targets, padId) = BuildExamples(sentences, tokenizer, BLOCK_SIZE);
var exampleCount = inputs.shape[0];

log.LogInformation("=== Mini-GPT Trainer ===");
log.LogInformation("Corpus:   {Sentences} sentences, {Chars} chars, vocab {Vocab}", sentences.Length, corpus.Length, tokenizer.VocabSize);
log.LogInformation("Model:    dim {Dim}, {Heads} heads, {Layers} layers, block {Block}", EMBED_DIM, HEADS, LAYERS, BLOCK_SIZE);
log.LogInformation("Weights:  blocks {Blocks}, I/O (embeddings+head) {Io}",
	USE_BITLINEAR ? "ternary {-1,0,+1}" : "full precision",
	USE_BITLINEAR_IO ? "ternary {-1,0,+1}" : "full precision");
log.LogInformation("Train:    {Examples} windows, Adam lr {Lr}, {Epochs} epochs", exampleCount, LEARNING_RATE, EPOCHS);

// Model vocabulary includes the reserved pad id (index tokenizer.VocabSize), so logits cover it.
var modelVocab = tokenizer.VocabSize + 1;

// Run all three feed-forward variants under identical conditions so the comparison is clean: the same
// corpus, seed, epochs, and LR — only the block's nonlinearity differs. GELU is the baseline; the logic
// variants swap in a differentiable logic-gate network (soft = fully differentiable, hard = STE circuit).
// var modes = new[] { MlpMode.Gelu, MlpMode.LogicSoft, MlpMode.LogicHard };
var modes = new[] { MlpMode.LogicSoft, MlpMode.LogicHard };
var summary = new List<(MlpMode Mode, double FinalLoss, int Passed)>();

foreach (var mode in modes)
	summary.Add(RunMode(mode));

// Side-by-side verdict. For a language task the bar is NOT an exact circuit (as it would be for a
// standalone adder) but parity with the GELU baseline: comparable loss and memorization count.
log.LogInformation("=== Feed-forward comparison ===");
foreach (var (mode, finalLoss, passed) in summary)
	log.LogInformation("  {Mode,-10}  final loss {Loss:0.0000}   memorized {Passed}/{Total}",
		mode, finalLoss, passed, sentences.Length);

return;

// Builds, trains, and memorization-tests one model with the given feed-forward mode; returns its final
// loss and memorization count. The logic modes anneal their softmax temperature from hot to cold across
// training (like the standalone LogicMlp ladder) so wire/gate selections sharpen gradually; GELU ignores it.
(MlpMode Mode, double FinalLoss, int Passed) RunMode(MlpMode mode)
{
	manual_seed(SEED); // reset per mode so every variant sees identical initialization draws
	log.LogInformation("########## Feed-forward mode: {Mode} ##########", mode);

	var model = new MiniGpt(modelVocab, BLOCK_SIZE, EMBED_DIM, HEADS, LAYERS, mode, TEMP_START, USE_BITLINEAR, USE_BITLINEAR_IO);

	// Report the parameter budget so the three feed-forward variants are comparable at a glance (the
	// logic MLP's learnable wire/gate logits replace the GELU MLP's two weight matrices).
	long paramCount = model.parameters().Sum(p => p.numel());
	log.LogInformation("Params:   {Count:N0} learnable ({Millions:0.00} M)", paramCount, paramCount / 1_000_000.0);

	var optimizer = optim.Adam(model.parameters(), lr: LEARNING_RATE);
	// ignore_index = padId: padded target positions contribute nothing to the loss, so the model is
	// trained only on real next-token predictions, not on producing padding.
	var lossFn = nn.CrossEntropyLoss(ignore_index: padId);

	var exampleN = inputs.shape[0];
	var lastLoss = 0.0;
	for (var epoch = 0; epoch < EPOCHS; epoch++)
	{
		// Anneal logic temperature geometrically (no-op for GELU). Hot early = smooth blends and easy
		// gradients; cold late = selections near one-hot so soft training approaches the hard circuit.
		if (mode != MlpMode.Gelu)
		{
			var progress = (double)epoch / (EPOCHS - 1);
			model.SetLogicTemperature(TEMP_START * Math.Pow(TEMP_END / TEMP_START, progress));
		}

		model.train();

		// Mini-batch training: shuffle the corpus and take one Adam step per batch, so each epoch runs
		// ceil(N / BatchSize) steps instead of a single full-batch step. More (noisier) steps per epoch
		// converge faster than full-batch on a fixed dataset. randperm under the manual seed keeps it
		// reproducible. epochLoss is the batch-size-weighted mean, comparable to the old full-batch loss.
		var perm = randperm(exampleN);
		var epochLoss = 0.0;
		for (long start = 0; start < exampleN; start += BATCH_SIZE)
		{
			var count = Math.Min(BATCH_SIZE, exampleN - start);
			var idx = perm.narrow(0, start, count);
			var batchInputs = inputs.index_select(0, idx);
			var batchTargets = targets.index_select(0, idx);

			optimizer.zero_grad();
			var logits = model.forward(batchInputs);              // [batch, block, vocab]
			var flatLogits = logits.reshape(-1, modelVocab);      // CE wants [N, vocab] vs [N] targets
			var flatTargets = batchTargets.reshape(-1);
			var loss = lossFn.forward(flatLogits, flatTargets);

			loss.backward();
			optimizer.step();

			epochLoss += loss.item<float>() * count;              // weight by batch size for a true mean
		}

		lastLoss = epochLoss / exampleN;
		if (epoch == 0 || (epoch + 1) % LOG_EVERY == 0)
			log.LogInformation("  epoch {Epoch,5}/{Total}   loss {Loss:0.0000}", epoch + 1, EPOCHS, lastLoss);
	}

	// Memorization test: prompt each sentence's opening and greedily generate the rest; a memorized
	// corpus reproduces each verbatim.
	log.LogInformation("=== [{Mode}] Memorization test ===", mode);
	var passed = 0;
	foreach (var sentence in sentences)
	{
		var promptText = sentence[..Math.Min(10, sentence.Length)];
		var promptIds = tokenizer.Encode(promptText).ToArray();
		var prompt = tensor(promptIds, dtype: ScalarType.Int64).reshape(1, promptIds.Length);

		var generated = model.Generate(prompt, maxNewTokens: sentence.Length - promptText.Length + 1);
		var generatedIds = generated.reshape(-1).data<long>().ToArray();
		var decoded = tokenizer.Decode(generatedIds);

		var newlineAt = decoded.IndexOf('\n');
		var produced = newlineAt >= 0 ? decoded[..newlineAt] : decoded;

		var ok = produced == sentence;
		if (ok)
			passed++;
		log.LogInformation("  [{Mark}] \"{Prompt}...\" -> {Produced}", ok ? "PASS" : "FAIL", promptText, produced);
	}
	log.LogInformation("[{Mode}] Memorization: {Passed}/{Total}", mode, passed, sentences.Length);

	// Cullability: how much of the trained logic MLP is real logic vs prunable structure (constants /
	// passthroughs). Earns-or-refutes the "the MLP became cullable logic" claim. No-op for GELU.
	if (mode != MlpMode.Gelu)
	{
		log.LogInformation("=== [{Mode}] Logic-MLP cullability ===", mode);
		model.ReportLogicCullability(log);
	}
	// Compile step: export the trained ternary weights as packed trits and report the deployable
	// artifact's size/compression AND sparsity (weights trained to exactly zero). No-op / "nothing to
	// compile" for a full-precision model. Bit-identical to inference.
	TernaryCompiler.CompileAndReport(model, log);

	model.Dispose();
	return (mode, lastLoss, passed);
}

// === Helpers ===========================================================================

// Builds one next-token-prediction example per sentence, each padded to blockSize. For a sentence
// with token ids t0..t_{n-1} (the trailing newline included as a stop token), the input is
// t0..t_{n-2} and the target is t1..t_{n-1} (each position predicts the following token). Both are
// right-padded to blockSize with a reserved pad id; the target's pad positions use the pad id as an
// ignore_index so padding never becomes a training target. Returns the batched inputs/targets and
// the pad id (which is tokenizer.VocabSize — a token index reserved beyond the real vocabulary).
static (Tensor Inputs, Tensor Targets, long PadId) BuildExamples(string[] sentences, CharTokenizer tokenizer, int blockSize)
{
	var padId = (long)tokenizer.VocabSize;
	var inputRows = new List<long[]>();
	var targetRows = new List<long[]>();

	foreach (var sentence in sentences)
	{
		var ids = tokenizer.Encode(sentence + "\n").ToArray();
		if (ids.Length - 1 > blockSize)
			throw new ArgumentException($"Sentence too long ({ids.Length} tokens) for block size {blockSize}.");

		var input = new long[blockSize];
		var target = new long[blockSize];
		Array.Fill(input, padId);
		Array.Fill(target, padId);

		// input[i] = ids[i], target[i] = ids[i+1], for i in 0..ids.Length-2.
		for (var i = 0; i < ids.Length - 1; i++)
		{
			input[i] = ids[i];
			target[i] = ids[i + 1];
		}

		inputRows.Add(input);
		targetRows.Add(target);
	}

	var inputs = stack(inputRows.Select(r => tensor(r, dtype: ScalarType.Int64)).ToArray());
	var targets = stack(targetRows.Select(r => tensor(r, dtype: ScalarType.Int64)).ToArray());
	return (inputs, targets, padId);
}

static ILoggerFactory CreateLoggerFactory(IConfiguration configuration)
{
	// Console sink + levels come from appsettings.yaml; the file sink is added here with a per-run
	// timestamped filename so each run writes its own log rather than stacking into one day-rolling file.
	var runStamp = DateTime.Now.ToString("yyyyMMdd-HHmmss");
	var logPath = Path.Combine(AppContext.BaseDirectory, "logs", $"mini-gpt-{runStamp}.log");

	var serilogLogger = new Serilog.LoggerConfiguration()
		.ReadFrom.Configuration(configuration)
		.WriteTo.File(
			path: logPath,
			outputTemplate: "{Timestamp:yyyy-MM-dd HH:mm:ss.fff zzz} [{Level:u3}] {Message:lj}{NewLine}{Exception}")
		.CreateLogger();

	return new SerilogLoggerFactory(serilogLogger, dispose: true);
}
