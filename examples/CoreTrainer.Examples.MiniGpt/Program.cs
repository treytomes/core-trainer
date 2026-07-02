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

// The sentences to memorize. Each ends with a newline so the model learns a natural stop token.
string[] sentences =
[
	"the quick brown fox jumps over the lazy dog.",
	"a journey of a thousand miles begins with a single step.",
	"to be or not to be that is the question.",
	"the mitochondria is the powerhouse of the cell.",
	"all that glitters is not gold.",
];
var corpus = string.Join("\n", sentences) + "\n";

// === Hyperparameters =================================================================
const int BLOCK_SIZE = 64;   // context length; comfortably longer than our longest sentence
const int EMBED_DIM = 128;
const int HEADS = 4;
const int LAYERS = 3;
const int EPOCHS = 3000;
const double LEARNING_RATE = 3e-4;
const int LOG_EVERY = 200;
const int SEED = 1;
// Logic-MLP temperature anneal (ignored by the GELU baseline): hot start = smooth gate/wire blends and
// easy gradients; cold end = selections near one-hot so the soft net approaches its hard circuit.
const double TEMP_START = 2.0;
const double TEMP_END = 0.1;

using var loggerFactory = CreateLoggerFactory();
var log = loggerFactory.CreateLogger("MiniGpt");

manual_seed(SEED);

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
log.LogInformation("Train:    {Examples} windows, Adam lr {Lr}, {Epochs} epochs", exampleCount, LEARNING_RATE, EPOCHS);

// Model vocabulary includes the reserved pad id (index tokenizer.VocabSize), so logits cover it.
var modelVocab = tokenizer.VocabSize + 1;

// Run all three feed-forward variants under identical conditions so the comparison is clean: the same
// corpus, seed, epochs, and LR — only the block's nonlinearity differs. GELU is the baseline; the logic
// variants swap in a differentiable logic-gate network (soft = fully differentiable, hard = STE circuit).
var modes = new[] { MlpMode.Gelu, MlpMode.LogicSoft, MlpMode.LogicHard };
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

	var model = new MiniGpt(modelVocab, BLOCK_SIZE, EMBED_DIM, HEADS, LAYERS, mode, TEMP_START);
	var optimizer = optim.Adam(model.parameters(), lr: LEARNING_RATE);
	// ignore_index = padId: padded target positions contribute nothing to the loss, so the model is
	// trained only on real next-token predictions, not on producing padding.
	var lossFn = nn.CrossEntropyLoss(ignore_index: padId);

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
		optimizer.zero_grad();

		var logits = model.forward(inputs);                   // [examples, block, vocab]
		var flatLogits = logits.reshape(-1, modelVocab);      // CE wants [N, vocab] vs [N] targets
		var flatTargets = targets.reshape(-1);
		var loss = lossFn.forward(flatLogits, flatTargets);

		loss.backward();
		optimizer.step();

		lastLoss = loss.item<float>();
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
