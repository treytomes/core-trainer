using CoreTrainer.Examples.MiniGptScratch;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Serilog;
using Serilog.Extensions.Logging;
// Disambiguate ILogger (both MS and Serilog define one); the example logs via the MS abstraction.
using ILogger = Microsoft.Extensions.Logging.ILogger;

// ---------------------------------------------------------------------------------------
// Mini-GPT from scratch — the same GPT-2-style transformer as CoreTrainer.Examples.MiniGpt, but
// with NO tensor library. Every matrix multiply, softmax, LayerNorm, attention pattern, their
// GRADIENTS, and the Adam update are hand-written and steppable. It exists to reveal exactly the
// algorithms TorchSharp hides behind `.backward()` and `optimizer.step()`.
//
// The program does three things:
//   1. A GRADIENT CHECK — compares the analytic backprop gradient against a numerical
//      finite-difference estimate on a tiny model, proving the hand-rolled backward pass is correct.
//   2. TRAINING — memorizes the same handful of sentences via next-token prediction + Adam.
//   3. A MEMORIZATION TEST — prompts with each sentence's opening and checks verbatim reproduction.
// ---------------------------------------------------------------------------------------

using var loggerFactory = CreateLoggerFactory();
var log = loggerFactory.CreateLogger("MiniGptScratch");

// === 1. Gradient check ===============================================================
// Before trusting training, verify backprop numerically. For a single weight w, the analytic
// gradient dL/dw (from Backward) must match (L(w+h) − L(w−h)) / 2h for small h. Agreement to a few
// significant figures is strong evidence every hand-derived backward is correct.
GradientCheck(log);

// === 2. Training =====================================================================
string[] sentences =
[
	"the quick brown fox jumps over the lazy dog.",
	"a journey of a thousand miles begins with a single step.",
	"to be or not to be that is the question.",
	"the mitochondria is the powerhouse of the cell.",
	"all that glitters is not gold.",
];
var corpus = string.Join("\n", sentences) + "\n";

const int BLOCK_SIZE = 64;
const int EMBED_DIM = 64;
const int HEADS = 4;
const int LAYERS = 2;
const int EPOCHS = 2000;
const double LEARNING_RATE = 3e-3;
const int LOG_EVERY = 100;
const int SEED = 1;

var tokenizer = new CharTokenizer(corpus);
var padId = tokenizer.VocabSize;             // reserved id beyond the real vocab, used for the loss ignore
var modelVocab = tokenizer.VocabSize + 1;

// One (input, target) example per sentence — each sentence is a first-class, equally-weighted
// target (the sliding-window scheme under-trains the final sentence; per-sentence avoids that).
var examples = BuildExamples(sentences, tokenizer);

var rng = new Random(SEED);
var model = new ScratchGpt(modelVocab, BLOCK_SIZE, EMBED_DIM, HEADS, LAYERS, rng);
var optimizer = new AdamOptimizer(model.Parameters(), LEARNING_RATE);

log.LogInformation("=== Mini-GPT (from scratch) Trainer ===");
log.LogInformation("Corpus:   {Sentences} sentences, {Chars} chars, vocab {Vocab}", sentences.Length, corpus.Length, tokenizer.VocabSize);
log.LogInformation("Model:    dim {Dim}, {Heads} heads, {Layers} layers, block {Block}", EMBED_DIM, HEADS, LAYERS, BLOCK_SIZE);
log.LogInformation("Train:    {Examples} sentences, Adam lr {Lr}, {Epochs} epochs", examples.Count, LEARNING_RATE, EPOCHS);

for (var epoch = 0; epoch < EPOCHS; epoch++)
{
	optimizer.ZeroGrad();
	var epochLoss = 0.0;

	// Accumulate gradients across all sentences, then take one Adam step (full-batch).
	foreach (var (input, target) in examples)
	{
		var logits = model.Forward(input);
		var (loss, dLogits) = Ops.CrossEntropy(logits, target);
		model.Backward(dLogits);
		epochLoss += loss;
	}

	optimizer.Step();

	if (epoch == 0 || (epoch + 1) % LOG_EVERY == 0)
		log.LogInformation("  epoch {Epoch,5}/{Total}   loss {Loss:0.0000}", epoch + 1, EPOCHS, epochLoss / examples.Count);
}

// === 3. Memorization test ============================================================
log.LogInformation("=== Memorization test (prompt -> greedy continuation) ===");
var passed = 0;
foreach (var sentence in sentences)
{
	var promptText = sentence[..Math.Min(10, sentence.Length)];
	var prompt = tokenizer.Encode(promptText);
	var generatedIds = model.Generate(prompt, maxNewTokens: sentence.Length - promptText.Length + 1);
	var decoded = tokenizer.Decode(generatedIds);

	var newlineAt = decoded.IndexOf('\n');
	var produced = newlineAt >= 0 ? decoded[..newlineAt] : decoded;

	var ok = produced == sentence;
	if (ok)
		passed++;

	log.LogInformation("  [{Mark}] prompt \"{Prompt}...\"", ok ? "PASS" : "FAIL", promptText);
	log.LogInformation("        expected: {Expected}", sentence);
	log.LogInformation("        produced: {Produced}", produced);
}

log.LogInformation("Memorization: {Passed}/{Total} sentences reproduced verbatim.", passed, sentences.Length);
if (passed == sentences.Length)
	log.LogInformation("SUCCESS: the from-scratch model memorized every sentence.");

return;

// === Helpers ===========================================================================

// Builds one next-token example per sentence: input = ids[0..n-2], target = ids[1..n-1] (with the
// trailing newline as a stop token). Targets are int[] with -1 marking positions to ignore (none
// here, since each example is exactly its sentence length).
static List<(int[] Input, int[] Target)> BuildExamples(string[] sentences, CharTokenizer tokenizer)
{
	var examples = new List<(int[], int[])>();
	foreach (var sentence in sentences)
	{
		var ids = tokenizer.Encode(sentence + "\n");
		var input = new int[ids.Length - 1];
		var target = new int[ids.Length - 1];
		for (var i = 0; i < ids.Length - 1; i++)
		{
			input[i] = ids[i];
			target[i] = ids[i + 1];
		}
		examples.Add((input, target));
	}
	return examples;
}

// Verifies the hand-rolled backward pass against numerical finite differences on a tiny model, so a
// wrong gradient is caught here rather than silently crippling training. Perturbs a handful of
// weights across different parameter tensors and compares dL/dw (analytic) to the central difference.
static void GradientCheck(ILogger log)
{
	log.LogInformation("=== Gradient check (analytic vs numerical) ===");

	var rng = new Random(42);
	// A tiny model + short sequence keeps the finite-difference sweep cheap.
	const int vocab = 6;
	var model = new ScratchGpt(vocabSize: vocab, blockSize: 8, embedDim: 8, heads: 2, layers: 1, rng);
	int[] input = [1, 3, 2, 4];
	int[] target = [3, 2, 4, 5];

	// Analytic gradient from one forward/backward.
	var parameters = model.Parameters();
	foreach (var p in parameters)
		p.ZeroGrad();
	var logits = model.Forward(input);
	var (baseLoss, dLogits) = Ops.CrossEntropy(logits, target);
	model.Backward(dLogits);

	// Loss as a pure function of the current weights (for finite differences).
	double LossAt() => Ops.CrossEntropy(model.Forward(input), target).Loss;

	const double h = 1e-5;
	var maxRelErr = 0.0;
	var checkm = 0;

	// Sample a few weights from the first few parameter tensors.
	foreach (var p in parameters.Take(6))
	{
		var data = p.Value.Data;
		var step = Math.Max(1, data.Length / 3);
		for (var i = 0; i < data.Length && checkm < 20; i += step)
		{
			var original = data[i];
			var analytic = p.Grad.Data[i];

			data[i] = original + h;
			var lossPlus = LossAt();
			data[i] = original - h;
			var lossMinus = LossAt();
			data[i] = original;

			var numeric = (lossPlus - lossMinus) / (2.0 * h);
			var denom = Math.Max(1e-8, Math.Abs(analytic) + Math.Abs(numeric));
			var relErr = Math.Abs(analytic - numeric) / denom;
			maxRelErr = Math.Max(maxRelErr, relErr);
			checkm++;
		}
	}

	log.LogInformation("  base loss {Loss:0.0000}, checked {N} weights, max relative error {Err:0.###E+0}",
		baseLoss, checkm, maxRelErr);
	if (maxRelErr < 1e-4)
		log.LogInformation("  PASS: analytic gradients match numerical (max rel err < 1e-4).");
	else
		log.LogWarning("  WARN: gradient mismatch (max rel err {Err:0.###E+0}) — backprop may be wrong.", maxRelErr);
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
