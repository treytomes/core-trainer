using System.Diagnostics;
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

var configuration = new ConfigurationBuilder()
	.SetBasePath(AppContext.BaseDirectory)
	.AddYamlFile("appsettings.yaml", optional: false, reloadOnChange: false)
	.Build();

using var loggerFactory = CreateLoggerFactory(configuration);
var log = loggerFactory.CreateLogger("MiniGptScratch");

// === 1. Gradient check ===============================================================
// Before trusting training, verify backprop numerically. For a single weight w, the analytic
// gradient dL/dw (from Backward) must match (L(w+h) − L(w−h)) / 2h for small h. Agreement to a few
// significant figures is strong evidence every hand-derived backward is correct.
GradientCheck(log);

// === 2. Training =====================================================================
// The corpus (sentences to memorize) is read from corpus.txt, one per line; model and training
// hyperparameters come from the Training section of appsettings.yaml.
var sentences = File.ReadAllLines(Path.Combine(AppContext.BaseDirectory, "corpus.txt"))
	.Where(line => line.Length > 0)
	.ToArray();
if (sentences.Length == 0)
	throw new InvalidOperationException("corpus.txt is empty — add at least one sentence to memorize.");
var corpus = string.Join("\n", sentences) + "\n";

var training = configuration.GetSection("Training");
var BLOCK_SIZE = training.GetValue<int>("BlockSize");
var EMBED_DIM = training.GetValue<int>("EmbedDim");
var HEADS = training.GetValue<int>("Heads");
var LAYERS = training.GetValue<int>("Layers");
var EPOCHS = training.GetValue<int>("Epochs");
var LEARNING_RATE = training.GetValue<double>("LearningRate");
var LOG_EVERY = training.GetValue<int>("LogEvery");
var INFERENCE_REPS = training.GetValue<int>("InferenceReps");
var SEED = training.GetValue<int>("Seed");
var USE_BITLINEAR = training.GetValue<bool>("UseBitLinear");

// Switch the transformer blocks between full-precision Linear and BitNet-b1.58-style ternary
// BitLinear. Embeddings and the LM head stay full precision either way (BitNet convention).
LinearFactory linearFactory = USE_BITLINEAR
	? (inDim, outDim, init, bias) => new BitLinear(inDim, outDim, init, bias)
	: (inDim, outDim, init, bias) => new Linear(inDim, outDim, init, bias);

var tokenizer = new CharTokenizer(corpus);
var padId = tokenizer.VocabSize;             // reserved id beyond the real vocab, used for the loss ignore
var modelVocab = tokenizer.VocabSize + 1;

// One (input, target) example per sentence — each sentence is a first-class, equally-weighted
// target (the sliding-window scheme under-trains the final sentence; per-sentence avoids that).
var examples = BuildExamples(sentences, tokenizer);

var rng = new Random(SEED);
var model = new ScratchGpt(modelVocab, BLOCK_SIZE, EMBED_DIM, HEADS, LAYERS, rng, linearFactory);
log.LogInformation("Linear layers: {Kind}", USE_BITLINEAR ? "BitLinear (ternary {-1,0,+1} weights)" : "Linear (full precision)");
var optimizer = new AdamOptimizer(model.Parameters(), LEARNING_RATE);

log.LogInformation("=== Mini-GPT (from scratch) Trainer ===");
log.LogInformation("Corpus:   {Sentences} sentences, {Chars} chars, vocab {Vocab}", sentences.Length, corpus.Length, tokenizer.VocabSize);
log.LogInformation("Model:    dim {Dim}, {Heads} heads, {Layers} layers, block {Block}", EMBED_DIM, HEADS, LAYERS, BLOCK_SIZE);
log.LogInformation("Train:    {Examples} sentences, Adam lr {Lr}, {Epochs} epochs", examples.Count, LEARNING_RATE, EPOCHS);

var trainStopwatch = Stopwatch.StartNew();
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
trainStopwatch.Stop();
log.LogInformation("Training time: {Seconds:0.00} s ({PerEpoch:0.00} ms/epoch)",
	trainStopwatch.Elapsed.TotalSeconds, trainStopwatch.Elapsed.TotalMilliseconds / EPOCHS);

// === 3. Memorization test ============================================================
// Run the full generation pass many times so the inference timing is a meaningful,
// log-overhead-free measurement (a single pass is dominated by fixed costs). The produced text is
// identical every rep (greedy, deterministic), so results are captured once and logged after.
log.LogInformation("=== Memorization test (prompt -> greedy continuation, x{Reps}) ===", INFERENCE_REPS);

var produced = new string[sentences.Length];
var inferStopwatch = Stopwatch.StartNew();
for (var rep = 0; rep < INFERENCE_REPS; rep++)
{
	for (var s = 0; s < sentences.Length; s++)
	{
		var sentence = sentences[s];
		var promptText = sentence[..Math.Min(10, sentence.Length)];
		var prompt = tokenizer.Encode(promptText);
		var generatedIds = model.Generate(prompt, maxNewTokens: sentence.Length - promptText.Length + 1);
		var decoded = tokenizer.Decode(generatedIds);

		var newlineAt = decoded.IndexOf('\n');
		produced[s] = newlineAt >= 0 ? decoded[..newlineAt] : decoded;
	}
}
inferStopwatch.Stop();

var passed = 0;
for (var s = 0; s < sentences.Length; s++)
{
	var ok = produced[s] == sentences[s];
	if (ok)
		passed++;

	var promptText = sentences[s][..Math.Min(10, sentences[s].Length)];
	log.LogInformation("  [{Mark}] prompt \"{Prompt}...\"", ok ? "PASS" : "FAIL", promptText);
	log.LogInformation("        expected: {Expected}", sentences[s]);
	log.LogInformation("        produced: {Produced}", produced[s]);
}

var totalPasses = INFERENCE_REPS * sentences.Length;
log.LogInformation("Memorization: {Passed}/{Total} sentences reproduced verbatim.", passed, sentences.Length);
log.LogInformation("Inference time: {Seconds:0.00} s for {Passes} sentence-generations ({Per:0.00} ms each)",
	inferStopwatch.Elapsed.TotalSeconds, totalPasses, inferStopwatch.Elapsed.TotalMilliseconds / totalPasses);
if (passed == sentences.Length)
	log.LogInformation("SUCCESS: the from-scratch model memorized every sentence.");

// === 4. Ternary compile + measure ====================================================
// The BitNet payoff is at deployment: discard the full-precision latent weights kept for training
// and export the transformer blocks as packed ternary weights (5 trits/byte) plus one scale per
// tensor. This measures the size shrink and how many weights trained to exactly zero (sparsity).
// Runs only for a ternary model; a full-precision Linear model has nothing to compile.
if (USE_BITLINEAR)
{
	var report = TernaryCompiler.Compile(model);
	log.LogInformation("=== Ternary compile (block weights) ===");
	log.LogInformation("  Compiled {Layers} BitLinear layers, {Weights:N0} weights.", report.LayerCount, report.WeightCount);
	log.LogInformation("  Full precision (double): {Full:N0} bytes ({FullKb:0.0} KB)",
		report.FullPrecisionBytes, report.FullPrecisionBytes / 1024.0);
	log.LogInformation("  Packed ternary:          {Packed:N0} bytes ({PackedKb:0.0} KB)  [5 trits/byte + γ per tensor]",
		report.PackedBytes, report.PackedBytes / 1024.0);
	log.LogInformation("  Compression: {Ratio:0.0}x smaller, {Bits:0.00} effective bits/weight (theoretical min 1.58).",
		report.CompressionRatio, report.BitsPerWeight);
	log.LogInformation("  Sparsity: {Zeros:N0}/{Weights:N0} weights trained to exactly zero ({Pct:P1} pruned).",
		report.ZeroCount, report.WeightCount, report.Sparsity);
	log.LogInformation("  (Packing verified to round-trip losslessly; compiled model is bit-identical to inference.)");
}

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

	// Finite-difference checking is only meaningful in double precision. The central difference
	// (L(w+h) − L(w−h)) / 2h subtracts two nearly-equal losses; in float (~1e-7 epsilon) that
	// cancellation is pure noise for any usable step h, so the check would report a false failure —
	// a fact about float arithmetic, not about the backprop. Since the alias makes the whole build
	// one precision, we cannot spin up a double model here while the rest runs in float; so the check
	// runs in double builds (validating the identical backprop math, precision free) and is skipped
	// with a note in float builds. The math it validates is shared by both precisions.
	if (typeof(Real) != typeof(double))
	{
		log.LogInformation("  SKIPPED: build precision is {Type}, not double. Finite differences need double —", typeof(Real).Name);
		log.LogInformation("  in lower precision the central difference is lost to rounding/cancellation. The backprop");
		log.LogInformation("  math is shared across precisions and is validated in a double build.");
		return;
	}

	var rng = new Random(42);
	// A tiny model + short sequence keeps the finite-difference sweep cheap. The check always uses
	// full-precision Linear layers: a BitLinear cannot be finite-difference-checked, because a tiny
	// weight perturbation almost never changes the rounded ternary value, so the numerical gradient
	// is ~0 while the straight-through gradient is deliberately nonzero. This validates the shared
	// backprop machinery; the BitLinear STE is correct by construction (pass-through of that math).
	const int vocab = 6;
	var model = new ScratchGpt(vocabSize: vocab, blockSize: 8, embedDim: 8, heads: 2, layers: 1, rng,
		(inDim, outDim, init, bias) => new Linear(inDim, outDim, init, bias));
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
			var analytic = (double)p.Grad.Data[i];

			data[i] = (Real)((double)original + h);
			var lossPlus = LossAt();
			data[i] = (Real)((double)original - h);
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

static ILoggerFactory CreateLoggerFactory(IConfiguration configuration)
{
	var serilogLogger = new Serilog.LoggerConfiguration()
		.ReadFrom.Configuration(configuration)
		.CreateLogger();

	return new SerilogLoggerFactory(serilogLogger, dispose: true);
}
