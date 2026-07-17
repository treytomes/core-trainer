using Microsoft.Extensions.Logging;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace CoreTrainer.Examples.MiniGpt;

/// <summary>
/// A drop-in replacement for the transformer block's MLP whose nonlinearity is a <b>differentiable
/// logic-gate network</b> instead of GELU. The continuous residual stream is projected down, quantized
/// to bipolar bits, transformed by a network that learns both its wiring and each node's Boolean gate,
/// then projected back up — so the block's per-token feature transform becomes (cullable) logic.
/// </summary>
/// <remarks>
/// <para>
/// Shape: <c>encode: embedDim → hidden</c> (Linear), <b>bipolarize</b> to <c>[-1,+1]</c>, a stack of
/// logic layers over a cumulative skip pool, then <c>decode: hidden → embedDim</c> (Linear). The two
/// Linears are the continuous↔bipolar boundary — exactly BitNet's activation-quantization idea, with a
/// logic network where GELU used to be. This is the fusion of CoreTrainer.Examples.LogicMlp
/// (learns gate function) and DiffNand (learns wiring), moved inside a transformer.
/// </para>
/// <para>
/// <b>Soft vs hard.</b> In <c>hard</c> mode the bipolarizer is <c>sign</c> and the wire/gate selections
/// are argmax, both via straight-through — the block computes a discrete logic circuit. In <c>soft</c>
/// mode the bipolarizer is <c>tanh</c> and selections are softmax blends — fully differentiable, no
/// commitment. Unlike a standalone circuit (which must hit 100% to be a valid adder), an MLP is only a
/// feature transform smoothed by LayerNorm, the residual add, and the next attention layer, so it does
/// not need to be an exact circuit — just a useful nonlinearity. That makes soft mode the natural fit;
/// hard mode tests whether end-to-end training tolerates the STE commitment that hurt standalone circuits.
/// </para>
/// </remarks>
public sealed class LogicFeedForward : nn.Module<Tensor, Tensor>
{
	#region Fields

	private readonly bool _hard;
	private double _temperature;
	private readonly int[] _layerWidths;

	private readonly nn.Module<Tensor, Tensor> _encode;   // embedDim -> hidden (to logic inputs)
	private readonly nn.Module<Tensor, Tensor> _decode;   // final logic layer width -> embedDim

	// Per logic layer: two wire-selection logit matrices (poolSize, width) and gate-function logits
	// (width, 16). Registered as Parameters so they live in the module's parameter tree and train.
	private readonly List<Parameter> _wire1 = new();
	private readonly List<Parameter> _wire2 = new();
	private readonly List<Parameter> _gate = new();
	private readonly int[] _poolSizes;

	private readonly Tensor _coeff; // [4,16] constant multilinear gate coefficients (registered buffer)

	#endregion

	#region Constructors

	/// <param name="embedDim">Model dimension (block input/output width).</param>
	/// <param name="hidden">Logic input width (first pool size); the "expand" dimension.</param>
	/// <param name="logicWidths">Gate count per logic layer; the last entry is the decode input width.</param>
	/// <param name="hard">Hard (sign + argmax, STE) vs soft (tanh + softmax blend) logic.</param>
	/// <param name="temperature">Softmax/anneal temperature for wire and gate selection.</param>
	/// <param name="useBitLinear">Use BitNet ternary-weight linears for the encode/decode boundary.</param>
	public LogicFeedForward(int embedDim, int hidden, int[] logicWidths, bool hard, double temperature,
		bool useBitLinear = false)
		: base(nameof(LogicFeedForward))
	{
		_hard = hard;
		_temperature = temperature;
		_layerWidths = logicWidths;

		_encode = Linears.Make(embedDim, hidden, useBitLinear);
		_decode = Linears.Make(logicWidths[^1], embedDim, useBitLinear);

		_coeff = BipolarGateAlgebra.Coefficients();

		// Cumulative skip pool: layer 0 reads the `hidden` encoded features; each layer appends its
		// outputs, so later layers can wire to any earlier signal (gives signals a live forward path).
		_poolSizes = new int[logicWidths.Length];
		var pool = hidden;
		for (var layer = 0; layer < logicWidths.Length; layer++)
		{
			_poolSizes[layer] = pool;
			var width = logicWidths[layer];

			_wire1.Add(new Parameter(randn(pool, width) * 0.1));
			_wire2.Add(new Parameter(randn(pool, width) * 0.1));

			// Passthrough-biased gate init (a uniform 16-gate mixture outputs 0 for all inputs — dead;
			// see BipolarGateAlgebra), so the block starts near identity and gradients flow.
			var g = randn(width, 16) * 0.1;
			g[TensorIndex.Ellipsis, BipolarGateAlgebra.PassthroughAGate] += BipolarGateAlgebra.PassthroughInitBias;
			_gate.Add(new Parameter(g));

			pool += width;
		}

		RegisterComponents();
	}

	#endregion

	#region Methods

	/// <summary>Sets the softmax/anneal temperature for wire and gate selection (called per epoch).</summary>
	public void SetTemperature(double temperature) => _temperature = temperature;

	// The 16 two-input Boolean gates by truth-table index. Constants (FALSE/TRUE) are dead structure;
	// passthroughs (A/B) are pure wiring, not logic. Everything else is real logic.
	private static readonly string[] GateNames =
	{
		"FALSE", "NOR", "A>B", "NOTB", "B>A", "NOTA", "XOR", "NAND",
		"AND", "XNOR", "A", "B->A", "B", "A->B", "OR", "TRUE",
	};

	/// <summary>
	/// Reports how much of the trained logic net is real logic versus prunable structure — the
	/// cullability metric. For each gate, its committed function is the argmax of its (trained) gate
	/// logits; a gate is a <b>constant</b> (FALSE/TRUE — dead, prunable) or a <b>passthrough</b> (A/B —
	/// pure wiring) or real logic. This earns-or-refutes the "the MLP became cullable logic" claim.
	/// </summary>
	/// <remarks>
	/// Meaningful for hard mode, where each node truly commits to its argmax gate. In soft mode the
	/// argmax is only the dominant term of a 16-gate blend (the node is not that single gate), so treat
	/// the soft numbers as indicative, not a literal circuit — hence the caller labels the mode.
	/// </remarks>
	public void ReportCullability(ILogger log, string label)
	{
		using var noGrad = no_grad();
		var counts = new int[16];
		int total = 0, constants = 0, passthroughs = 0;

		for (var layer = 0; layer < _layerWidths.Length; layer++)
		{
			var choice = _gate[layer].argmax(1).data<long>().ToArray();
			foreach (var f in choice)
			{
				counts[f]++;
				total++;
				if (f is 0 or 15) constants++;        // FALSE / TRUE
				else if (f is 10 or 12) passthroughs++; // A / B
			}
		}

		var logic = total - constants - passthroughs;
		log.LogInformation("  [{Label}] logic-MLP cullability: {Total} gates — constants {Const} ({CPct:P0}), passthroughs {Pass} ({PPct:P0}), real logic {Logic} ({LPct:P0})",
			label, total, constants, (double)constants / total, passthroughs, (double)passthroughs / total, logic, (double)logic / total);

		var order = Enumerable.Range(0, 16).Where(k => counts[k] > 0).OrderByDescending(k => counts[k]);
		log.LogInformation("  [{Label}] gate mix: {Mix}", label, string.Join(", ", order.Select(k => $"{GateNames[k]}={counts[k]}")));
	}

	/// <summary>Runs the block over <c>[batch, time, embedDim]</c>, returning the same shape.</summary>
	public override Tensor forward(Tensor x)
	{
		using var scope = NewDisposeScope();

		var batch = x.shape[0];
		var time = x.shape[1];
		var embedDim = x.shape[2];

		// Flatten tokens to rows so the logic net sees a plain [rows, features] batch.
		var rows = x.reshape(batch * time, embedDim);
		var encoded = _encode.forward(rows);                  // [rows, hidden]

		// Continuous -> bipolar. Soft: tanh (smooth, differentiable). Hard: sign with STE (crisp bits,
		// gradient through tanh) so the logic net sees true {-1,+1} while staying trainable.
		var pool = _hard ? SignSte(encoded.tanh()) : encoded.tanh();

		for (var layer = 0; layer < _layerWidths.Length; layer++)
		{
			var a = SelectWire(pool, _wire1[layer]);
			var b = SelectWire(pool, _wire2[layer]);
			var output = ApplyGates(a, b, _gate[layer]);
			pool = layer < _layerWidths.Length - 1 ? cat(new[] { pool, output }, dim: 1) : output;
		}

		var decoded = _decode.forward(pool);                  // [rows, embedDim]
		return decoded.reshape(batch, time, embedDim).MoveToOuterDisposeScope();
	}

	// Wire selection over the pool. Soft: softmax-weighted blend. Hard: argmax one-hot via STE.
	private Tensor SelectWire(Tensor pool, Parameter logits)
	{
		var soft = (logits / _temperature).softmax(0);        // (poolSize, width)
		var blend = pool.matmul(soft);                        // (rows, width)
		if (!_hard)
			return blend;

		var idx = soft.argmax(0);
		var oneHot = nn.functional.one_hot(idx, soft.shape[0]).to_type(soft.dtype);
		var hardBlend = pool.matmul(oneHot.transpose(0, 1));
		return blend + (hardBlend - blend).detach();
	}

	// Gate application: blend all 16 bipolar multilinear gates (soft) or commit to argmax gate (hard STE).
	private Tensor ApplyGates(Tensor a, Tensor b, Parameter gateLogits)
	{
		var basis = stack(new[] { ones_like(a), a, b, a * b }, dim: 2); // [rows, width, 4]
		var gateOut = basis.matmul(_coeff);                             // [rows, width, 16]

		var p = (gateLogits / _temperature).softmax(1);                 // [width, 16]
		var soft = (gateOut * p.unsqueeze(0)).sum(new long[] { 2 });    // [rows, width]
		if (!_hard)
			return soft;

		var idx = p.argmax(1);
		var gather = idx.reshape(1, -1, 1).expand(gateOut.shape[0], -1, 1);
		var hardOut = gateOut.gather(2, gather).squeeze(2);
		return soft + (hardOut - soft).detach();
	}

	// sign(x) with a straight-through estimator: crisp {-1,+1} forward, identity-ish gradient (through
	// the tanh applied before this) so the bipolarizer is trainable.
	private static Tensor SignSte(Tensor x)
	{
		var hard = x.sign();
		// sign(0)=0 would drop a bit to 0; nudge exact zeros to +1 so outputs stay strictly bipolar.
		hard = where(hard.eq(0), ones_like(hard), hard);
		return x + (hard - x).detach();
	}

	#endregion
}
