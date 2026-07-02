using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace CoreTrainer.Examples.LogicMlp;

/// <summary>
/// A differentiable logic gate network (after Petersen et al., "Deep Differentiable Logic Gate
/// Networks"), the <b>dual</b> of <c>CoreTrainer.Examples.DiffNand</c>: where DiffNand fixes the gate
/// (always NAND) and <i>learns the wiring</i>, this fixes the wiring (random, frozen at construction)
/// and <i>learns each node's gate function</i> — which of the 16 two-input Boolean gates it computes.
/// </summary>
/// <remarks>
/// <para>
/// <b>Bipolar, not [0,1].</b> Signals are bipolar (<c>-1</c>=false, <c>+1</c>=true). This is the
/// load-bearing choice: over {-1,+1} every gate has an exact <i>multilinear</i> polynomial
/// <c>f(a,b) = α + β·a + γ·b + δ·(a·b)</c> that is smooth on the cube and exact at the corners, and —
/// unlike [0,1] logic — a signal can carry a <i>sign</i>, so a node can argue <i>against</i> a feature,
/// not merely accumulate. All 16 gates share the same basis <c>{1, a, b, ab}</c> and differ only in
/// their coefficient vector, so evaluating all 16 is one small matmul.
/// </para>
/// <para>
/// <b>How a gradient selects a gate.</b> Each node holds 16 learnable logits. Its soft output is the
/// softmax-weighted blend of all 16 gate polynomials; the gradient flows into the logits (a node's
/// XOR-logit rises whenever "behave more like XOR here" lowers the loss). The learnable parameter is
/// not <i>inside</i> a gate — it is the mixture <i>over</i> gates. This is the dual of a perceptron:
/// a perceptron learns continuous weights on a fixed sum; a node here learns a categorical choice over
/// fixed operations, and its output is a bit.
/// </para>
/// <para>
/// <b>Hardening (STE).</b> The forward pass commits to each node's argmax gate, so the network output
/// <i>is</i> the discrete circuit it will be scored on; the backward pass flows through the soft blend
/// (straight-through estimator). Annealing the softmax temperature toward zero sharpens the blend so
/// the gradient and the hard pick agree, closing the soft-vs-hard rounding gap by construction — the
/// same discipline DiffNand uses for wiring and BitNet uses for ternary weights, here applied to gate
/// identity.
/// </para>
/// </remarks>
public sealed class LogicGateNet : ILogicNet
{
	#region Fields

	private readonly int[] _layerWidths;

	// Per layer: the 16 pre-softmax gate logits for every gate, shape (width, 16). Column k selects how
	// much this gate behaves like the k-th two-input Boolean function.
	private readonly List<Parameter> _gateLogits = new();
	// Per layer: the two fixed wire indices per gate into the PREVIOUS pool (frozen at construction).
	private readonly List<Tensor> _wireA = new();
	private readonly List<Tensor> _wireB = new();

	// Constant [4, 16] coefficient matrix: column k is the multilinear coefficients (α, β, γ, δ) of the
	// k-th gate over the basis {1, a, b, ab}. Built once; shared by every layer.
	private readonly Tensor _coeff;

	#endregion

	#region Constructors

	/// <summary>
	/// Builds the network. The final layer is forced to <paramref name="outputCount"/> gates whose
	/// outputs are the network outputs; <paramref name="hiddenWidths"/> sets the gate count of each
	/// preceding layer. Wiring is random and frozen; only the per-gate function logits are learnable.
	/// </summary>
	public LogicGateNet(int inputCount, int outputCount, int[] hiddenWidths, Generator generator)
	{
		_layerWidths = new int[hiddenWidths.Length + 1];
		Array.Copy(hiddenWidths, _layerWidths, hiddenWidths.Length);
		_layerWidths[^1] = outputCount;

		_coeff = BipolarGateAlgebra.Coefficients();

		// Each layer reads only the previous layer's outputs (the network inputs for layer 0). Fixed
		// random wiring picks two source signals per gate; a "passthrough" gate (function A or B) lets a
		// signal survive across layers, which is how a frozen topology still routes information forward.
		var poolSize = inputCount;
		foreach (var width in _layerWidths)
		{
			// Passthrough-biased init. A uniform gate mixture is a DEAD initialization: at any input
			// corner exactly 8 of the 16 gates output +1 and 8 output -1, so the near-uniform blend
			// averages to 0 for every input — signal dies after one layer and the gradient vanishes
			// through the depth (confirmed empirically: loss frozen at 0.25). Biasing every gate toward
			// the "A" passthrough (index 10, f = a) instead starts the network as a stack of wires that
			// faithfully propagate signal — the logic-gate analogue of identity-initialized ResNets —
			// so gradients flow and gates differentiate away from passthrough into real logic during
			// training. Small random noise breaks symmetry so gates don't move in lockstep.
			var logits = randn(new long[] { width, 16 }, generator: generator) * 0.1;
			logits[TensorIndex.Ellipsis, BipolarGateAlgebra.PassthroughAGate] += BipolarGateAlgebra.PassthroughInitBias;
			_gateLogits.Add(new Parameter(logits));
			_wireA.Add(randint(poolSize, new long[] { width }, generator: generator).to_type(ScalarType.Int64));
			_wireB.Add(randint(poolSize, new long[] { width }, generator: generator).to_type(ScalarType.Int64));
			poolSize = width; // next layer reads from this layer's outputs
		}
	}

	#endregion

	#region Properties

	/// <summary>All learnable parameters (the per-gate function logits), for handing to an optimizer.</summary>
	public IEnumerable<Parameter> Parameters => _gateLogits;

	#endregion

	#region Methods

	/// <summary>
	/// Evaluates the network on a batch of inputs (shape <c>[batch, inputCount]</c>, values in {0,1})
	/// and returns the output bits (shape <c>[batch, outputCount]</c>, values in {0,1}).
	/// </summary>
	/// <param name="x01">Input batch in {0,1}.</param>
	/// <param name="temperature">Softmax temperature; lower sharpens the gate blend toward one gate.</param>
	/// <param name="hard">
	/// When true, the forward pass commits to each node's argmax gate via straight-through (the output is
	/// the exact discrete circuit). When false, a fully soft blend for warmup.
	/// </param>
	public Tensor Forward(Tensor x01, double temperature, bool hard)
	{
		// Convert {0,1} inputs to bipolar {-1,+1}; all internal signals are bipolar.
		var pool = x01 * 2.0 - 1.0;

		for (var layer = 0; layer < _layerWidths.Length; layer++)
			pool = ForwardLayer(pool, layer, temperature, hard);

		// Convert the final bipolar outputs back to {0,1} to match the {0,1} targets/metrics.
		return (pool + 1.0) * 0.5;
	}

	/// <summary>Runs one layer: gather each gate's two wired inputs, blend the 16 gate functions, harden.</summary>
	private Tensor ForwardLayer(Tensor pool, int layer, double temperature, bool hard)
	{
		var a = pool.index_select(1, _wireA[layer]); // [batch, width]
		var b = pool.index_select(1, _wireB[layer]); // [batch, width]

		// Basis {1, a, b, ab} stacked on a new trailing axis, then one matmul against the [4,16]
		// coefficient matrix gives all 16 gate outputs at once: [batch, width, 16].
		var basis = stack(new[] { ones_like(a), a, b, a * b }, dim: 2); // [batch, width, 4]
		var gateOut = basis.matmul(_coeff);                             // [batch, width, 16]

		// Softmax over the 16 gates per node → mixture weights, then blend.
		var p = (_gateLogits[layer] / temperature).softmax(1);          // [width, 16]
		var soft = (gateOut * p.unsqueeze(0)).sum(new long[] { 2 });    // [batch, width]

		if (!hard)
			return soft;

		// Straight-through: pick each node's argmax gate for the value, keep the soft blend's gradient.
		var idx = p.argmax(1);                                          // [width]
		var gather = idx.reshape(1, -1, 1).expand(gateOut.shape[0], -1, 1); // [batch, width, 1]
		var hardOut = gateOut.gather(2, gather).squeeze(2);             // [batch, width]
		return soft + (hardOut - soft).detach();
	}

	/// <summary>
	/// Extracts the hardened circuit: for each layer, every gate's chosen function (argmax of its logits)
	/// and its two frozen wire indices into the previous pool. Layer 0 wires read the network inputs;
	/// layer L's wires read layer L−1's gate outputs.
	/// </summary>
	public IReadOnlyList<ExtractedGate[]> ExtractNetlist()
	{
		var netlist = new List<ExtractedGate[]>();
		for (var layer = 0; layer < _layerWidths.Length; layer++)
		{
			using var noGrad = no_grad();
			var choice = _gateLogits[layer].argmax(1).data<long>().ToArray();
			var wa = _wireA[layer].data<long>().ToArray();
			var wb = _wireB[layer].data<long>().ToArray();

			var gates = new ExtractedGate[choice.Length];
			for (var g = 0; g < choice.Length; g++)
				gates[g] = new ExtractedGate((int)choice[g], (int)wa[g], (int)wb[g]);
			netlist.Add(gates);
		}
		return netlist;
	}

	#endregion
}

/// <summary>One extracted gate: the chosen Boolean function (0..15) and its two input wire indices.</summary>
public readonly record struct ExtractedGate(int Function, int Input1, int Input2)
{
	/// <summary>Human-readable names of the 16 two-input Boolean functions, indexed by truth-table value.</summary>
	public static readonly string[] FunctionNames =
	{
		"FALSE", "NOR", "A>B", "NOTB", "B>A", "NOTA", "XOR", "NAND",
		"AND", "XNOR", "A", "B->A", "B", "A->B", "OR", "TRUE",
	};

	/// <summary>The chosen function's name.</summary>
	public string Name => FunctionNames[Function];

	/// <summary>Whether this gate is dead structure: a constant (TRUE/FALSE) — prunable from the circuit.</summary>
	public bool IsConstant => Function is 0 or 15;

	/// <summary>Whether this gate merely passes a signal through (A or B) — a wire, not logic.</summary>
	public bool IsPassthrough => Function is 10 or 12;
}
