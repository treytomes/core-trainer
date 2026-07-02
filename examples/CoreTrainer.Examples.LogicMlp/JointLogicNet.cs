using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace CoreTrainer.Examples.LogicMlp;

/// <summary>
/// A logic-gate network that learns <b>both</b> the wiring <b>and</b> the gate function — the fusion of
/// <see cref="LogicGateNet"/> (learns gate, fixes wiring) and <c>CoreTrainer.Examples.DiffNand</c>
/// (learns wiring, fixes gate). Motivated by the LogicMlp ladder result: with fixed random wiring the
/// binding constraint was <i>routing</i> (a node could learn the right gate but not get the right
/// signals to meet at it), so this frees that degree of freedom too.
/// </summary>
/// <remarks>
/// <para>
/// Each gate has three learnable pieces: two softmax distributions selecting its inputs from a
/// <b>cumulative skip pool</b> (the network inputs plus every earlier gate's output — DiffNand's trick,
/// which gives every signal a live forward path and lets depth be spent on logic rather than
/// re-derivation), and a 16-way softmax choosing its Boolean function (LogicMlp's mechanism, over the
/// bipolar multilinear gates). All three are hardened by straight-through and sharpened by one shared
/// annealed temperature, so the forward pass is the exact discrete circuit while gradients flow through
/// the soft blends.
/// </para>
/// <para>
/// Signals are bipolar <c>{-1,+1}</c> throughout. Gate logits are passthrough-biased at init (see
/// <see cref="BipolarGateAlgebra"/>) so the network starts as identity wiring rather than the dead
/// zero-output state a uniform gate mixture produces.
/// </para>
/// </remarks>
public sealed class JointLogicNet : ILogicNet
{
	#region Fields

	private readonly int[] _layerWidths;
	private readonly List<int> _poolSizes = new();          // pool size feeding each layer

	// Per layer: input-selection logits (poolSize, width) for each of the two inputs, and gate-function
	// logits (width, 16). All learnable.
	private readonly List<Parameter> _wireLogits1 = new();
	private readonly List<Parameter> _wireLogits2 = new();
	private readonly List<Parameter> _gateLogits = new();

	private readonly Tensor _coeff; // [4,16] shared multilinear gate coefficients

	#endregion

	#region Constructors

	/// <summary>
	/// Builds the network. The final layer is forced to <paramref name="outputCount"/> gates whose
	/// outputs are the network outputs; <paramref name="hiddenWidths"/> sets each preceding layer's width.
	/// </summary>
	public JointLogicNet(int inputCount, int outputCount, int[] hiddenWidths, Generator generator)
	{
		_layerWidths = new int[hiddenWidths.Length + 1];
		Array.Copy(hiddenWidths, _layerWidths, hiddenWidths.Length);
		_layerWidths[^1] = outputCount;

		_coeff = BipolarGateAlgebra.Coefficients();

		// The pool feeding layer 0 is just the inputs; each layer appends its outputs to the pool, so
		// later layers can wire to the inputs or any earlier gate (cumulative skip pool).
		var pool = inputCount;
		foreach (var width in _layerWidths)
		{
			_poolSizes.Add(pool);

			// Small random wire logits → near-uniform initial selection (no wire pre-favored).
			_wireLogits1.Add(new Parameter(randn(new long[] { pool, width }, generator: generator) * 0.1));
			_wireLogits2.Add(new Parameter(randn(new long[] { pool, width }, generator: generator) * 0.1));

			// Passthrough-biased gate logits (see BipolarGateAlgebra): start as identity wiring so signal
			// propagates and the gradient does not vanish (a uniform gate mixture outputs 0 for all inputs).
			var gate = randn(new long[] { width, 16 }, generator: generator) * 0.1;
			gate[TensorIndex.Ellipsis, BipolarGateAlgebra.PassthroughAGate] += BipolarGateAlgebra.PassthroughInitBias;
			_gateLogits.Add(new Parameter(gate));

			pool += width;
		}
	}

	#endregion

	#region Properties

	/// <summary>All learnable parameters (both wire-selection logits and gate-function logits per layer).</summary>
	public IEnumerable<Parameter> Parameters
	{
		get
		{
			for (var layer = 0; layer < _layerWidths.Length; layer++)
			{
				yield return _wireLogits1[layer];
				yield return _wireLogits2[layer];
				yield return _gateLogits[layer];
			}
		}
	}

	#endregion

	#region Methods

	/// <summary>
	/// Evaluates the network on a batch (shape <c>[batch, inputCount]</c>, values in {0,1}) and returns
	/// the output bits (shape <c>[batch, outputCount]</c>, values in {0,1}).
	/// </summary>
	/// <param name="temperature">Shared softmax temperature for wire and gate selection; lower = sharper.</param>
	/// <param name="hard">When true, hard one-hot wires and argmax gates via STE (the exact circuit).</param>
	public Tensor Forward(Tensor x01, double temperature, bool hard)
	{
		var pool = x01 * 2.0 - 1.0; // to bipolar {-1,+1}

		for (var layer = 0; layer < _layerWidths.Length; layer++)
		{
			var a = SelectWire(pool, _wireLogits1[layer], temperature, hard); // [batch, width]
			var b = SelectWire(pool, _wireLogits2[layer], temperature, hard); // [batch, width]
			var output = ApplyGates(a, b, _gateLogits[layer], temperature, hard);

			// Grow the cumulative pool with this layer's outputs (except after the final layer).
			pool = layer < _layerWidths.Length - 1
				? cat(new[] { pool, output }, dim: 1)
				: output;
		}

		return (pool + 1.0) * 0.5; // back to {0,1}
	}

	/// <summary>
	/// Gathers each gate's selected input from the pool. Soft: a softmax-weighted blend over all pool
	/// features. Hard (STE): the argmax wire as a one-hot, carrying the soft blend's gradient.
	/// </summary>
	private static Tensor SelectWire(Tensor pool, Parameter logits, double temperature, bool hard)
	{
		var soft = (logits / temperature).softmax(0); // (poolSize, width), columns sum to 1
		var blend = pool.matmul(soft);                // (batch, width)
		if (!hard)
			return blend;

		var idx = soft.argmax(0);                                                    // (width,)
		var oneHot = nn.functional.one_hot(idx, soft.shape[0]).to_type(soft.dtype);  // (width, poolSize)
		var hardBlend = pool.matmul(oneHot.transpose(0, 1));                         // (batch, width)
		return blend + (hardBlend - blend).detach();
	}

	/// <summary>
	/// Blends the 16 gate functions per node (soft), or commits to each node's argmax gate (hard, STE).
	/// Basis {1, a, b, ab} × the [4,16] coefficient matrix gives all 16 gate outputs at once.
	/// </summary>
	private Tensor ApplyGates(Tensor a, Tensor b, Parameter gateLogits, double temperature, bool hard)
	{
		var basis = stack(new[] { ones_like(a), a, b, a * b }, dim: 2); // [batch, width, 4]
		var gateOut = basis.matmul(_coeff);                             // [batch, width, 16]

		var p = (gateLogits / temperature).softmax(1);                  // [width, 16]
		var soft = (gateOut * p.unsqueeze(0)).sum(new long[] { 2 });    // [batch, width]
		if (!hard)
			return soft;

		var idx = p.argmax(1);                                          // [width]
		var gather = idx.reshape(1, -1, 1).expand(gateOut.shape[0], -1, 1);
		var hardOut = gateOut.gather(2, gather).squeeze(2);             // [batch, width]
		return soft + (hardOut - soft).detach();
	}

	/// <summary>
	/// Extracts the hardened circuit: per layer, each gate's chosen function and its two selected pool
	/// indices (argmax of the current wire logits). Pool index ordering matches <see cref="Forward"/>'s
	/// growth: inputs first, then each earlier layer's outputs.
	/// </summary>
	public IReadOnlyList<ExtractedGate[]> ExtractNetlist()
	{
		var netlist = new List<ExtractedGate[]>();
		for (var layer = 0; layer < _layerWidths.Length; layer++)
		{
			using var noGrad = no_grad();
			var choice = _gateLogits[layer].argmax(1).data<long>().ToArray();
			var idx1 = _wireLogits1[layer].argmax(0).data<long>().ToArray();
			var idx2 = _wireLogits2[layer].argmax(0).data<long>().ToArray();

			var gates = new ExtractedGate[choice.Length];
			for (var g = 0; g < choice.Length; g++)
				gates[g] = new ExtractedGate((int)choice[g], (int)idx1[g], (int)idx2[g]);
			netlist.Add(gates);
		}
		return netlist;
	}

	#endregion
}
