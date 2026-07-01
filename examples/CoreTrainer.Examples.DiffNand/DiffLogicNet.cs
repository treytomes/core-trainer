using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace CoreTrainer.Examples.DiffNand;

/// <summary>
/// A differentiable network of soft-NAND gates that hardens into a concrete NAND circuit.
/// </summary>
/// <remarks>
/// <para>
/// Every gate computes <c>1 - a·b</c> — which on the corners {0,1} is exactly NAND, and off the
/// corners is a smooth, differentiable relaxation. So the gate <i>function</i> is never
/// approximated; it is real NAND the instant its inputs are crisp. The only learnable degree of
/// freedom is therefore <b>connectivity</b>: which two earlier outputs each gate reads. Each of a
/// gate's two inputs holds a softmax distribution over <i>all</i> prior outputs (the network inputs
/// plus every earlier gate — a cumulative pool that gives automatic skip connections, so a signal
/// need not be re-derived at each layer and depth is spent on logic).
/// </para>
/// <para>
/// Hardening uses a straight-through estimator (STE): the forward pass uses hard one-hot wire picks
/// and hard {0,1} NAND, so the network's output <i>is</i> the discrete circuit it will be scored on;
/// the backward pass flows through the smooth softmax and <c>1 - a·b</c>. Annealing the softmax
/// temperature toward zero sharpens the soft distributions so the gradient and the hard pick agree.
/// Because the forward pass is already discrete, the training accuracy over the full input set equals
/// the extracted netlist's exhaustive accuracy — the rounding gap that plagues soft-only logic nets
/// is closed by construction.
/// </para>
/// </remarks>
public sealed class DiffLogicNet
{
	#region Fields

	private readonly int _inputCount;
	private readonly int _outputCount;
	private readonly int[] _layerWidths;
	// Per layer: two weight matrices of shape (poolSize, layerWidth). Column g of each holds the
	// pre-softmax logits selecting that gate's first / second input from the prior pool.
	private readonly List<(Parameter In1, Parameter In2)> _layers = new();
	private readonly List<int> _poolSizes = new(); // pool size feeding each layer

	#endregion

	#region Constructors

	/// <summary>
	/// Builds the network's parameters. The final layer is forced to <paramref name="outputCount"/>
	/// gates whose outputs are the network outputs; <paramref name="hiddenWidths"/> sets the gate
	/// count of each preceding layer (and thus the circuit depth available for the carry chain).
	/// </summary>
	public DiffLogicNet(int inputCount, int outputCount, int[] hiddenWidths, Generator generator)
	{
		_inputCount = inputCount;
		_outputCount = outputCount;
		_layerWidths = new int[hiddenWidths.Length + 1];
		Array.Copy(hiddenWidths, _layerWidths, hiddenWidths.Length);
		_layerWidths[^1] = outputCount;

		// The pool feeding layer 0 is just the inputs; each layer appends its outputs to the pool,
		// so later layers can wire to the inputs or to any earlier gate (cumulative skip pool).
		var pool = inputCount;
		foreach (var width in _layerWidths)
		{
			_poolSizes.Add(pool);
			// Small random logits so the initial softmax is near-uniform (no wire pre-favored).
			var in1 = new Parameter(randn(new long[] { pool, width }, generator: generator) * 0.1);
			var in2 = new Parameter(randn(new long[] { pool, width }, generator: generator) * 0.1);
			_layers.Add((in1, in2));
			pool += width;
		}
	}

	#endregion

	#region Properties

	/// <summary>All learnable parameters, for handing to an optimizer.</summary>
	public IEnumerable<Parameter> Parameters
	{
		get
		{
			foreach (var (in1, in2) in _layers)
			{
				yield return in1;
				yield return in2;
			}
		}
	}

	#endregion

	#region Methods

	/// <summary>
	/// Evaluates the network on a batch of inputs (shape <c>[batch, inputCount]</c>, values in
	/// [0,1]) and returns the output bits (shape <c>[batch, outputCount]</c>).
	/// </summary>
	/// <param name="x">Input batch.</param>
	/// <param name="temperature">Softmax temperature; lower sharpens wire selection toward one-hot.</param>
	/// <param name="hard">
	/// When true, the forward pass uses hard one-hot wires and hard {0,1} NAND via straight-through
	/// (the output is the exact discrete circuit). When false, a fully soft pass for warmup.
	/// </param>
	public Tensor Forward(Tensor x, double temperature, bool hard)
	{
		var pool = x;
		Tensor output = x; // overwritten by the last layer

		for (var layer = 0; layer < _layers.Count; layer++)
		{
			var (w1, w2) = _layers[layer];
			var in1 = Select(pool, w1, temperature, hard);
			var in2 = Select(pool, w2, temperature, hard);
			output = Nand(in1, in2, hard);

			// Append this layer's outputs to the pool for subsequent layers.
			pool = layer < _layers.Count - 1
				? cat(new[] { pool, output }, dim: 1)
				: pool; // no need to grow the pool after the final layer
		}

		return output;
	}

	/// <summary>
	/// Gathers each gate's selected input from the pool. Soft path: a softmax-weighted blend over all
	/// pool features. Hard path (STE): the argmax wire as a one-hot, with the soft blend's gradient.
	/// </summary>
	private static Tensor Select(Tensor pool, Parameter logits, double temperature, bool hard)
	{
		var soft = (logits / temperature).softmax(0); // (poolSize, width), columns sum to 1
		var blend = pool.matmul(soft);                // (batch, width)
		if (!hard)
			return blend;

		// Hard one-hot selection: argmax down the pool dimension, one row per gate.
		var idx = soft.argmax(0);                                         // (width,)
		var oneHot = nn.functional.one_hot(idx, soft.shape[0]).to_type(soft.dtype); // (width, poolSize)
		var hardSel = oneHot.transpose(0, 1);                             // (poolSize, width)
		var hardBlend = pool.matmul(hardSel);                             // (batch, width)
		// Straight-through: value is the hard gather, gradient flows through the soft blend.
		return blend + (hardBlend - blend).detach();
	}

	/// <summary>
	/// The soft-NAND gate <c>1 - a·b</c>. Hard path (STE): crisp {0,1} output in the forward pass,
	/// smooth gradient in the backward pass.
	/// </summary>
	private static Tensor Nand(Tensor a, Tensor b, bool hard)
	{
		var soft = 1.0 - a * b;
		if (!hard)
			return soft;

		var hardOut = (soft > 0.5).to_type(soft.dtype);
		return soft + (hardOut - soft).detach();
	}

	/// <summary>
	/// Extracts the hardened circuit as, for each gate, the two pool indices its inputs select
	/// (argmax of the current logits). Layer <c>L</c> gate <c>g</c> reads from the pool of inputs
	/// plus all earlier layers' outputs; index ordering matches <see cref="Forward"/>'s pool growth.
	/// </summary>
	public IReadOnlyList<(int Input1, int Input2)[]> ExtractNetlist()
	{
		var netlist = new List<(int, int)[]>();
		foreach (var (w1, w2) in _layers)
		{
			var idx1 = w1.argmax(0).data<long>().ToArray();
			var idx2 = w2.argmax(0).data<long>().ToArray();
			var gates = new (int, int)[idx1.Length];
			for (var g = 0; g < idx1.Length; g++)
				gates[g] = ((int)idx1[g], (int)idx2[g]);
			netlist.Add(gates);
		}
		return netlist;
	}

	#endregion
}
