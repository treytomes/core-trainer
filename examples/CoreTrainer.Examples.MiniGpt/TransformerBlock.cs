using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace CoreTrainer.Examples.MiniGpt;

/// <summary>
/// One GPT-2-style transformer block: pre-layer-norm masked self-attention and a position-wise
/// MLP, each wrapped in a residual connection.
/// </summary>
/// <remarks>
/// Pre-LN (normalize before each sub-layer, add the residual after) is the arrangement GPT-2 uses;
/// it keeps gradients well-scaled through deep stacks and trains more stably than the original
/// post-LN transformer. The MLP expands to 4x the embedding dimension and back — the conventional
/// ratio — with a GELU nonlinearity in between.
/// </remarks>
public sealed class TransformerBlock : nn.Module<Tensor, Tensor>
{
	#region Fields

	private readonly LayerNorm _ln1;
	private readonly CausalSelfAttention _attention;
	private readonly LayerNorm _ln2;
	private readonly nn.Module<Tensor, Tensor> _mlp;

	#endregion

	#region Constructors

	/// <param name="mlpMode">
	/// Which feed-forward to use: the standard GELU MLP, or a differentiable logic-gate network (soft or
	/// hard). The logic variants expand to 4·embedDim (matching the MLP's hidden width) and use a stack
	/// of logic layers in place of the GELU nonlinearity.
	/// </param>
	/// <param name="logicTemperature">Softmax/anneal temperature for the logic variants (ignored for GELU).</param>
	public TransformerBlock(int embedDim, int heads, MlpMode mlpMode = MlpMode.Gelu, double logicTemperature = 1.0)
		: base(nameof(TransformerBlock))
	{
		_ln1 = nn.LayerNorm(embedDim);
		_attention = new CausalSelfAttention(embedDim, heads);
		_ln2 = nn.LayerNorm(embedDim);
		_mlp = BuildMlp(embedDim, mlpMode, logicTemperature);

		RegisterComponents();
	}

	// The feed-forward sub-layer. GELU is the conventional MLP; the logic modes swap the nonlinearity for
	// a logic-gate network, keeping the same 4·embedDim hidden width so parameter budgets are comparable.
	private static nn.Module<Tensor, Tensor> BuildMlp(int embedDim, MlpMode mode, double temperature)
	{
		var hidden = 4 * embedDim;
		if (mode == MlpMode.Gelu)
			return nn.Sequential(
				nn.Linear(embedDim, hidden),
				nn.GELU(),
				nn.Linear(hidden, embedDim));

		// A few logic layers over the expanded width. Depth gives the gate network room to compose;
		// width (= hidden) gives its learned wiring enough signals to route.
		var logicWidths = new[] { hidden, hidden, hidden };
		return new LogicFeedForward(embedDim, hidden, logicWidths, hard: mode == MlpMode.LogicHard, temperature);
	}

	#endregion

	#region Methods

	/// <summary>Forwards a logic-temperature update to the MLP if it is a logic feed-forward (else no-op).</summary>
	public void SetLogicTemperature(double temperature)
	{
		if (_mlp is LogicFeedForward logic)
			logic.SetTemperature(temperature);
	}

	/// <summary>
	/// Runs the block over a batch of sequences (shape <c>[batch, time, embedDim]</c>), returning the
	/// same shape.
	/// </summary>
	public override Tensor forward(Tensor x)
	{
		using var scope = NewDisposeScope();

		// Residual around pre-LN attention, then residual around pre-LN MLP.
		x = x + _attention.forward(_ln1.forward(x));
		x = x + _mlp.forward(_ln2.forward(x));
		return x.MoveToOuterDisposeScope();
	}

	#endregion
}
