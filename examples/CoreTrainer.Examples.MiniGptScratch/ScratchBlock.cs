namespace CoreTrainer.Examples.MiniGptScratch;

/// <summary>
/// One GPT-2-style transformer block, hand-rolled with an explicit backward pass: pre-LayerNorm
/// causal self-attention and a pre-LayerNorm MLP, each wrapped in a residual connection.
/// </summary>
/// <remarks>
/// Forward: <c>x = x + Attention(LN1(x));  x = x + MLP(LN2(x))</c>. The residual additions are the
/// key to the backward pass — since the output is a sum, the upstream gradient flows to <i>both</i>
/// the identity path and the sub-layer path, so each backward step adds the sub-layer's input
/// gradient back onto the passed-through gradient.
/// </remarks>
public sealed class ScratchBlock
{
	#region Fields

	private readonly ScratchSelfAttention _attention;
	private readonly ILinearLayer _fc1;   // MLP expand: embedDim -> 4·embedDim
	private readonly ILinearLayer _fc2;   // MLP contract: 4·embedDim -> embedDim

	// LayerNorm parameters (scale γ, shift β) for the two pre-norms.
	private readonly Parameter _ln1Gamma, _ln1Beta;
	private readonly Parameter _ln2Gamma, _ln2Beta;

	// Saved forward activations for backward.
	private Matrix? _inputToLn1, _inputToLn2, _mlpHidden, _mlpPreGelu;
	private double[]? _ln1Mean, _ln1InvStd, _ln2Mean, _ln2InvStd;

	#endregion

	#region Constructors

	public ScratchBlock(int embedDim, int heads, Random init, LinearFactory linear)
	{
		_attention = new ScratchSelfAttention(embedDim, heads, init, linear);
		_fc1 = linear(embedDim, 4 * embedDim, init, true);
		_fc2 = linear(4 * embedDim, embedDim, init, true);

		_ln1Gamma = OnesRow(embedDim);
		_ln1Beta = new Parameter(new Matrix(1, embedDim));
		_ln2Gamma = OnesRow(embedDim);
		_ln2Beta = new Parameter(new Matrix(1, embedDim));
	}

	#endregion

	#region Methods

	/// <summary>Collects every learnable parameter in the block.</summary>
	public IEnumerable<Parameter> Parameters()
	{
		foreach (var p in _attention.Parameters())
			yield return p;
		foreach (var p in _fc1.Parameters())
			yield return p;
		foreach (var p in _fc2.Parameters())
			yield return p;
		yield return _ln1Gamma;
		yield return _ln1Beta;
		yield return _ln2Gamma;
		yield return _ln2Beta;
	}

	/// <summary>Forward pass over one sequence (shape <c>[time, embedDim]</c>).</summary>
	public Matrix Forward(Matrix x)
	{
		// Residual around pre-LN attention.
		_inputToLn1 = x;
		var (ln1, ln1Mean, ln1InvStd) = Ops.LayerNorm(x, _ln1Gamma.Value, _ln1Beta.Value);
		_ln1Mean = ln1Mean;
		_ln1InvStd = ln1InvStd;
		var attended = _attention.Forward(ln1);
		var afterAttn = x.Add(attended);

		// Residual around pre-LN MLP.
		_inputToLn2 = afterAttn;
		var (ln2, ln2Mean, ln2InvStd) = Ops.LayerNorm(afterAttn, _ln2Gamma.Value, _ln2Beta.Value);
		_ln2Mean = ln2Mean;
		_ln2InvStd = ln2InvStd;

		_mlpPreGelu = _fc1.Forward(ln2);            // [time, 4·embedDim]
		_mlpHidden = Ops.Gelu(_mlpPreGelu);
		var mlpOut = _fc2.Forward(_mlpHidden);      // [time, embedDim]

		return afterAttn.Add(mlpOut);
	}

	/// <summary>
	/// Backward pass: returns the gradient w.r.t. the block input and accumulates all internal
	/// parameter gradients.
	/// </summary>
	public Matrix Backward(Matrix dOut)
	{
		if (_inputToLn1 is null || _inputToLn2 is null || _mlpHidden is null || _mlpPreGelu is null
			|| _ln1Mean is null || _ln1InvStd is null || _ln2Mean is null || _ln2InvStd is null)
			throw new InvalidOperationException("Backward called before Forward.");

		// --- MLP residual: out = afterAttn + MLP(LN2(afterAttn)) ---
		// dOut flows to both the identity branch (afterAttn) and the MLP branch.
		var dMlpOut = dOut;
		var dMlpHidden = _fc2.Backward(dMlpOut);                        // into GELU output
		var dPreGelu = Ops.GeluBackward(dMlpHidden, _mlpPreGelu);       // through GELU
		var dLn2 = _fc1.Backward(dPreGelu);                            // into LN2 output
		var dAfterAttnFromMlp = Ops.LayerNormBackward(
			dLn2, _inputToLn2, _ln2Gamma.Value, _ln2Mean, _ln2InvStd, _ln2Gamma.Grad, _ln2Beta.Grad);
		// Identity branch + MLP branch.
		var dAfterAttn = dOut.Add(dAfterAttnFromMlp);

		// --- Attention residual: afterAttn = x + Attention(LN1(x)) ---
		var dAttended = dAfterAttn;
		var dLn1 = _attention.Backward(dAttended);                     // into LN1 output
		var dxFromAttn = Ops.LayerNormBackward(
			dLn1, _inputToLn1, _ln1Gamma.Value, _ln1Mean, _ln1InvStd, _ln1Gamma.Grad, _ln1Beta.Grad);
		// Identity branch (dAfterAttn) + attention branch.
		return dAfterAttn.Add(dxFromAttn);
	}

	/// <summary>A [1, dim] parameter initialized to ones — the identity scale for LayerNorm.</summary>
	private static Parameter OnesRow(int dim)
	{
		var m = new Matrix(1, dim);
		m.Fill(1.0);
		return new Parameter(m);
	}

	#endregion
}
