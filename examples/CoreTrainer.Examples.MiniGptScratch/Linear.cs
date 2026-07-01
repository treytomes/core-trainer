namespace CoreTrainer.Examples.MiniGptScratch;

/// <summary>
/// A fully-connected (affine) layer: <c>y = x·W + b</c>, the transformer's fundamental learnable
/// transform, used for the attention Q/K/V/output projections, the MLP, and the final LM head.
/// </summary>
/// <remarks>
/// Forward saves its input so the backward pass can form the weight gradient. The backward math is
/// the canonical linear-layer rule:
/// <code>
///   dX = dY · Wᵀ          // gradient to the input (flows further back)
///   dW = Xᵀ · dY          // gradient to the weights (accumulated)
///   db = Σrows dY         // gradient to the bias   (accumulated)
/// </code>
/// Weights are stored <c>[inputDim, outputDim]</c> so <c>x·W</c> maps <c>[time, inputDim]</c> to
/// <c>[time, outputDim]</c>.
/// </remarks>
public sealed class Linear : ILinearLayer
{
	#region Fields

	private readonly bool _hasBias;
	private Matrix? _lastInput; // saved by forward for use in backward

	#endregion

	#region Constructors

	/// <param name="inputDim">Input feature dimension.</param>
	/// <param name="outputDim">Output feature dimension.</param>
	/// <param name="init">Random source for weight initialization.</param>
	/// <param name="hasBias">Whether to include a learnable bias.</param>
	public Linear(int inputDim, int outputDim, Random init, bool hasBias = true)
	{
		// Standard fan-in scaled init (Kaiming-style) keeps activation variance stable through depth.
		var scale = Math.Sqrt(2.0 / inputDim);
		var w = new Matrix(inputDim, outputDim);
		for (var i = 0; i < w.Data.Length; i++)
			w.Data[i] = NextGaussian(init) * scale;
		Weight = new Parameter(w);

		_hasBias = hasBias;
		if (hasBias)
			Bias = new Parameter(new Matrix(1, outputDim));
	}

	#endregion

	#region Properties

	/// <summary>The weight matrix parameter, shape <c>[inputDim, outputDim]</c>.</summary>
	public Parameter Weight { get; }

	/// <summary>The bias parameter, shape <c>[1, outputDim]</c>, or null when disabled.</summary>
	public Parameter? Bias { get; }

	#endregion

	#region Methods

	/// <summary>Collects this layer's learnable parameters.</summary>
	public IEnumerable<Parameter> Parameters()
	{
		yield return Weight;
		if (Bias is not null)
			yield return Bias;
	}

	/// <summary>Forward pass <c>y = x·W (+ b)</c>. Saves <paramref name="x"/> for the backward pass.</summary>
	public Matrix Forward(Matrix x)
	{
		_lastInput = x;
		var y = x.MatMul(Weight.Value);
		return _hasBias ? y.AddRowVector(Bias!.Value) : y;
	}

	/// <summary>
	/// Backward pass. Accumulates weight (and bias) gradients and returns the gradient w.r.t. the
	/// input. Requires a preceding <see cref="Forward"/> call.
	/// </summary>
	public Matrix Backward(Matrix dY)
	{
		if (_lastInput is null)
			throw new InvalidOperationException("Backward called before Forward.");

		// dW = Xᵀ · dY, accumulated across sequence positions (and, over a training step, sequences).
		var dW = _lastInput.Transpose().MatMul(dY);
		Weight.Grad.AddInPlace(dW);

		if (_hasBias)
			Bias!.Grad.AddInPlace(dY.SumRows()); // db = Σrows dY

		// dX = dY · Wᵀ flows to the previous layer.
		return dY.MatMul(Weight.Value.Transpose());
	}

	/// <summary>Box–Muller standard normal sample.</summary>
	private static double NextGaussian(Random rng)
	{
		var u1 = 1.0 - rng.NextDouble();
		var u2 = 1.0 - rng.NextDouble();
		return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
	}

	#endregion
}
