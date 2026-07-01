namespace CoreTrainer.Examples.MiniGptScratch;

/// <summary>
/// Creates a linear layer of the requested shape. Lets the model be built with either full-precision
/// <see cref="Linear"/> or ternary <see cref="BitLinear"/> projections without changing its wiring —
/// the driver picks the implementation and threads this factory into the transformer blocks.
/// </summary>
/// <param name="inputDim">Input feature dimension.</param>
/// <param name="outputDim">Output feature dimension.</param>
/// <param name="init">Random source for weight initialization.</param>
/// <param name="hasBias">Whether the layer includes a learnable bias.</param>
public delegate ILinearLayer LinearFactory(int inputDim, int outputDim, Random init, bool hasBias);

/// <summary>
/// A learnable affine transform <c>y = f(x·W (+ b))</c> with forward and (hand-rolled) backward
/// passes. Implemented by both the full-precision <see cref="Linear"/> and the ternary-weight
/// <see cref="BitLinear"/>, so the transformer's blocks can hold either interchangeably and a
/// layer can be made ternary without changing the model wiring.
/// </summary>
public interface ILinearLayer
{
	/// <summary>The layer's learnable parameters, for the optimizer.</summary>
	IEnumerable<Parameter> Parameters();

	/// <summary>Forward pass mapping <c>[time, inputDim]</c> to <c>[time, outputDim]</c>. Saves what
	/// the backward pass needs.</summary>
	Matrix Forward(Matrix x);

	/// <summary>Backward pass: accumulates parameter gradients and returns the gradient w.r.t. the
	/// input. Requires a preceding <see cref="Forward"/>.</summary>
	Matrix Backward(Matrix dY);
}
