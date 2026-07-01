namespace CoreTrainer.Examples.MiniGptScratch;

/// <summary>
/// A learnable parameter: its current value and the gradient accumulated for it during the backward
/// pass. The Adam optimizer keeps its own moment estimates in double precision (see
/// <see cref="AdamOptimizer"/>) rather than here, so optimizer state never inherits the model's
/// numeric grain — storing the tiny second moment at, say, fixed-point grain rounds it to zero and
/// explodes the update.
/// </summary>
public sealed class Parameter
{
	#region Constructors

	/// <summary>Creates a parameter wrapping <paramref name="value"/>, with a zeroed gradient.</summary>
	public Parameter(Matrix value)
	{
		Value = value;
		Grad = new Matrix(value.Rows, value.Cols);
	}

	#endregion

	#region Properties

	/// <summary>The parameter's current value.</summary>
	public Matrix Value { get; }

	/// <summary>Gradient of the loss w.r.t. this parameter, accumulated over the backward pass.</summary>
	public Matrix Grad { get; }

	/// <summary>
	/// Increments whenever <see cref="Value"/> is updated (once per optimizer step). Lets consumers
	/// that derive expensive artifacts from the weights — e.g. a <see cref="BitLinear"/> caching its
	/// ternarized/transposed weights across the several forward/backward passes of one training step —
	/// detect cheaply whether the weights have changed and their cache is stale.
	/// </summary>
	public long Version { get; private set; }

	#endregion

	#region Methods

	/// <summary>Resets the accumulated gradient to zero (called before each backward pass).</summary>
	public void ZeroGrad() => Grad.Zero();

	/// <summary>Signals that <see cref="Value"/> has changed, invalidating anything derived from it.</summary>
	public void MarkUpdated() => Version++;

	#endregion
}
