namespace CoreTrainer.Examples.MiniGptScratch;

/// <summary>
/// A learnable parameter: its current value, the gradient accumulated for it during the backward
/// pass, and the Adam optimizer's per-element moment estimates. Bundling all four together is what
/// a tensor library's autograd + optimizer would manage invisibly; here it is explicit so the
/// update step can be read directly.
/// </summary>
public sealed class Parameter
{
	#region Constructors

	/// <summary>Creates a parameter wrapping <paramref name="value"/>, with zeroed grad and moments.</summary>
	public Parameter(Matrix value)
	{
		Value = value;
		Grad = new Matrix(value.Rows, value.Cols);
		M = new Matrix(value.Rows, value.Cols);
		V = new Matrix(value.Rows, value.Cols);
	}

	#endregion

	#region Properties

	/// <summary>The parameter's current value.</summary>
	public Matrix Value { get; }

	/// <summary>Gradient of the loss w.r.t. this parameter, accumulated over the backward pass.</summary>
	public Matrix Grad { get; }

	/// <summary>Adam first-moment (mean of gradients) estimate.</summary>
	public Matrix M { get; }

	/// <summary>Adam second-moment (mean of squared gradients) estimate.</summary>
	public Matrix V { get; }

	#endregion

	#region Methods

	/// <summary>Resets the accumulated gradient to zero (called before each backward pass).</summary>
	public void ZeroGrad() => Grad.Zero();

	#endregion
}
