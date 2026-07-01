namespace CoreTrainer.Examples.MiniGptScratch;

/// <summary>
/// The stateless numeric operations of the transformer, each paired with an explicit backward
/// (gradient) pass. These are exactly the pieces a tensor library computes automatically via
/// autograd; spelling them out is the whole point of this example.
/// </summary>
/// <remarks>
/// Every function operates on <see cref="Matrix"/> shapes of <c>[time, features]</c> (one sequence,
/// no batch dimension — sequences are processed one at a time and their gradients summed). Backward
/// functions take the upstream gradient (d-loss w.r.t. the op's output) plus whatever forward
/// activations the derivative needs, and return the downstream gradient (d-loss w.r.t. the input).
/// </remarks>
public static class Ops
{
	private const double Epsilon = 1e-5;
	private static readonly double GeluCoeff = Math.Sqrt(2.0 / Math.PI);

	#region Softmax

	/// <summary>
	/// Row-wise softmax: each row becomes a probability distribution. Subtracts the row max first for
	/// numerical stability (does not change the result).
	/// </summary>
	public static Matrix SoftmaxRows(Matrix x)
	{
		var result = new Matrix(x.Rows, x.Cols);
		for (var r = 0; r < x.Rows; r++)
		{
			var max = RealMath.NegativeInfinity;
			for (var c = 0; c < x.Cols; c++)
				max = RealMath.Max(max, x[r, c]);

			Real sum = 0;
			for (var c = 0; c < x.Cols; c++)
			{
				var e = RealMath.Exp(x[r, c] - max);
				result[r, c] = e;
				sum += e;
			}

			for (var c = 0; c < x.Cols; c++)
				result[r, c] /= sum;
		}
		return result;
	}

	/// <summary>
	/// Backward through row-wise softmax. For each row with output <c>s</c> and upstream gradient
	/// <c>dOut</c>, the input gradient is <c>s_i · (dOut_i − Σ_j dOut_j·s_j)</c> — the Jacobian of
	/// softmax contracted with the upstream gradient.
	/// </summary>
	public static Matrix SoftmaxRowsBackward(Matrix dOut, Matrix softmaxOut)
	{
		var dx = new Matrix(dOut.Rows, dOut.Cols);
		for (var r = 0; r < dOut.Rows; r++)
		{
			Real dot = 0;
			for (var c = 0; c < dOut.Cols; c++)
				dot += dOut[r, c] * softmaxOut[r, c];

			for (var c = 0; c < dOut.Cols; c++)
				dx[r, c] = softmaxOut[r, c] * (dOut[r, c] - dot);
		}
		return dx;
	}

	#endregion

	#region GELU

	/// <summary>
	/// GELU activation (tanh approximation): <c>0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))</c>.
	/// </summary>
	public static Matrix Gelu(Matrix x) => x.Map(GeluScalar);

	/// <summary>Backward through GELU: <c>dIn = dOut ⊙ gelu'(x)</c>, element-wise.</summary>
	public static Matrix GeluBackward(Matrix dOut, Matrix input)
	{
		var dx = new Matrix(dOut.Rows, dOut.Cols);
		for (var i = 0; i < dOut.Data.Length; i++)
			dx.Data[i] = dOut.Data[i] * GeluDerivative(input.Data[i]);
		return dx;
	}

	// GELU's elementwise transcendental math is done in double (cheap, and keeps the activation and
	// its derivative accurate) and narrowed to Real at the boundary via RealMath.
	private static Real GeluScalar(Real xr)
	{
		double x = (double)xr;
		var inner = GeluCoeff * (x + 0.044715 * x * x * x);
		return (Real)(0.5 * x * (1.0 + Math.Tanh(inner)));
	}

	private static Real GeluDerivative(Real xr)
	{
		double x = (double)xr;
		var inner = GeluCoeff * (x + 0.044715 * x * x * x);
		var tanh = Math.Tanh(inner);
		var sech2 = 1.0 - tanh * tanh; // d/du tanh(u) = 1 - tanh²(u)
		var dInner = GeluCoeff * (1.0 + 3.0 * 0.044715 * x * x);
		return (Real)(0.5 * (1.0 + tanh) + 0.5 * x * sech2 * dInner);
	}

	#endregion

	#region LayerNorm

	/// <summary>
	/// Row-wise layer normalization with a learnable scale (<paramref name="gamma"/>) and shift
	/// (<paramref name="beta"/>), each shape <c>[1, features]</c>. Normalizes each row to zero mean /
	/// unit variance, then scales and shifts. Returns the output plus the per-row mean and inverse
	/// standard deviation needed by the backward pass.
	/// </summary>
	public static (Matrix Output, double[] Mean, double[] InvStd) LayerNorm(Matrix x, Matrix gamma, Matrix beta)
	{
		var output = new Matrix(x.Rows, x.Cols);
		var mean = new double[x.Rows];
		var invStd = new double[x.Rows];
		var n = x.Cols;

		for (var r = 0; r < x.Rows; r++)
		{
			// Per-row mean/variance are accumulated in double for stability, so Real cells are cast
			// to double where they meet these accumulators (the Fixed<->double boundary is explicit).
			var m = 0.0;
			for (var c = 0; c < n; c++)
				m += (double)x[r, c];
			m /= n;

			var variance = 0.0;
			for (var c = 0; c < n; c++)
			{
				var d = (double)x[r, c] - m;
				variance += d * d;
			}
			variance /= n;

			var inv = 1.0 / Math.Sqrt(variance + Epsilon);
			mean[r] = m;
			invStd[r] = inv;

			for (var c = 0; c < n; c++)
			{
				var norm = ((double)x[r, c] - m) * inv;
				output[r, c] = (Real)(norm * (double)gamma[0, c] + (double)beta[0, c]);
			}
		}

		return (output, mean, invStd);
	}

	/// <summary>
	/// Backward through layer norm. Returns the input gradient and accumulates the gradients for the
	/// scale and shift parameters into <paramref name="gammaGrad"/> / <paramref name="betaGrad"/>.
	/// Uses the standard per-row formula:
	/// <c>dx = (γ·invStd/N)·(N·dOut − Σ dOut − x̂·Σ(dOut·x̂))</c>, with x̂ the normalized input.
	/// </summary>
	public static Matrix LayerNormBackward(
		Matrix dOut, Matrix x, Matrix gamma, double[] mean, double[] invStd, Matrix gammaGrad, Matrix betaGrad)
	{
		var dx = new Matrix(x.Rows, x.Cols);
		var n = x.Cols;

		for (var r = 0; r < x.Rows; r++)
		{
			// Reductions run in double (matching the double per-row stats); Real cells are cast to
			// double at each access and results cast back to Real at Matrix writes.
			var sumDOut = 0.0;      // Σ (dOut · γ)
			var sumDOutXhat = 0.0;  // Σ (dOut · γ · x̂)
			for (var c = 0; c < n; c++)
			{
				var xhat = ((double)x[r, c] - mean[r]) * invStd[r];
				var dNorm = (double)dOut[r, c] * (double)gamma[0, c]; // gradient w.r.t. the normalized value
				sumDOut += dNorm;
				sumDOutXhat += dNorm * xhat;

				// Parameter grads: dβ = Σ dOut, dγ = Σ dOut·x̂ (summed across rows).
				betaGrad[0, c] += dOut[r, c];
				gammaGrad[0, c] += (Real)((double)dOut[r, c] * xhat);
			}

			for (var c = 0; c < n; c++)
			{
				var xhat = ((double)x[r, c] - mean[r]) * invStd[r];
				var dNorm = (double)dOut[r, c] * (double)gamma[0, c];
				dx[r, c] = (Real)(invStd[r] / n * (n * dNorm - sumDOut - xhat * sumDOutXhat));
			}
		}

		return dx;
	}

	#endregion

	#region Cross-entropy

	/// <summary>
	/// Mean cross-entropy loss over positions for next-token prediction. <paramref name="logits"/> is
	/// <c>[time, vocab]</c>; <paramref name="targets"/> gives the correct token id per position, or
	/// <c>-1</c> to ignore that position (padding). Returns the scalar loss and the gradient w.r.t.
	/// the logits (softmax − one-hot, averaged over the counted positions) — the two fused because
	/// the softmax-cross-entropy gradient is far simpler combined than as separate ops.
	/// </summary>
	public static (double Loss, Matrix DLogits) CrossEntropy(Matrix logits, int[] targets)
	{
		var probs = SoftmaxRows(logits);
		var dLogits = new Matrix(logits.Rows, logits.Cols);
		var loss = 0.0;
		var counted = 0;

		for (var r = 0; r < logits.Rows; r++)
		{
			var target = targets[r];
			if (target < 0)
				continue; // ignored (padding) position

			counted++;
			loss += -Math.Log((double)probs[r, target] + 1e-12);

			for (var c = 0; c < logits.Cols; c++)
				dLogits[r, c] = probs[r, c];
			dLogits[r, target] -= 1; // subtract the one-hot target
		}

		if (counted == 0)
			return (0.0, dLogits);

		loss /= counted;
		// Average the gradient over the counted positions to match the mean loss.
		var scale = (Real)(1.0 / counted);
		for (var i = 0; i < dLogits.Data.Length; i++)
			dLogits.Data[i] *= scale;

		return (loss, dLogits);
	}

	#endregion
}
