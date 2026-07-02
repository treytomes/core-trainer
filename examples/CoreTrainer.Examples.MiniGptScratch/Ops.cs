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

	#region Gradient clipping

	/// <summary>
	/// Shared value limit for residual-stream gradient clipping. Far above any legitimate gradient
	/// magnitude here, but well under the fixed-point range ceiling (±32768 at Fixed32/F=16), so it only
	/// clamps the rare LayerNorm-backward 1/std spike that would otherwise overflow — and is inert in
	/// double/float where gradients are tiny.
	/// </summary>
	public static readonly Real GradientClipLimit = (Real)256;

	/// <summary>
	/// Clamps every element of a gradient to <c>±limit</c>, in place. Value-based gradient clipping —
	/// the standard guard against exploding gradients, and here also a hard requirement for low-range
	/// fixed-point: LayerNorm's backward multiplies by <c>1/std</c>, which spikes when a normalized row
	/// had near-zero variance, and unclamped that spike overflows a 32-bit fixed-point value. Applied to
	/// the residual-stream gradients where the spike propagates. A no-op region for values already
	/// within range, so it does not distort normal training.
	/// </summary>
	public static void ClipGradient(Matrix grad, Real limit)
	{
		var negLimit = -limit;
		var data = grad.Data;
		for (var i = 0; i < data.Length; i++)
		{
			if (data[i] > limit)
				data[i] = limit;
			else if (data[i] < negLimit)
				data[i] = negLimit;
		}
	}

	#endregion

	#region Softmax

	/// <summary>
	/// Row-wise softmax: each row becomes a probability distribution. Subtracts the row max first for
	/// numerical stability (does not change the result).
	/// </summary>
	public static Matrix SoftmaxRows(Matrix x, int[]? rowValidCounts = null)
	{
		var result = new Matrix(x.Rows, x.Cols);
		for (var r = 0; r < x.Rows; r++)
		{
			// Columns [0, valid) participate; the rest are masked out to exactly 0. This replaces the
			// old "-inf sentinel then exp" masking, which subtracted the sentinel from the row max and
			// could overflow in low-range fixed-point. Masked cells never enter the max, exp, or sum.
			var valid = rowValidCounts?[r] ?? x.Cols;

			var max = RealMath.NegativeInfinity;
			for (var c = 0; c < valid; c++)
				max = RealMath.Max(max, x[r, c]);

			Real sum = 0;
			for (var c = 0; c < valid; c++)
			{
				var e = RealMath.Exp(x[r, c] - max);
				result[r, c] = e;
				sum += e;
			}

			for (var c = 0; c < valid; c++)
				result[r, c] /= sum;
			// Columns [valid, Cols) stay 0 (Matrix is zero-initialized).
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

	// Real-typed GELU constants, for the pure fixed-point path.
	private static readonly Real GeluCoeffR = (Real)Math.Sqrt(2.0 / Math.PI);
	private static readonly Real GeluCubic = (Real)0.044715;
	private static readonly Real Half = (Real)0.5;

	// GELU (tanh approximation). When the pure fixed-point transcendentals are enabled, the whole
	// computation runs in Real via RealMath.Tanh — no floating point. Otherwise it computes in double
	// (cheaper and more accurate) and narrows at the boundary.
	private static Real GeluScalar(Real xr)
	{
		if (Fixed.UseFixedTranscendentals)
		{
			var inner = GeluCoeffR * (xr + GeluCubic * xr * xr * xr);
			return Half * xr * ((Real)1 + RealMath.Tanh(inner));
		}
		double x = (double)xr;
		var innerD = GeluCoeff * (x + 0.044715 * x * x * x);
		return (Real)(0.5 * x * (1.0 + Math.Tanh(innerD)));
	}

	private static Real GeluDerivative(Real xr)
	{
		if (Fixed.UseFixedTranscendentals)
		{
			var innerR = GeluCoeffR * (xr + GeluCubic * xr * xr * xr);
			var tanhR = RealMath.Tanh(innerR);
			var sech2R = (Real)1 - tanhR * tanhR;
			var dInnerR = GeluCoeffR * ((Real)1 + (Real)3 * GeluCubic * xr * xr);
			return Half * ((Real)1 + tanhR) + Half * xr * sech2R * dInnerR;
		}
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

			// The inverse standard deviation is the one transcendental in LayerNorm. On the pure
			// fixed-point path it goes through RealMath.Sqrt (no Math.*); otherwise it uses Math.Sqrt.
			// The mean/variance sums above are plain arithmetic — a wider accumulator, not a float op.
			double inv;
			if (Fixed.UseFixedTranscendentals)
			{
				var stdR = RealMath.Sqrt((Real)(variance + Epsilon));
				inv = 1.0 / (double)stdR;
			}
			else
			{
				inv = 1.0 / Math.Sqrt(variance + Epsilon);
			}
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
