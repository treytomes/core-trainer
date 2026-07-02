using static TorchSharp.torch;

namespace CoreTrainer.Examples.MiniGpt;

/// <summary>
/// The bipolar-logic algebra for <see cref="LogicFeedForward"/>: the constant matrix that turns the
/// basis <c>{1, a, b, ab}</c> into all 16 two-input Boolean gate outputs at once. A local copy of the
/// same verified construction used by the CoreTrainer.Examples.LogicMlp example (kept local so the two
/// example projects stay independent).
/// </summary>
internal static class BipolarGateAlgebra
{
	/// <summary>
	/// Builds the constant <c>[4, 16]</c> multilinear-coefficient matrix. Column k holds (α, β, γ, δ) so
	/// that <c>f_k(a,b) = α + β·a + γ·b + δ·ab</c> is the exact bipolar multilinear extension of the k-th
	/// two-input Boolean gate (smooth on the cube, exact at the corners). Coefficients are the
	/// Walsh–Hadamard transform of the gate's four corner outputs.
	/// </summary>
	public static Tensor Coefficients()
	{
		var c = new float[4, 16]; // rows: {1, a, b, ab}; cols: gate index k
		for (var k = 0; k < 16; k++)
		{
			float alpha = 0, beta = 0, gamma = 0, delta = 0;
			for (var corner = 0; corner < 4; corner++)
			{
				var aBip = ((corner & 1) == 1) ? 1f : -1f;
				var bBip = (((corner >> 1) & 1) == 1) ? 1f : -1f;
				var vBip = (((k >> corner) & 1) == 1) ? 1f : -1f;
				alpha += vBip;
				beta += vBip * aBip;
				gamma += vBip * bBip;
				delta += vBip * aBip * bBip;
			}
			c[0, k] = alpha / 4f;
			c[1, k] = beta / 4f;
			c[2, k] = gamma / 4f;
			c[3, k] = delta / 4f;
		}
		return tensor(c);
	}

	/// <summary>Gate index 10 is the "A" passthrough (f = a); biasing initial gate logits toward it makes
	/// the logic net start near identity (a uniform 16-gate mixture outputs 0 for all inputs — dead).</summary>
	public const int PassthroughAGate = 10;

	/// <summary>Initial logit bias on the passthrough gate; softmax(3 vs ~0) ≈ 0.7 weight at init.</summary>
	public const double PassthroughInitBias = 3.0;
}
