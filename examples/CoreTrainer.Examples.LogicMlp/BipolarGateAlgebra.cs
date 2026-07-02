using static TorchSharp.torch;

namespace CoreTrainer.Examples.LogicMlp;

/// <summary>
/// The shared bipolar-logic algebra used by every logic-gate network here: the constant matrix that
/// turns the basis <c>{1, a, b, ab}</c> into all 16 two-input Boolean gate outputs at once.
/// </summary>
internal static class BipolarGateAlgebra
{
	/// <summary>
	/// Builds the constant <c>[4, 16]</c> multilinear-coefficient matrix. Column k holds (α, β, γ, δ) for
	/// the k-th two-input Boolean gate, so <c>f_k(a,b) = α + β·a + γ·b + δ·ab</c> is the exact bipolar
	/// multilinear extension of that gate (smooth on the cube, exact at the corners). The coefficients are
	/// the Walsh–Hadamard transform of the gate's four corner outputs: <c>¼ Σ_corner value·basis(corner)</c>.
	/// Verified to reproduce all 16 truth tables exactly.
	/// </summary>
	public static Tensor Coefficients()
	{
		var c = new float[4, 16]; // rows: {1, a, b, ab} basis; cols: gate index k
		for (var k = 0; k < 16; k++)
		{
			// Corner c ∈ 0..3 encodes (a_bit = c&1, b_bit = (c>>1)&1). Gate k's output bit at that corner
			// is (k >> c) & 1. Work in bipolar for both inputs and outputs.
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

	/// <summary>Gate index 10 is the "A" passthrough (f = a). Biasing initial gate logits toward it makes
	/// a network start as identity wiring — the fix for the dead uniform-mixture initialization.</summary>
	public const int PassthroughAGate = 10;

	/// <summary>Initial logit bias on the passthrough gate; softmax(3 vs ~0) ≈ 0.7 weight at init.</summary>
	public const double PassthroughInitBias = 3.0;
}
