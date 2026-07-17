using static TorchSharp.torch;

namespace CoreTrainer.Examples.MiniGpt;

/// <summary>
/// Shared BitNet-b1.58 ternary quantization: turns a full-precision latent weight into <c>{-1, 0, +1}</c>
/// values with a single per-tensor scale, using a straight-through estimator so the latent weights stay
/// trainable. Used by <see cref="BitLinear"/> and <see cref="BitEmbedding"/>.
/// </summary>
internal static class Ternary
{
	private const double Epsilon = 1e-5;

	/// <summary>
	/// Quantizes <paramref name="latent"/> to the effective ternary weight <c>round(clip(w/γ,-1,1))·γ</c>
	/// with <c>γ = mean(|w|)</c> (the BitNet absmean scale), via a straight-through estimator: the forward
	/// value is the crisp ternary weight, but the gradient flows to the latent weight as if quantization
	/// were the identity. So the optimizer trains smooth latent weights while the network computes with
	/// ternary ones — and many weights settle to exactly 0 (prunable), the cullability payoff.
	/// </summary>
	public static Tensor QuantizeSte(Tensor latent)
	{
		using var noGradScale = no_grad();
		var gamma = latent.abs().mean().item<float>();
		var inv = 1.0 / (gamma + Epsilon);

		// round(clip(w * inv, -1, 1)) in {-1,0,+1}, then rescale by gamma. Computed under no_grad and
		// re-attached via STE so the ternary values carry no graph of their own.
		var ternary = (latent * inv).clamp(-1.0, 1.0).round() * gamma;

		// Straight-through: value = ternary (detached), gradient = identity to the latent weight.
		return latent + (ternary - latent).detach();
	}
}
