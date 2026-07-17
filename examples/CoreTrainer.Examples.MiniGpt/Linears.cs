using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace CoreTrainer.Examples.MiniGpt;

/// <summary>
/// Factory for the model's linear layers, switching every call site between a full-precision
/// <see cref="Linear"/> and a BitNet ternary <see cref="BitLinear"/> from one flag — so the whole model
/// can be made ternary without changing its wiring.
/// </summary>
internal static class Linears
{
	/// <summary>A linear layer: BitNet ternary when <paramref name="useBitLinear"/>, else full precision.</summary>
	public static Module<Tensor, Tensor> Make(long inFeatures, long outFeatures, bool useBitLinear, bool hasBias = true) =>
		useBitLinear
			? new BitLinear(inFeatures, outFeatures, hasBias)
			: Linear(inFeatures, outFeatures, hasBias);
}
