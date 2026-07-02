namespace CoreTrainer.Examples.MiniGpt;

/// <summary>Which feed-forward block the transformer uses in place of the standard MLP.</summary>
public enum MlpMode
{
	/// <summary>The conventional <c>Linear → GELU → Linear</c> MLP (baseline).</summary>
	Gelu,

	/// <summary>A fully differentiable logic-gate network (tanh bipolarizer, softmax wire/gate blends).</summary>
	LogicSoft,

	/// <summary>A hardened logic circuit (sign bipolarizer, argmax wire/gate) via straight-through.</summary>
	LogicHard,
}
