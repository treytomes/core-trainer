using TorchSharp.Modules;
using static TorchSharp.torch;

namespace CoreTrainer.Examples.LogicMlp;

/// <summary>
/// The common surface of the logic-gate networks so the same training loop and metrics can drive
/// either architecture: <see cref="LogicGateNet"/> (learns gate function, fixed wiring) and
/// <see cref="JointLogicNet"/> (learns wiring and gate function). Lets the ladder run both and compare.
/// </summary>
public interface ILogicNet
{
	/// <summary>All learnable parameters, for the optimizer.</summary>
	IEnumerable<Parameter> Parameters { get; }

	/// <summary>
	/// Evaluates on a batch (shape <c>[batch, inputCount]</c>, values in {0,1}); returns output bits
	/// (shape <c>[batch, outputCount]</c>, {0,1}). See implementations for the temperature/hard semantics.
	/// </summary>
	Tensor Forward(Tensor x01, double temperature, bool hard);

	/// <summary>The hardened circuit: per layer, each gate's chosen function and its two input indices.</summary>
	IReadOnlyList<ExtractedGate[]> ExtractNetlist();
}
