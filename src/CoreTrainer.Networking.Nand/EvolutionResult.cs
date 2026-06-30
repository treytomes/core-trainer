namespace CoreTrainer.Networking.Nand;

/// <summary>
/// The outcome of an evolutionary run: the best genome found and its fitness.
/// </summary>
/// <param name="Best">The fittest genome discovered.</param>
/// <param name="Fitness">Its fitness score (higher is better).</param>
/// <param name="Generations">Number of generations evaluated.</param>
public sealed record EvolutionResult(NandGenome Best, double Fitness, int Generations);
