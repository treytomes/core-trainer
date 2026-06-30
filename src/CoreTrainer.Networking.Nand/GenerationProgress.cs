namespace CoreTrainer.Networking.Nand;

/// <summary>
/// A snapshot reported after each generation of a <see cref="CgpEvolution"/> run, for
/// progress reporting.
/// </summary>
/// <param name="Generation">Zero-based index of the generation just completed.</param>
/// <param name="Fitness">The current best (parent) fitness after this generation.</param>
/// <param name="Best">The current best (parent) genome after this generation.</param>
public sealed record GenerationProgress(int Generation, double Fitness, NandGenome Best);
