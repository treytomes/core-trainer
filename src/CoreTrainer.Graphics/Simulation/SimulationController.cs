using CoreTrainer.Core;

namespace CoreTrainer.Graphics.Simulation;

/// <summary>
/// Controls the execution of a Core Wars simulation with support for pause, step, and variable speed.
/// </summary>
public class SimulationController
{
	#region Fields

	private Mars? _mars;
	private float _cycleAccumulator;

	#endregion

	#region Constructors

	public SimulationController()
	{
		StepsPerSecond = 1;
		State = SimulationState.Stopped;
	}

	#endregion

	#region Properties

	/// <summary>
	/// The MARS simulation instance.
	/// </summary>
	public Mars? Mars => _mars;

	/// <summary>
	/// Current state of the simulation.
	/// </summary>
	public SimulationState State { get; private set; }

	/// <summary>
	/// Simulation speed in steps (cycles) per second.
	/// </summary>
	public int StepsPerSecond { get; set; }

	/// <summary>
	/// Last execution result from StepCycle.
	/// </summary>
	public ExecutionResult LastResult { get; private set; }

	#endregion

	#region Public Methods

	/// <summary>
	/// Initializes a new battle with the given warriors and their starting positions.
	/// </summary>
	/// <param name="warriors">The warriors to load, in order.</param>
	/// <param name="startAddresses">Starting address for each warrior (parallel to <paramref name="warriors"/>).</param>
	/// <param name="coreSize">Core memory size.</param>
	/// <param name="maxCycles">Maximum cycles.</param>
	/// <param name="maxProcesses">Maximum processes per warrior.</param>
	public void Initialize(
		IReadOnlyList<Warrior> warriors,
		IReadOnlyList<int> startAddresses,
		int coreSize = 8000,
		int maxCycles = 80000,
		int maxProcesses = 8000)
	{
		ArgumentNullException.ThrowIfNull(warriors);
		ArgumentNullException.ThrowIfNull(startAddresses);
		if (warriors.Count == 0)
			throw new ArgumentException("At least one warrior is required.", nameof(warriors));
		if (warriors.Count != startAddresses.Count)
			throw new ArgumentException("Each warrior must have exactly one start address.", nameof(startAddresses));

		_mars = new Mars(coreSize, maxCycles, maxProcesses, warriors.Count);
		for (var i = 0; i < warriors.Count; i++)
		{
			_mars.LoadWarrior(i, warriors[i], startAddresses[i]);
		}

		_cycleAccumulator = 0f;

		State = SimulationState.Paused;
		LastResult = ExecutionResult.Continue;
	}

	/// <summary>
	/// Starts or resumes the simulation.
	/// </summary>
	public void Run()
	{
		if (_mars == null)
			return;

		if (State == SimulationState.Finished)
			return;

		State = SimulationState.Running;
	}

	/// <summary>
	/// Pauses the simulation.
	/// </summary>
	public void Pause()
	{
		if (State == SimulationState.Running)
			State = SimulationState.Paused;
	}

	/// <summary>
	/// Toggles between running and paused.
	/// </summary>
	public void TogglePause()
	{
		if (State == SimulationState.Running)
			Pause();
		else if (State == SimulationState.Paused)
			Run();
	}

	/// <summary>
	/// Executes a single simulation cycle (ignores speed, always steps once).
	/// </summary>
	public void Step()
	{
		if (_mars == null || State == SimulationState.Finished)
			return;

		LastResult = _mars.StepCycle();

		if (LastResult != ExecutionResult.Continue)
		{
			State = SimulationState.Finished;
		}
	}

	/// <summary>
	/// Resets the simulation to its initial state.
	/// Requires re-initializing with warriors.
	/// </summary>
	public void Reset()
	{
		_mars = null;
		_cycleAccumulator = 0f;
		State = SimulationState.Stopped;
		LastResult = ExecutionResult.Continue;
	}

	/// <summary>
	/// Updates the simulation based on delta time and steps per second.
	/// Called every frame when running.
	/// </summary>
	/// <param name="deltaTime">Time since last frame in seconds.</param>
	public void Update(float deltaTime)
	{
		if (_mars == null || State != SimulationState.Running)
			return;

		// Accumulate cycles based on steps per second
		// deltaTime is in seconds, so multiply by StepsPerSecond to get cycles this frame
		_cycleAccumulator += deltaTime * StepsPerSecond;

		// Execute cycles based on accumulated time
		while (_cycleAccumulator >= 1.0f)
		{
			LastResult = _mars.StepCycle();
			_cycleAccumulator -= 1.0f;

			if (LastResult != ExecutionResult.Continue)
			{
				State = SimulationState.Finished;
				break;
			}
		}
	}

	/// <summary>
	/// Increases the simulation speed.
	/// </summary>
	public void IncreaseSpeed()
	{
		StepsPerSecond = StepsPerSecond switch
		{
			< 1 => 1,
			< 10 => 10,
			< 60 => 60,
			< 600 => 600,
			< 6000 => 6000,
			< 60000 => 60000,
			_ => 60000
		};
	}

	/// <summary>
	/// Decreases the simulation speed.
	/// </summary>
	public void DecreaseSpeed()
	{
		StepsPerSecond = StepsPerSecond switch
		{
			> 60000 => 60000,
			> 6000 => 6000,
			> 600 => 600,
			> 60 => 60,
			> 10 => 10,
			> 1 => 1,
			_ => 1
		};
	}

	/// <summary>
	/// Gets a display string for the current speed.
	/// </summary>
	public string GetSpeedDisplay()
	{
		return StepsPerSecond switch
		{
			>= 60000 => "Max",
			>= 1000 => $"{StepsPerSecond / 1000}k/s",
			_ => $"{StepsPerSecond}/s"
		};
	}

	#endregion
}

/// <summary>
/// Represents the state of the simulation.
/// </summary>
public enum SimulationState
{
	/// <summary>
	/// Simulation has not been initialized.
	/// </summary>
	Stopped,

	/// <summary>
	/// Simulation is initialized but paused.
	/// </summary>
	Paused,

	/// <summary>
	/// Simulation is actively running.
	/// </summary>
	Running,

	/// <summary>
	/// Simulation has completed (winner determined or max cycles reached).
	/// </summary>
	Finished
}
