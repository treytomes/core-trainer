namespace CoreTrainer.Examples.MiniGptScratch;

/// <summary>
/// A dense 2-D matrix of Reals stored row-major, the single numeric workhorse for the
/// from-scratch transformer. Every activation and parameter in the model is a <see cref="Matrix"/>,
/// and every forward/backward operation is expressed in terms of the handful of primitives here.
/// </summary>
/// <remarks>
/// This deliberately replaces what a tensor library (TorchSharp in the sibling example) would hide:
/// there is no autograd and no vectorized backend — just explicit loops so every multiply and sum
/// in a transformer is visible and steppable. Shapes are 2-D <c>[rows, cols]</c>; in the model rows
/// are almost always sequence positions (time) and cols are feature dimensions.
/// </remarks>
public sealed class Matrix
{
	#region Fields

	private readonly Real[] _data;

	#endregion

	#region Constructors

	/// <summary>Creates a zero-filled matrix of the given shape.</summary>
	public Matrix(int rows, int cols)
	{
		Rows = rows;
		Cols = cols;
		_data = new Real[rows * cols];
	}

	/// <summary>Wraps an existing row-major backing array (no copy).</summary>
	private Matrix(int rows, int cols, Real[] data)
	{
		Rows = rows;
		Cols = cols;
		_data = data;
	}

	#endregion

	#region Properties

	/// <summary>Number of rows.</summary>
	public int Rows { get; }

	/// <summary>Number of columns.</summary>
	public int Cols { get; }

	/// <summary>The flat row-major backing store, for callers that need raw access.</summary>
	public Real[] Data => _data;

	/// <summary>Element access at (row, col).</summary>
	public Real this[int row, int col]
	{
		get => _data[row * Cols + col];
		set => _data[row * Cols + col] = value;
	}

	#endregion

	#region Methods

	/// <summary>Creates a matrix from a 2-D array literal (rows of columns).</summary>
	public static Matrix From(Real[,] values)
	{
		var rows = values.GetLength(0);
		var cols = values.GetLength(1);
		var m = new Matrix(rows, cols);
		for (var r = 0; r < rows; r++)
			for (var c = 0; c < cols; c++)
				m[r, c] = values[r, c];
		return m;
	}

	/// <summary>Returns a deep copy.</summary>
	public Matrix Clone() => new(Rows, Cols, (Real[])_data.Clone());

	/// <summary>Fills every element with <paramref name="value"/>.</summary>
	public void Fill(Real value) => Array.Fill(_data, value);

	/// <summary>Zeroes every element.</summary>
	public void Zero() => Array.Clear(_data);

	/// <summary>
	/// Standard matrix product <c>this · other</c>: <c>[Rows, K] · [K, Cols] -> [Rows, Cols]</c>.
	/// </summary>
	/// <remarks>
	/// The dominant cost of the whole model, so it is parallelized over output rows: each row of the
	/// result depends only on the corresponding row of <c>this</c> and (read-only) all of
	/// <paramref name="other"/>, and writes only to its own slice of the result — so rows are fully
	/// independent with no shared writes, making a <see cref="Parallel.For"/> safe without locks.
	/// Small products stay serial, where thread-scheduling overhead would outweigh the work.
	/// </remarks>
	public Matrix MatMul(Matrix other)
	{
		if (Cols != other.Rows)
			throw new ArgumentException($"MatMul shape mismatch: [{Rows},{Cols}] · [{other.Rows},{other.Cols}].");

		var result = new Matrix(Rows, other.Cols);

		// Parallelize only when there is enough work to amortize the thread overhead; otherwise the
		// tiny matrices (e.g. the gradient-check model) run faster serially.
		const long ParallelThreshold = 512 * 512;
		var work = (long)Rows * Cols * other.Cols;
		if (work >= ParallelThreshold && Rows > 1)
			Parallel.For(0, Rows, i => MultiplyRow(i, other, result));
		else
			for (var i = 0; i < Rows; i++)
				MultiplyRow(i, other, result);

		return result;
	}

	/// <summary>
	/// Computes one output row <paramref name="i"/> of <c>this · other</c> into <paramref name="result"/>.
	/// Writes only to row <paramref name="i"/> of the result, so distinct rows are independent.
	/// </summary>
	private void MultiplyRow(int i, Matrix other, Matrix result)
	{
		var rowOffset = i * result.Cols;
		var aRowOffset = i * Cols;
		// Each output cell is a full dot product accumulated at doubled fractional width (see Mac) and
		// narrowed once — instead of the naive "result += a*b" which rounds every product to the model
		// grain and compounds K roundings into a gradient-direction error at low precision. The loop is
		// j-outer / k-inner so each cell owns one accumulator; the zero-skip on `a` is preserved.
		for (var j = 0; j < result.Cols; j++)
		{
			var mac = new Mac();
			for (var k = 0; k < Cols; k++)
			{
				var a = _data[aRowOffset + k];
				if (a == default)
					continue;
				mac.MultiplyAdd(a, other._data[k * other.Cols + j]);
			}
			result._data[rowOffset + j] = mac.Result;
		}
	}

	/// <summary>
	/// Ternary matrix product <c>this · (γ·T)</c>: multiplies dense <c>this</c> <c>[Rows, K]</c> by a
	/// ternary weight <c>T</c> <c>[K, Cols]</c> whose entries are -1, 0, or +1, scaled by
	/// <paramref name="gamma"/>. Because the weight entries are ternary, the inner product needs
	/// <b>no multiplies</b>: +1 adds the input, -1 subtracts it, 0 is skipped, and the whole result is
	/// scaled by γ once at the end. This is BitNet's add-only matmul — it both skips the (often
	/// majority) zero weights and replaces every remaining multiply with an add/subtract, which for a
	/// fixed-point <c>Real</c> avoids the expensive widening multiply entirely.
	/// </summary>
	/// <param name="trits">Ternary weights in {-1,0,+1}, row-major <c>[K, Cols]</c> (K = this.Cols).</param>
	/// <param name="cols">Number of output columns (the weight's column count).</param>
	/// <param name="gamma">The per-tensor scale applied once to every output.</param>
	/// <remarks>
	/// Parallelized over output rows exactly like <see cref="MatMul"/>: each row of the result is
	/// written by one iteration and reads only shared inputs, so there are no cross-row races.
	/// </remarks>
	public Matrix TernaryMatMul(sbyte[] trits, int cols, Real gamma)
	{
		if (trits.Length != Cols * cols)
			throw new ArgumentException($"Ternary weight length {trits.Length} != {Cols}·{cols}.");

		var result = new Matrix(Rows, cols);

		const long ParallelThreshold = 512 * 512;
		var work = (long)Rows * Cols * cols;
		if (work >= ParallelThreshold && Rows > 1)
			Parallel.For(0, Rows, i => TernaryRow(i, trits, cols, gamma, result));
		else
			for (var i = 0; i < Rows; i++)
				TernaryRow(i, trits, cols, gamma, result);

		return result;
	}

	/// <summary>
	/// Accumulates one output row of <see cref="TernaryMatMul"/> using add/subtract/skip, then scales
	/// the finished row by γ. Iterating over the ternary weight <c>k</c> lets the zero-skip fire on the
	/// sparse operand (the whole point), and reuses each input element across all output columns.
	/// </summary>
	private void TernaryRow(int i, sbyte[] trits, int cols, Real gamma, Matrix result)
	{
		var rowOffset = i * cols;
		var inRowOffset = i * Cols;
		for (var k = 0; k < Cols; k++)
		{
			var x = _data[inRowOffset + k];
			if (x == default)
				continue;
			var tritOffset = k * cols;
			for (var j = 0; j < cols; j++)
			{
				var t = trits[tritOffset + j];
				if (t > 0)
					result._data[rowOffset + j] += x;      // +1: add
				else if (t < 0)
					result._data[rowOffset + j] -= x;      // -1: subtract
				// 0: skip (the majority case for a sparse ternary weight)
			}
		}

		// Apply the per-tensor scale once, after the add-only accumulation.
		for (var j = 0; j < cols; j++)
			result._data[rowOffset + j] *= gamma;
	}

	/// <summary>Returns the transpose <c>[Cols, Rows]</c>.</summary>
	public Matrix Transpose()
	{
		var result = new Matrix(Cols, Rows);
		for (var r = 0; r < Rows; r++)
			for (var c = 0; c < Cols; c++)
				result[c, r] = this[r, c];
		return result;
	}

	/// <summary>Element-wise sum <c>this + other</c> (same shape).</summary>
	public Matrix Add(Matrix other)
	{
		RequireSameShape(other);
		var result = new Matrix(Rows, Cols);
		for (var i = 0; i < _data.Length; i++)
			result._data[i] = _data[i] + other._data[i];
		return result;
	}

	/// <summary>Adds <paramref name="other"/> into this matrix in place (same shape).</summary>
	public void AddInPlace(Matrix other)
	{
		RequireSameShape(other);
		for (var i = 0; i < _data.Length; i++)
			_data[i] += other._data[i];
	}

	/// <summary>
	/// Adds a row vector <paramref name="bias"/> (shape <c>[1, Cols]</c>) to every row — the standard
	/// broadcast of a bias across sequence positions.
	/// </summary>
	public Matrix AddRowVector(Matrix bias)
	{
		if (bias.Rows != 1 || bias.Cols != Cols)
			throw new ArgumentException($"Bias must be [1,{Cols}], got [{bias.Rows},{bias.Cols}].");

		var result = new Matrix(Rows, Cols);
		for (var r = 0; r < Rows; r++)
			for (var c = 0; c < Cols; c++)
				result[r, c] = this[r, c] + bias[0, c];
		return result;
	}

	/// <summary>Scales every element by <paramref name="scalar"/>.</summary>
	public Matrix Scale(Real scalar)
	{
		var result = new Matrix(Rows, Cols);
		for (var i = 0; i < _data.Length; i++)
			result._data[i] = _data[i] * scalar;
		return result;
	}

	/// <summary>Element-wise (Hadamard) product with <paramref name="other"/> (same shape).</summary>
	public Matrix Hadamard(Matrix other)
	{
		RequireSameShape(other);
		var result = new Matrix(Rows, Cols);
		for (var i = 0; i < _data.Length; i++)
			result._data[i] = _data[i] * other._data[i];
		return result;
	}

	/// <summary>Applies <paramref name="fn"/> to every element, returning a new matrix.</summary>
	public Matrix Map(Func<Real, Real> fn)
	{
		var result = new Matrix(Rows, Cols);
		for (var i = 0; i < _data.Length; i++)
			result._data[i] = fn(_data[i]);
		return result;
	}

	/// <summary>Sums the rows into a single <c>[1, Cols]</c> row vector (column-wise totals).</summary>
	public Matrix SumRows()
	{
		var result = new Matrix(1, Cols);
		for (var r = 0; r < Rows; r++)
			for (var c = 0; c < Cols; c++)
				result[0, c] += this[r, c];
		return result;
	}

	private void RequireSameShape(Matrix other)
	{
		if (Rows != other.Rows || Cols != other.Cols)
			throw new ArgumentException($"Shape mismatch: [{Rows},{Cols}] vs [{other.Rows},{other.Cols}].");
	}

	#endregion
}
