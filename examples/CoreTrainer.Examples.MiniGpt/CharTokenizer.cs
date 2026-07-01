namespace CoreTrainer.Examples.MiniGpt;

/// <summary>
/// A character-level tokenizer: every distinct character in the training text becomes one token.
/// </summary>
/// <remarks>
/// Character-level keeps the example fully self-contained — the vocabulary is just the set of
/// characters that appear in the corpus, so there is no out-of-vocabulary case, no external
/// tokenizer model, and the "did it memorize the sentences?" test is exact: decode the generated
/// ids back to a string and compare. The trade-off (long sequences, no sub-word structure) does not
/// matter for a tiny memorization demo.
/// </remarks>
public sealed class CharTokenizer
{
	#region Fields

	private readonly Dictionary<char, int> _charToId;
	private readonly char[] _idToChar;

	#endregion

	#region Constructors

	/// <summary>Builds the vocabulary from the distinct characters in <paramref name="corpus"/>.</summary>
	public CharTokenizer(string corpus)
	{
		_idToChar = corpus.Distinct().OrderBy(c => c).ToArray();
		_charToId = new Dictionary<char, int>(_idToChar.Length);
		for (var i = 0; i < _idToChar.Length; i++)
			_charToId[_idToChar[i]] = i;
	}

	#endregion

	#region Properties

	/// <summary>Number of distinct tokens (characters).</summary>
	public int VocabSize => _idToChar.Length;

	#endregion

	#region Methods

	/// <summary>Encodes text to a list of token ids.</summary>
	public IReadOnlyList<long> Encode(string text)
	{
		var ids = new long[text.Length];
		for (var i = 0; i < text.Length; i++)
			ids[i] = _charToId[text[i]];
		return ids;
	}

	/// <summary>Decodes token ids back to text.</summary>
	public string Decode(IEnumerable<long> ids) =>
		new(ids.Select(id => _idToChar[(int)id]).ToArray());

	#endregion
}
