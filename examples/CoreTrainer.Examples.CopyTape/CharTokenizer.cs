namespace CoreTrainer.Examples.CopyTape;

/// <summary>
/// A character-level tokenizer: every distinct character in the training text becomes one token. The
/// vocabulary is fixed from a representative corpus string, so encode/decode is exact and there is no
/// out-of-vocabulary case (every character in an 8-bit tape also appears in a 1-bit tape).
/// </summary>
public sealed class CharTokenizer
{
	private readonly Dictionary<char, int> _charToId;
	private readonly char[] _idToChar;

	/// <summary>Builds the vocabulary from the distinct characters in <paramref name="corpus"/>.</summary>
	public CharTokenizer(string corpus)
	{
		_idToChar = corpus.Distinct().OrderBy(c => c).ToArray();
		_charToId = new Dictionary<char, int>(_idToChar.Length);
		for (var i = 0; i < _idToChar.Length; i++)
			_charToId[_idToChar[i]] = i;
	}

	/// <summary>Number of distinct tokens (characters).</summary>
	public int VocabSize => _idToChar.Length;

	/// <summary>Encodes text to token ids.</summary>
	public long[] Encode(string text)
	{
		var ids = new long[text.Length];
		for (var i = 0; i < text.Length; i++)
			ids[i] = _charToId[text[i]];
		return ids;
	}

	/// <summary>Decodes token ids back to text.</summary>
	public string Decode(IEnumerable<long> ids) =>
		new(ids.Select(id => _idToChar[(int)id]).ToArray());
}
