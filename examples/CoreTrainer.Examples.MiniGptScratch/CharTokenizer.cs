namespace CoreTrainer.Examples.MiniGptScratch;

/// <summary>
/// A character-level tokenizer: every distinct character in the training text becomes one token.
/// Identical in spirit to the TorchSharp example's tokenizer — kept local so this project has no
/// cross-references and can be read start to finish on its own.
/// </summary>
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

	/// <summary>Encodes text to token ids.</summary>
	public int[] Encode(string text)
	{
		var ids = new int[text.Length];
		for (var i = 0; i < text.Length; i++)
			ids[i] = _charToId[text[i]];
		return ids;
	}

	/// <summary>Decodes token ids back to text.</summary>
	public string Decode(IEnumerable<int> ids) =>
		new(ids.Select(id => _idToChar[id]).ToArray());

	#endregion
}
