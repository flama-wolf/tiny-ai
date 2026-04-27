class CharTokenizer:
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        self.unk_idx = 0

    def fit(self, text: str):
        chars = sorted(list(set(text)))
        if '<UNK>' not in chars:
            chars.insert(0, '<UNK>')
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.unk_idx = self.char_to_idx['<UNK>']

    def encode(self, text: str) -> list[int]:
        unk = getattr(self, 'unk_idx', 0)
        return [self.char_to_idx.get(ch, unk) for ch in text]

    def decode(self, indices: list[int]) -> str:
        return ''.join([self.idx_to_char.get(idx, '<UNK>') for idx in indices])
