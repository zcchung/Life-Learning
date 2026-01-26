from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CharTokenizer:
    stoi: Dict[str, int]
    itos: Dict[int, str]

    @classmethod
    def build(cls, text: str) -> "CharTokenizer":
        vocab = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(vocab)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def build_sequences(ids: List[int], seq_len: int) -> List[List[int]]:
    return [ids[i : i + seq_len] for i in range(0, len(ids) - seq_len, seq_len)]


def main() -> None:
    text = "hello tiny llm"
    tokenizer = CharTokenizer.build(text)
    ids = tokenizer.encode(text)
    seqs = build_sequences(ids, seq_len=4)
    print("vocab size:", len(tokenizer.stoi))
    print("first seq:", seqs[0], "->", tokenizer.decode(seqs[0]))


if __name__ == "__main__":
    main()
