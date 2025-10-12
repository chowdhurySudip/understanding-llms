import os
from time import time
import multiprocessing
from functools import partial
from collections import Counter, defaultdict

import regex as re
from tqdm import tqdm

from cs336_basics.tokenizers.create_chunks import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
ENDOFTEXT_TOKEN = "<|endoftext|>"


def read_file(input_path: str | os.PathLike) -> str:
    """Read the input file and return its content as a string."""
    with open(input_path, "r", encoding="utf-8") as f:
        return f.read()


def find_chunks(input_path: str | os.PathLike, num_processes: int = 1) -> list[tuple[int, int]]:
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, ENDOFTEXT_TOKEN.encode("utf-8"))
        return list(zip(boundaries[:-1], boundaries[1:]))


def load_file_chunk(input_path: str | os.PathLike, start: int, end: int) -> str:
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        return chunk

def _worker_count_chunk_pretokens(
        input_path: str | os.PathLike,  
        special_tokens: list[str], 
        pattern: str,
        boundary: tuple[int, int]
) -> Counter:
    raw_text = load_file_chunk(input_path, *boundary)
    split_pattern = "|".join(re.escape(st) for st in special_tokens)
    text_segments = re.split(split_pattern, raw_text)

    return Counter(
        m.group() 
        for text in text_segments
        for m in re.finditer(pattern, text)
    )

class BPETokenizer:
    """
    Fast Byte-Pair Encoding (BPE) Tokenizer implementation.
    """
    MAX_BYTES = 256

    def __init__(self, pattern: str, max_vocab_size: int, special_tokens: list[str]) -> None:
        self.pattern = pattern
        self.max_vocab_size = max_vocab_size
        self.special_tokens = special_tokens
        
        self._initialize_vocab()

        self.initial_vocab_len = len(self.vocab)
        self.num_merges = max_vocab_size - self.initial_vocab_len
        self.merges = []
        self.pretoken_counts = []
        self.pair_count = defaultdict(int)
        self.pair_token_mapping = defaultdict(set)

    def _initialize_vocab(self) -> None:
        """Initialize vocabulary with byte values and special tokens."""
        self.vocab = {i: bytes([i]) for i in range(self.MAX_BYTES)}
        for idx, t in enumerate(self.special_tokens):
            self.vocab[self.MAX_BYTES + idx] = t.encode("utf-8")

    def _update_stats(self) -> None:
        """Update pair statistics for BPE merges."""
        for token_id, (token, count) in enumerate(self.pretoken_counts):
            for a, b in zip(token, token[1:]):
                pair = (a, b)
                self.pair_count[pair] += count
                self.pair_token_mapping[pair].add(token_id)

    def _get_max_pair(self) -> tuple[bytes, bytes]:
        """Get the most frequent pair for merging."""
        return max(self.pair_count, key=lambda k: (self.pair_count[k], k))

    def _merge(
        self,
        token: tuple[bytes, ...],
        freq: int,
        pair: tuple[bytes, bytes],
        new_id: bytes,
        token_id: int
    ) -> tuple[bytes, ...]:
        """Merge the most frequent pair in a token."""
        new_tokens, i = [], 0
        token_mapping_to_remove = set()
        while i < len(token):
            if i < len(token) - 1 and token[i] == pair[0] and token[i + 1] == pair[1]:
                if i > 0:
                    self.pair_count[(new_tokens[-1], new_id)] += freq
                    self.pair_token_mapping[(new_tokens[-1], new_id)].add(token_id)
                    self.pair_count[(new_tokens[-1], token[i])] -= freq
                    token_mapping_to_remove.add((new_tokens[-1], token[i]))
                if (i + 2) < len(token):
                    self.pair_count[(new_id, token[i + 2])] += freq
                    self.pair_token_mapping[(new_id, token[i + 2])].add(token_id)
                    self.pair_count[(token[i + 1], token[i + 2])] -= freq
                    token_mapping_to_remove.add((token[i + 1], token[i + 2]))
                new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(token[i])
                i += 1

        for pair_to_remove in token_mapping_to_remove:
            available = False
            for token_pair in zip(new_tokens, new_tokens[1:]):
                if token_pair == pair_to_remove:
                    available = True
            if not available:
                self.pair_token_mapping[pair_to_remove].remove(token_id)

        return tuple(new_tokens)

    def train(self, input_path: str | os.PathLike, num_processes: int = 1) -> None:
        """Train the tokenizer on raw text."""

        start = time()
        if num_processes > 1:
            chunk_endpoints = find_chunks(input_path, num_processes)

            map_function = partial(_worker_count_chunk_pretokens, input_path, self.special_tokens, self.pattern)
            with multiprocessing.Pool(processes=num_processes) as pool:
                pretoken_counters = pool.map(map_function, chunk_endpoints)
            
            final_counter = Counter()
            for counter in pretoken_counters:
                final_counter.update(counter)

        else:
            raw_text = read_file(input_path)

            split_pattern = "|".join(re.escape(st) for st in self.special_tokens)
            text_segments = re.split(split_pattern, raw_text)

            final_counter = Counter(
                m.group()
                for text in text_segments
                for m in re.finditer(self.pattern, text)
            )
        print(f"pretoken_counters - Total: {time()-start:.2f}s")

        start = time()
        self.pretoken_counts = [
            (tuple([bytes([b]) for b in token.encode("utf-8")]), count)
            for token, count in final_counter.items()
        ]
        print(f"self.pretoken_counts - Total: {time()-start:.2f}s")

        start = time()
        self._update_stats()
        print(f"_update_stats - Total: {time()-start:.2f}s")

        start = time()
        for idx in tqdm(range(self.num_merges)):
            max_pair = self._get_max_pair()
            self.max_pair = max_pair
            self.merges.append(max_pair)
            bytepair = b"".join(max_pair)
            self.vocab[self.initial_vocab_len + idx] = bytepair

            affected_token_ids = list(self.pair_token_mapping[max_pair])
            for token_id in affected_token_ids:
                token, count = self.pretoken_counts[token_id]
                merged_token = self._merge(token, count, max_pair, bytepair, token_id)
                self.pretoken_counts[token_id] = (merged_token, count)
            
            del self.pair_count[max_pair]
        print(f"_merge - Total: {time()-start:.2f}s")


def train_tokenizer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the input text file.
    
    Args:
        input_path (str): Path to the input text file.
        vocab_size (int): Desired vocabulary size.
        special_tokens (list[str]): List of special tokens to include in the vocabulary.
    
    Returns:
        vocab (dict[int, bytes]): The learned vocabulary mapping token IDs to byte strings.
        merges (list[tuple[bytes, bytes]]): The list of merges applied during training.
    """
    num_processes = kwargs.get("num_processes", 3)
    tokenizer = BPETokenizer(PAT, vocab_size, special_tokens)
    tokenizer.train(input_path, num_processes=num_processes)

    return tokenizer.vocab, tokenizer.merges

if __name__ == "__main__":
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    start = time()
    train_tokenizer(input_path, 10000, [ENDOFTEXT_TOKEN], num_processes=10)
    print(f"Total: {time()-start:.2f}s")