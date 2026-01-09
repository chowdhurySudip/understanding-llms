"""Byte-Pair Encoding (BPE) Tokenizer implementation.

This module implements a fast BPE tokenizer for natural language processing.
BPE is a data compression technique that iteratively merges the most frequent
pairs of bytes in a text corpus to build a vocabulary.

Key features:
- Multiprocessing support for handling large datasets efficiently
- Special token handling
- Vocabulary and merge serialization

Typical usage:
    tokenizer = BPETokenizer(pattern=PAT, max_vocab_size=10000, special_tokens=["<|endoftext|>"])
    tokenizer.train(input_path="data.txt", num_processes=4)
    tokenizer.save(dest="artifacts/")
"""

import os
import json
from pathlib import Path
from time import time
import multiprocessing
from functools import partial
from collections import Counter, defaultdict

import regex as re
from tqdm import tqdm

from .create_chunks import find_chunk_boundaries

# Regular expression pattern for pre-tokenization (splits text into base units)
# Handles contractions, letters, numbers, symbols, and whitespace
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Default special token for marking text boundaries
ENDOFTEXT_TOKEN = "<|endoftext|>"


def read_file(input_path: str | os.PathLike) -> str:
    """Read the input file and return its content as a string.
    
    Args:
        input_path: Path to the text file to read
        
    Returns:
        The complete file content as a UTF-8 string
    """
    with open(input_path, "r", encoding="utf-8") as f:
        return f.read()


def find_chunks(input_path: str | os.PathLike, num_processes: int = 1) -> list[tuple[int, int]]:
    """Find chunk boundaries in a file for parallel processing.
    
    Divides the file into chunks at special token boundaries to enable
    independent parallel processing while preserving token integrity.
    
    Args:
        input_path: Path to the file to chunk
        num_processes: Number of processes for parallel processing (default: 1)
        
    Returns:
        List of (start, end) byte position tuples defining each chunk
    """
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, ENDOFTEXT_TOKEN.encode("utf-8"))
        return list(zip(boundaries[:-1], boundaries[1:]))


def load_file_chunk(input_path: str | os.PathLike, start: int, end: int) -> str:
    """Load a specific byte range from a file as a UTF-8 string.
    
    Args:
        input_path: Path to the file to read from
        start: Starting byte position
        end: Ending byte position (exclusive)
        
    Returns:
        The decoded UTF-8 string from the specified byte range,
        with invalid sequences replaced
    """
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
    """Worker function to count pre-tokens in a file chunk (for multiprocessing).
    
    Loads a chunk of text, splits it at special tokens, then applies
    regex pre-tokenization and counts the frequency of each pre-token.
    
    Args:
        input_path: Path to the input file
        special_tokens: List of special tokens to preserve as boundaries
        pattern: Regular expression pattern for pre-tokenization
        boundary: (start, end) byte positions defining the chunk
        
    Returns:
        Counter mapping pre-token strings to their frequency counts
    """
    # Load the chunk of text
    raw_text = load_file_chunk(input_path, *boundary)
    
    # Split text at special token boundaries
    split_pattern = "|".join(re.escape(st) for st in special_tokens)
    text_segments = re.split(split_pattern, raw_text)

    # Count occurrences of each pre-token across all segments
    return Counter(
        m.group() 
        for text in text_segments
        for m in re.finditer(pattern, text)
    )

class BPETokenizer:
    """Fast Byte-Pair Encoding (BPE) Tokenizer implementation.
    
    Implements the BPE algorithm which iteratively merges the most frequent
    pairs of bytes in a corpus to build a compact vocabulary. The tokenizer
    supports multiprocessing for efficient training on large datasets.
    
    The vocabulary consists of:
    - 256 base byte tokens (0-255)
    - Special tokens (e.g., <|endoftext|>)
    - Learned merge tokens
    
    Attributes:
        pattern: Regular expression for pre-tokenization
        max_vocab_size: Target vocabulary size including base bytes
        special_tokens: List of special tokens to preserve
        vocab: Mapping of token ID to byte sequences
        merges: Ordered list of (byte_pair) merges learned during training
        
    Example:
        >>> tokenizer = BPETokenizer(PAT, 10000, ["<|endoftext|>"])
        >>> tokenizer.train("data.txt", num_processes=4)
        >>> tokenizer.save("artifacts/")
    """
    MAX_BYTES = 256  # Number of possible byte values (0-255)

    def __init__(self, pattern: str, max_vocab_size: int, special_tokens: list[str]) -> None:
        """Initialize BPE tokenizer.
        
        Args:
            pattern: Regular expression pattern for pre-tokenization
            max_vocab_size: Maximum vocabulary size (must be >= 256 + len(special_tokens))
            special_tokens: List of special tokens to add to vocabulary
        """
        self.pattern = pattern
        self.max_vocab_size = max_vocab_size
        self.special_tokens = special_tokens
        
        # Initialize vocabulary with base bytes and special tokens
        self._initialize_vocab()

        # Calculate how many merges we need to reach target vocabulary size
        self.initial_vocab_len = len(self.vocab)
        self.num_merges = max_vocab_size - self.initial_vocab_len
        
        # Training state variables
        self.merges = []  # Ordered list of learned merges
        self.pretoken_counts = []  # (token_tuple, count) pairs for each pre-token
        self.pair_count = defaultdict(int)  # Frequency of each byte pair
        self.pair_token_mapping = defaultdict(set)  # Which tokens contain each pair

    def _initialize_vocab(self) -> None:
        """Initialize vocabulary with byte values and special tokens.
        
        Creates the initial vocabulary containing:
        - IDs 0-255: Single byte tokens
        - IDs 256+: Special tokens (encoded as UTF-8 bytes)
        """
        # Add all possible single bytes as base tokens
        self.vocab = {i: bytes([i]) for i in range(self.MAX_BYTES)}
        
        # Append special tokens after the byte tokens
        for idx, t in enumerate(self.special_tokens):
            self.vocab[self.MAX_BYTES + idx] = t.encode("utf-8")

    def _update_stats(self) -> None:
        """Update pair statistics for BPE merges.
        
        Iterates through all pre-tokens and counts the frequency of each
        adjacent byte pair. Also maintains a mapping of which tokens contain
        each pair for efficient updates during merging.
        """
        for token_id, (token, count) in enumerate(self.pretoken_counts):
            # Count all adjacent pairs in this token
            for a, b in zip(token, token[1:]):
                pair = (a, b)
                self.pair_count[pair] += count  # Add frequency
                self.pair_token_mapping[pair].add(token_id)  # Track which tokens have this pair

    def _get_max_pair(self) -> tuple[bytes, bytes]:
        """Get the most frequent pair for merging.
        
        Selects the byte pair with the highest frequency. In case of ties,
        uses lexicographic ordering of the pair as a tiebreaker for determinism.
        
        Returns:
            The (byte_a, byte_b) pair to merge next
        """
        return max(self.pair_count, key=lambda k: (self.pair_count[k], k))

    def _merge(
        self,
        token: tuple[bytes, ...],
        freq: int,
        pair: tuple[bytes, bytes],
        new_id: bytes,
        token_id: int
    ) -> tuple[bytes, ...]:
        """Merge the most frequent pair in a token and update statistics.
        
        Replaces all occurrences of the specified byte pair with a new merged
        token ID, and updates pair frequency counts for newly created and
        removed pairs.
        
        Args:
            token: Tuple of byte sequences representing the current token
            freq: Frequency count of this token in the corpus
            pair: The (byte_a, byte_b) pair to merge
            new_id: The new merged byte sequence to replace the pair
            token_id: Index of this token in pretoken_counts
            
        Returns:
            New token tuple with the pair merged
        """
        new_tokens, i = [], 0
        token_mapping_to_remove = set()
        
        while i < len(token):
            # Check if current position starts the pair to merge
            if i < len(token) - 1 and token[i] == pair[0] and token[i + 1] == pair[1]:
                # Update pair counts for the new adjacent pairs created by merging
                if i > 0:
                    # New pair: (previous_token, merged_token)
                    self.pair_count[(new_tokens[-1], new_id)] += freq
                    self.pair_token_mapping[(new_tokens[-1], new_id)].add(token_id)
                    # Old pair: (previous_token, first_byte_of_pair) is removed
                    self.pair_count[(new_tokens[-1], token[i])] -= freq
                    token_mapping_to_remove.add((new_tokens[-1], token[i]))
                if (i + 2) < len(token):
                    # New pair: (merged_token, next_token)
                    self.pair_count[(new_id, token[i + 2])] += freq
                    self.pair_token_mapping[(new_id, token[i + 2])].add(token_id)
                    # Old pair: (second_byte_of_pair, next_token) is removed
                    self.pair_count[(token[i + 1], token[i + 2])] -= freq
                    token_mapping_to_remove.add((token[i + 1], token[i + 2]))
                
                # Add the merged token
                new_tokens.append(new_id)
                i += 2  # Skip both bytes of the merged pair
            else:
                # Keep this token unchanged
                new_tokens.append(token[i])
                i += 1

        # Clean up pair_token_mapping for pairs that no longer exist in this token
        for pair_to_remove in token_mapping_to_remove:
            # Only remove if the pair truly doesn't exist anymore
            available = False
            for token_pair in zip(new_tokens, new_tokens[1:]):
                if token_pair == pair_to_remove:
                    available = True
            if not available:
                self.pair_token_mapping[pair_to_remove].remove(token_id)

        return tuple(new_tokens)

    def train(self, input_path: str | os.PathLike, num_processes: int = 1) -> None:
        """Train the tokenizer on raw text using BPE algorithm.
        
        The training process:
        1. Load and pre-tokenize the text (optionally using multiprocessing)
        2. Count frequency of each pre-token
        3. Initialize pair statistics
        4. Iteratively merge the most frequent pair until reaching target vocab size
        
        Args:
            input_path: Path to the input text file
            num_processes: Number of processes for parallel processing (default: 1)
        """
        # Store filename for later use in save()
        self.filename = Path(input_path).stem

        # Step 1: Count pre-tokens (parallelized if num_processes > 1)
        start = time()
        if num_processes > 1:
            # Find chunk boundaries for parallel processing
            chunk_endpoints = find_chunks(input_path, num_processes)

            # Process chunks in parallel using multiprocessing
            map_function = partial(_worker_count_chunk_pretokens, input_path, self.special_tokens, self.pattern)
            with multiprocessing.Pool(processes=num_processes) as pool:
                pretoken_counters = pool.map(map_function, chunk_endpoints)
            
            # Merge counters from all chunks
            final_counter = Counter()
            for counter in pretoken_counters:
                final_counter.update(counter)

        else:
            # Single-process version: load entire file and pre-tokenize
            raw_text = read_file(input_path)

            # Split text at special tokens to preserve them
            split_pattern = "|".join(re.escape(st) for st in self.special_tokens)
            text_segments = re.split(split_pattern, raw_text)

            # Apply regex pre-tokenization and count frequencies
            final_counter = Counter(
                m.group()
                for text in text_segments
                for m in re.finditer(self.pattern, text)
            )
        print(f"pretoken_counters - Total: {time()-start:.2f}s")

        # Step 2: Convert pre-tokens to byte tuples with their counts
        start = time()
        self.pretoken_counts = [
            (tuple([bytes([b]) for b in token.encode("utf-8")]), count)
            for token, count in final_counter.items()
        ]
        print(f"self.pretoken_counts - Total: {time()-start:.2f}s")

        # Step 3: Initialize pair statistics for all pre-tokens
        start = time()
        self._update_stats()
        print(f"_update_stats - Total: {time()-start:.2f}s")

        # Step 4: Iteratively merge the most frequent pair
        start = time()
        for idx in tqdm(range(self.num_merges)):
            # Find the most frequent byte pair
            max_pair = self._get_max_pair()
            self.max_pair = max_pair
            self.merges.append(max_pair)
            
            # Create new vocabulary entry for the merged pair
            bytepair = b"".join(max_pair)
            self.vocab[self.initial_vocab_len + idx] = bytepair

            # Update all tokens that contain this pair
            affected_token_ids = list(self.pair_token_mapping[max_pair])
            for token_id in affected_token_ids:
                token, count = self.pretoken_counts[token_id]
                merged_token = self._merge(token, count, max_pair, bytepair, token_id)
                self.pretoken_counts[token_id] = (merged_token, count)
            
            # Remove the merged pair from consideration
            del self.pair_count[max_pair]
        print(f"_merge - Total: {time()-start:.2f}s")
    
    def save(self, dest):
        """Save vocabulary and merges to disk.
        
        Creates two files:
        - {filename}-vocab.json: Maps token IDs to their string representations
        - {filename}-merges.txt: Lists merge operations in order (JSON format)
        
        The vocabulary is saved with latin-1 encoding to preserve byte values.
        
        Args:
            dest: Destination directory path where files will be saved
        """
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)

        # Save vocabulary as JSON
        # Convert bytes to strings using latin-1 encoding for safe serialization
        vocab_json = {
            token_id: token_bytes.decode("latin-1")
            for token_id, token_bytes in self.vocab.items()
        }
        with open(dest/f"{self.filename}-vocab.json", "w", encoding="utf-8") as fp:
            json.dump(vocab_json, fp, ensure_ascii=False, indent=2)
        
        # Save merges as text file (one merge per line in JSON format)
        with open(dest/f"{self.filename}-merges.txt", "w", encoding="utf-8") as fp:
            for a, b in self.merges:
                a, b = a.decode("latin-1"), b.decode("latin-1")
                fp.write(json.dumps([a, b]) + "\n")

def train_tokenizer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a BPE tokenizer on the input text file.
    
    Convenience function that creates a tokenizer, trains it, saves it,
    and returns the learned vocabulary and merges.
    
    Args:
        input_path: Path to the input text file
        vocab_size: Desired vocabulary size (must be >= 256 + len(special_tokens))
        special_tokens: List of special tokens to include in the vocabulary
        **kwargs: Additional keyword arguments:
            - num_processes (int): Number of parallel processes (default: 3)
            - dest (str): Directory to save artifacts (default: "artifacts")
    
    Returns:
        tuple containing:
            - vocab: Dictionary mapping token IDs to byte strings
            - merges: Ordered list of (byte_a, byte_b) merge pairs
    """
    num_processes = kwargs.get("num_processes", 3)
    
    # Create and train tokenizer
    tokenizer = BPETokenizer(PAT, vocab_size, special_tokens)
    tokenizer.train(input_path, num_processes=num_processes)
    
    # Save to disk
    tokenizer.save(kwargs.get("dest", "artifacts"))
    
    return tokenizer.vocab, tokenizer.merges

def load_tokenizers(vocab_path, merges_path):
    """Load vocabulary and merges from saved files.
    
    Reads a previously saved tokenizer's vocabulary and merge operations
    from disk. Handles both token->id and id->token vocab formats.
    
    Args:
        vocab_path: Path to the vocabulary JSON file
        merges_path: Path to the merges text file
        
    Returns:
        tuple containing:
            - id_to_bytes: Dictionary mapping token IDs to byte sequences
            - merges: Ordered list of (byte_a, byte_b) merge pairs
    """
    # Load vocabulary from JSON
    with open(vocab_path, "r", encoding="utf-8") as vf:
        raw_vocab = json.load(vf)
    
    id_to_bytes: dict[int, bytes] = {}
    if raw_vocab:
        sample_key, sample_val = next(iter(raw_vocab.items()))
        
        if isinstance(sample_val, int):
            # Format: token -> id mapping
            for token_str, idx in raw_vocab.items():
                # Decode using latin-1 to match save() encoding
                id_to_bytes[idx] = token_str.encode("latin-1")
        else:
            # Format: id -> token mapping (keys may be string ints)
            for idx_str, token_str in raw_vocab.items():
                try:
                    idx_int = int(idx_str)
                except ValueError:
                    continue
                id_to_bytes[idx_int] = token_str.encode("latin-1")
    
    # Load merges from text file
    merges: list[tuple[bytes, bytes]] = []
    with open(merges_path, "r", encoding="utf-8") as mf:
        for line in mf:
            cleaned = line.rstrip("\n")
            if not cleaned:
                continue
            try:
                # Try parsing as JSON first (new format)
                parts = json.loads(cleaned)
                if len(parts) != 2:
                    continue
            except json.JSONDecodeError:
                # Fall back to space-separated format (old format)
                parts = cleaned.split(" ", 1)
                if len(parts) != 2:
                    continue
            a, b = parts

            # Encode back to bytes using latin-1
            merges.append((a.encode("latin-1"), b.encode("latin-1")))
    
    return id_to_bytes, merges

if __name__ == "__main__":
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    start = time()
    vocab, merges = train_tokenizer(input_path, 10000, [ENDOFTEXT_TOKEN], num_processes=10)
    print(f"Total: {time()-start:.2f}s")
    loaded_vocab, loaded_merges = load_tokenizers(
        "artifacts/TinyStoriesV2-GPT4-train-vocab.json", 
        "artifacts/TinyStoriesV2-GPT4-train-merges.txt"
    )
    assert vocab == loaded_vocab
    assert merges == loaded_merges  # List equality preserves order