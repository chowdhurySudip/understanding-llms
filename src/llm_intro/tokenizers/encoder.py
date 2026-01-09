import os
import json
import time
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import regex as re
import numpy as np

# The base pattern used for pre-tokenization
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(
        self, 
        vocab: Dict[int, bytes], 
        merges: List[Tuple[bytes, bytes]], 
        special_tokens: Optional[List[str]] = None
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.merges_len = len(merges)
        self.special_tokens = special_tokens or []
        self.pat_compiled = re.compile(PAT)

        # Check if any special token is not available in the vocabulary
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in vocab.values():
                self.vocab[len(self.vocab)] = token_bytes
        
        # Mapping for encoding: bytes -> id
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}
        
        # Set eos_token_id if <|endoftext|> is in special tokens
        self.eos_token_id = None
        if "<|endoftext|>" in self.special_tokens:
            eos_bytes = "<|endoftext|>".encode("utf-8")
            self.eos_token_id = self.bytes_to_id[eos_bytes]

        # Pre-compile the special tokens pattern (longest first to handle overlaps)
        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pattern = re.compile(
                f"({'|'.join(re.escape(st) for st in sorted_tokens)})"
            )
        else:
            self.special_pattern = None
        
        # Mapping for merges: pair -> rank
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}

        # Cache to avoid re-calculating BPE for the same words
        self.cache: Dict[bytes, List[int]] = {}
    
    @classmethod
    def from_files(
        cls, 
        vocab_filepath: str, 
        merges_filepath: str, 
        special_tokens: Optional[List[str]] = None
    ) -> "Tokenizer":
        with open(vocab_filepath, "r", encoding="utf-8") as vf:
            raw_vocab = json.load(vf)
        
        # Convert keys to int and values to bytes using latin-1
        vocab = {int(k): v.encode("latin-1") for k, v in raw_vocab.items()}

        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as mf:
            for line in mf:
                line = line.strip()
                if not line: continue
                pair = json.loads(line)
                merges.append((pair[0].encode("latin-1"), pair[1].encode("latin-1")))
        
        return cls(vocab, merges, special_tokens)
    
    def _find_min_pair(self, token_bytes_list: List[bytes]) -> Optional[Tuple[bytes, bytes]]:
        if len(token_bytes_list) < 2:
            return None
        pair_position = min(
            (
                self.merge_ranks.get(pair, self.merges_len)
                for pair in zip(token_bytes_list, token_bytes_list[1:])
            ),
            default=self.merges_len
        )
        if pair_position == self.merges_len:
            return None
        return self.merges[pair_position]
    
    def _apply_merge(
        self, 
        token_bytes_list: List[bytes], 
        pair_to_merge: Tuple[bytes, bytes]
    ) -> List[bytes]:
        merged_bytes = []
        i = 0
        while i < len(token_bytes_list):
            if (
                i < len(token_bytes_list) - 1 and 
                (token_bytes_list[i], token_bytes_list[i + 1]) == pair_to_merge
            ):
                merged_bytes.append(token_bytes_list[i] + token_bytes_list[i + 1])
                i += 2
            else:
                merged_bytes.append(token_bytes_list[i])
                i += 1
        return merged_bytes
    
    def _tokenize_each_token(self, token_bytes: bytes) -> List[int]:
        if token_bytes in self.cache:
            return self.cache[token_bytes]

        # Initialize the list of symbols (bytes)
        token_bytes_list = [bytes([b]) for b in token_bytes]
        while True:
            min_pair = self._find_min_pair(token_bytes_list)
            if min_pair is None:
                break
            token_bytes_list = self._apply_merge(token_bytes_list, min_pair)
        ids = [self.bytes_to_id[t] for t in token_bytes_list]
        self.cache[token_bytes] = ids
        return ids
    
    def _encode_normal_text(self, text: str) -> List[int]:
        ids = []
        for match in self.pat_compiled.finditer(text):
            word_bytes = match.group().encode("utf-8")
            ids.extend(self._tokenize_each_token(word_bytes))
        return ids
    
    def encode(self, text: str) -> List[int]:
        if not self.special_pattern:
            return self._encode_normal_text(text)

        ids = []
        parts = self.special_pattern.split(text)
        for text in parts:
            if text in self.special_tokens:
                ids.append(self.bytes_to_id[text.encode("utf-8")])
            elif text:
                ids.extend(self._encode_normal_text(text))
        return ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id
    
    def decode(self, ids: List[int]) -> str:
        byte_parts = []
        for idx in ids:
            byte_parts.append(self.vocab[idx])
        return b"".join(byte_parts).decode("utf-8", errors="replace")
    

def tokenize_file_to_npy(
    input_path: str,
    vocab_path: str,
    merges_path: str,
    output_npy_path: str,
    special_tokens: Optional[list[str]] = None,
    dtype: np.dtype = np.dtype(np.uint32)  # uint32 handles vocab sizes up to 4 billion
):
    """
    Tokenizes a large text file and saves IDs to a .npy file.
    Memory-efficient: uses streaming and memmap.
    """
    # 1. Initialize the tokenizer
    start_time = time.time()
    tokenizer = Tokenizer.from_files(
        vocab_path, 
        merges_path, 
        special_tokens=special_tokens
    )
    init_time = time.time() - start_time
    print(f"Tokenizer initialized in {init_time:.2f} seconds")

    # 2. First pass: Count total tokens (Optional but recommended for memmap)
    # If the file is truly massive, we need to know the size to pre-allocate.
    # If you don't want to pass twice, you can append to a list, 
    # but that might hit RAM limits.
    print("Counting tokens...")
    count_start = time.time()
    total_tokens = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for _ in tokenizer.encode_iterable(f):
            total_tokens += 1
    count_time = time.time() - count_start
    print(f"Total tokens found: {total_tokens} (took {count_time:.2f} seconds)")

    # 3. Pre-allocate the .npy file using memmap
    # This creates a file on disk that acts like a numpy array
    memmap_start = time.time()
    ids_array = np.memmap(
        "temp_ids.bin", 
        dtype=dtype, 
        mode='w+', 
        shape=(total_tokens,)
    )
    memmap_time = time.time() - memmap_start
    print(f"Memmap allocated in {memmap_time:.2f} seconds")

    # 4. Second pass: Tokenize and write directly to disk
    print("Encoding and writing to disk...")
    encode_start = time.time()
    with open(input_path, "r", encoding="utf-8") as f:
        for i, token_id in enumerate(tokenizer.encode_iterable(f)):
            ids_array[i] = token_id
            
            # Optional: Progress update for very large files
            if i % 1_000_000 == 0 and i > 0:
                print(f"Processed {i} tokens...")
    encode_time = time.time() - encode_start
    print(f"Encoding completed in {encode_time:.2f} seconds")

    # 5. Flush to disk and save as a standard .npy file
    save_start = time.time()
    ids_array.flush()
    
    # Move the raw binary to a final .npy file
    # Note: np.save includes a header; memmap is raw. 
    # To make it a valid .npy, we load the memmap and save it.
    final_array = np.array(ids_array)
    np.save(output_npy_path, final_array)
    
    # Clean up temp file
    os.remove("temp_ids.bin")
    save_time = time.time() - save_start
    total_time = time.time() - start_time
    print(f"Successfully saved to {output_npy_path} (took {save_time:.2f} seconds)")
    print(f"Total time: {total_time:.2f} seconds")


if __name__=="__main__":
    # Example usage:
    src_file = "llm_intro/data/TinyStoriesV2-GPT4-train.txt"
    vocab_file = "llm_intro/artifacts/TinyStoriesV2-GPT4-train-vocab.json"
    merges_file = "llm_intro/artifacts/TinyStoriesV2-GPT4-train-merges.txt"
    tokenize_file_to_npy(
        src_file, 
        vocab_file, 
        merges_file, 
        "llm_intro/artifacts/TinyStoriesV2-GPT4-train-encoded.npy",
        special_tokens=["<|endoftext|>"]
    )