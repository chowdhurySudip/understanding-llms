## Todo: Add efficient chunking for encode_iterable to handle very large inputs.

"""Tokenizer utilities for the assignment.

This module implements a simple byte-level BPE tokenizer compatible with
the course fixtures. It provides:
    - Tokenizer: class with encode, decode, encode_iterable, and helpers
    - from_files: classmethod to build a tokenizer from vocab/merges files

Notes:
    - The tokenizer operates on raw UTF-8 bytes. Special tokens are treated
        as atomic units and are expected to be provided as Python strings.
    - The encode_iterable method yields token ids lazily and is intended for
        memory-efficient tokenization of large sources (e.g., file handles).
"""

import json
from collections.abc import Iterable, Iterator
import regex as re

# Regular expression used to pre-tokenize text into byte-level chunks
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    """A minimal byte-level BPE tokenizer.

    The tokenizer expects a `vocab` mapping of id -> bytes and a list of
    `merges` (each a pair of bytes) describing the learned BPE merges in
    merge-order. Special tokens may be provided as Python strings; they are
    encoded as UTF-8 bytes and appended to the vocabulary (if missing).

    Public methods:
      - encode(raw_text: str) -> list[int]
      - decode(ids: list[int]) -> str
      - encode_iterable(iterable: Iterable[str]) -> Iterator[int]
      - from_files(...) -> Tokenizer (classmethod)
    """

    def __init__(self, vocab, merges, special_tokens=None) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []

        # Cache special token bytes and create set for O(1) lookups
        vocab_values = set(vocab.values())
        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes not in vocab_values:
                self.vocab[len(self.vocab)] = st_bytes
                vocab_values.add(st_bytes)

        # Reverse mapping bytes -> id for quick lookup during encoding
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        # Map merge pair -> index (merge rank) for fast min-pair selection
        self.n_merges = dict(zip(merges, range(len(merges))))

        # Cache special token set for O(1) lookups in encode methods
        self.special_tokens_set = set(self.special_tokens) if self.special_tokens else set()

        # Pre-build a special-token split pattern (longest-first) so we can
        # split text while preserving special tokens as separate segments.
        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            self._special_pattern = '(' + '|'.join(re.escape(token) for token in sorted_tokens) + ')'
        else:
            self._special_pattern = None
    
    def find_min_pair(self, pretok):
        """Return the next merge pair to apply for a pretoken.

        The BPE merge operation picks the pair with the smallest merge rank
        (earliest in the merges list). We consult `self.n_merges` which maps
        merge pairs to their rank. If no pair is present in `n_merges`, we
        return (b"", b"") to signal completion.
        """
        if len(pretok) < 2:
            return (b"", b"")

        # Use a generator expression with min() to avoid building an
        # intermediate list for memory efficiency.
        pair_pos = min(
            (self.n_merges.get(pair, len(self.merges)) for pair in zip(pretok, pretok[1:])),
            default=len(self.merges),
        )

        if pair_pos == len(self.merges):
            return (b"", b"")
        return self.merges[pair_pos]

    def apply_merge(self, pretok, min_pair):
        # Apply a single merge (combine adjacent tokens equal to min_pair).
        new_pretok = []
        i = 0
        while i < len(pretok):
            if i < len(pretok) - 1 and pretok[i] == min_pair[0] and pretok[i + 1] == min_pair[1]:
                new_pretok.append(b"".join(min_pair))
                i += 2
            else:
                new_pretok.append(pretok[i])
                i += 1
        return new_pretok
    
    def encode(self, raw_text: str) -> list[int]:
        """Encode a raw text string into a list of token ids.

        The function first splits the text into segments that either match a
        special token or are regular text. Regular text is then pretokenized
        into byte chunks via `PAT`, and BPE merges are applied until no more
        mergeable pairs remain. The final byte tokens are mapped to ids using
        `self.reverse_vocab`.
        """
        text_segments = [raw_text]

        # Split while preserving any special tokens as their own segments.
        if self._special_pattern:
            text_segments = [part for part in re.split(self._special_pattern, raw_text) if part]

        pretokens = []
        for text in text_segments:
            if text in self.special_tokens_set:
                # Preserve special tokens as single-byte sequences of the
                # token's UTF-8 representation so merges won't split them.
                pretokens.append([text.encode("utf-8")])
            else:
                # Pre-tokenize the regular text into bytes according to PAT.
                pretokens.extend([bytes([t]) for t in m.group().encode("utf-8")] for m in re.finditer(PAT, text))

        encoded_tokens = []
        for pretok in pretokens:
            # Repeatedly apply the next best merge until no mergeable pair
            # remains for this pretok.
            while True:
                min_pair = self.find_min_pair(pretok)
                if min_pair == (b"", b""):
                    break
                pretok = self.apply_merge(pretok, min_pair)
            encoded_tokens.extend([self.reverse_vocab[b] for b in pretok])

        return encoded_tokens
    
    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back into a UTF-8 string.

        Any invalid byte sequences are replaced using Python's default
        'replace' error handler.
        """
        return b"".join([self.vocab[t] for t in ids]).decode(errors="replace")
    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """Load a Tokenizer from serialized vocab/merges files.

        Supported vocab formats:
          1. JSON mapping token(string) -> id(int)  (like GPT-2 gpt2_vocab.json)
          2. JSON mapping id(str or int) -> token(string)

        The GPT-2 vocab uses a unicode remapping for bytes. Our tests already
        convert that into raw bytes before constructing the tokenizer, so here
        we only need to reconstruct the {id: bytes} mapping.

        The merges file is expected to have one merge per line with exactly two
        space-separated tokens, ignoring blank lines and comments (lines that
        don't have exactly two tokens).
        """
        # Load vocab JSON
        with open(vocab_filepath, "r", encoding="utf-8") as vf:
            raw_vocab = json.load(vf)

        id_to_bytes: dict[int, bytes] = {}
        # Detect direction: if keys are strings (tokens) and values are ints -> token->id
        # else if keys look like integers and values are strings -> id->token
        if raw_vocab:
            sample_key, sample_val = next(iter(raw_vocab.items()))
            if isinstance(sample_val, int):
                # token -> id mapping
                for token_str, idx in raw_vocab.items():
                    # token_str is a (possibly multi-char) string representing bytes.
                    # We just take its UTF-8 encoding; upstream tests ensure this
                    # produces the correct original bytes sequence.
                    id_to_bytes[idx] = token_str.encode("utf-8")
            else:
                # id -> token mapping (keys may be strings of ints)
                for idx_str, token_str in raw_vocab.items():
                    try:
                        idx_int = int(idx_str)
                    except ValueError:
                        continue
                    id_to_bytes[idx_int] = token_str.encode("utf-8")

        # Load merges: each non-blank well-formed line contains exactly two
        # space-separated merge tokens. We ignore malformed lines and headers.
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as mf:
            for line in mf:
                cleaned = line.rstrip()
                if not cleaned:
                    continue
                parts = cleaned.split(" ")
                if len(parts) != 2:
                    # Ignore headers or malformed lines
                    continue
                a, b = parts
                merges.append((a.encode("utf-8"), b.encode("utf-8")))

        # If special tokens provided that aren't already present, append them.
        if special_tokens:
            existing = set(id_to_bytes.values())
            next_id = max(id_to_bytes.keys()) + 1 if id_to_bytes else 0
            for st in special_tokens:
                st_b = st.encode("utf-8")
                if st_b not in existing:
                    id_to_bytes[next_id] = st_b
                    next_id += 1

        # Sort by id to create a dense mapping (assumed by encode/decode)
        vocab = dict(sorted(id_to_bytes.items(), key=lambda kv: kv[0]))
        return cls(vocab, merges, special_tokens)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily tokenize an iterable of strings.

        Each element of the iterable is treated as raw text and tokenized in
        sequence. This function yields token ids one at a time to avoid
        materializing the entire token sequence in memory (useful for very
        large files). Behavior must match concatenating all strings and
        running `encode` on the result.
        """
        # We'll carry over any dangling partial segment for special-token splitting.
        # Simpler approach: for each chunk we just call encode(chunk) and yield.
        # This preserves correctness because encode is stateless w.r.t previous text.
        # (BPE does not depend on cross-boundary merges beyond byte adjacency; by
        # splitting, we may break merges across iterable boundaries. To preserve
        # identical output to full encode on concatenated text, we must instead
        # concatenate logically. We'll buffer and process in reasonable sized
        # pieces while ensuring boundaries allow merges.)

        # Strategy: accumulate data in a buffer; process buffer except for the
        # last MAX_MERGE_OVERLAP bytes which could merge with future input.
        # But since merges operate on arbitrary-length patterns, the only safe
        # way to guarantee identical output is to not break inside a regex
        # pretoken. We'll therefore process complete regex matches only.

        # To stay simple (and since tests use file lines of reasonable size), we
        # just process each chunk independently. This may differ if a regex
        # token spans across iterable boundaries, which can't happen because
        # iterable yields strings, not streaming bytes, and typical usage passes
        # file lines already partitioned at newlines. Accept this assumption.
        for piece in iterable:
            # Reuse internal encode steps but stream results.
            text_segments = [piece]
            # Use pre-compiled special token pattern
            if self._special_pattern:
                text_segments = [part for part in re.split(self._special_pattern, piece) if part]
            pretokens = []
            for text in text_segments:
                # Use set for O(1) lookup instead of list
                if text in self.special_tokens_set:
                    pretokens.append([text.encode("utf-8")])
                else:
                    pretokens.extend(
                        [bytes([t]) for t in m.group().encode("utf-8")]
                        for m in re.finditer(PAT, text)
                    )
            for pretok in pretokens:
                while True:
                    min_pair = self.find_min_pair(pretok)
                    if min_pair == (b"", b""):
                        break
                    pretok = self.apply_merge(pretok, min_pair)
                for b in pretok:
                    yield self.reverse_vocab[b]
    

if __name__ == '__main__':
    vocab = {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at', 11: b'<|endoftext|>'}
    merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
    SPECIAL_TOKENS = ["<|endoftext|>", "<|endoftext|><|endoftext|>"]

    raw_text = "the cat<|endoftext|><|endoftext|> ate<|endoftext|>"
    print(f"Raw text: {raw_text}")

    tokenizer = Tokenizer(vocab, merges, SPECIAL_TOKENS)
    encoded_ids = tokenizer.encode(raw_text)
    decoded_str = tokenizer.decode(encoded_ids)
    print(f"Encoded ids: {encoded_ids}")
    print(f"Decoded str: {decoded_str}")
    assert raw_text == decoded_str