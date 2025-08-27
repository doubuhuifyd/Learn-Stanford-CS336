__all__ = ["train_bpe_tokenizer", "Tokenizer"]

import regex as re
import os
import json
from typing import BinaryIO, Iterable, Iterator
from collections import defaultdict
from tqdm.contrib.concurrent import process_map
from functools import partial
import config

PAT =  re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
num_processes = 12


def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes
) -> list[int]:
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def pre_token(file: str, special_tokens: list) -> tuple[defaultdict, defaultdict]:
    if not special_tokens:
        chunks = [file]
    else:
        special_tokens = sorted(special_tokens, key=len, reverse=True)
        pattern = "|".join(re.escape(tok) for tok in special_tokens)
        chunks = re.split(pattern, file)
    word_table = defaultdict(int)
    chunk_list = []
    for chunk in chunks:
        print(chunk)
        ans = PAT.finditer(chunk)
        for item in ans:
            utf8_text = item.group(0).encode("utf-8")
            word_table[utf8_text] += 1
            chunk_list.append(tuple(bytes([i]) for i in utf8_text))
    pair_table = defaultdict(int)
    for key, value in word_table.items():
        for i in range(len(key) - 1):
            pair = (key[i:i + 1], key[i + 1:i + 2])
            pair_table[pair] += value
    return word_table, pair_table


def train_bpe_tokenizer(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocabulary_dict = {i: bytes([i]) for i in range(256)}
    vocabulary_num = 256
    for special_token in special_tokens:
        vocabulary_dict[vocabulary_num] = special_token.encode("utf-8")
        vocabulary_num += 1

    file = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            file.append(chunk)

    pre_token_func = partial(pre_token, special_tokens=special_tokens)
    if len(file) < 5:
        with open(input_path, "r", encoding="utf-8") as f:
            file = f.read()
        chunks, pair_table = pre_token_func(file)
    else:
        results = process_map(pre_token_func, file, max_workers=num_processes, chunksize=1)
        chunks = []
        word_table = defaultdict(int)
        pair_table = defaultdict(int)
        for word_tab, pair_tab in results:
            for pair, count in word_tab.items():
                word_table[pair] += count
            for pair, count in pair_tab.items():
                pair_table[pair] += count

    merges = []
    while vocabulary_num < vocab_size:
        merge_pairs = max(pair_table.items(), key=lambda x: (x[1], x[0]))[0]
        merges.append(merge_pairs)
        new_token_bytes = merge_pairs[0] + merge_pairs[1]
        vocabulary_dict[vocabulary_num] = new_token_bytes
        vocabulary_num += 1
        l, r = merge_pairs
        del pair_table[merge_pairs]

        new_chunks = []
        for chunk in chunks:
            i = 0
            new_chunk = []
            while i < len(chunk):
                if i < len(chunk) - 1 and chunk[i] == l and chunk[i + 1] == r:
                    new_chunk.append(new_token_bytes)
                    if i > 0:
                        new_pair = (chunk[i - 1], new_token_bytes)
                        pair_table[new_pair] += 1

                        left_pair = (chunk[i - 1], l)
                        pair_table[left_pair] -= 1
                        if pair_table[left_pair] <= 0:
                            del pair_table[left_pair]

                    if i < len(chunk) - 2:
                        new_pair = (new_token_bytes, chunk[i + 2])
                        pair_table[new_pair] += 1

                        right_pair = (r, chunk[i + 2])
                        pair_table[right_pair] -= 1
                        if pair_table[right_pair] <= 0:
                            del pair_table[right_pair]

                    i += 2
                else:
                    new_chunk.append(chunk[i])
                    i += 1
            new_chunks.append(tuple(new_chunk))
        chunks = new_chunks
    return vocabulary_dict, merges




class Tokenizer():
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        tokens = vocab.values()
        vocab_size = len(tokens)
        print(vocab, merges)
        if special_tokens:
            for special_token in special_tokens:
                if special_token.encode("utf-8") not in tokens:
                    vocab_size += 1
                    vocab[vocab_size] = special_token.encode("utf-8")

        vocab_to_id = {}
        for vocabulary, idx in vocab.items():
            vocab_to_id[vocabulary] = idx
        self.vocab_to_id = vocab_to_id
        self.vocab = vocab
        self.merges = set(merges)
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
            vocab = {int(k): bytes(v, 'latin1') if isinstance(v, str) else bytes(v) for k, v in vocab.items()}

        with open(merges_filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            merge_pairs = [line.strip().split() for line in lines]
            merges = [(a.encode('utf-8'), b.encode('utf-8')) for a, b in merge_pairs]
        print(vocab, merges)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str)-> list[int]:
        special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "|".join(re.escape(token) for token in special_tokens)
        pattern = f"({pattern})"
        chunks = re.split(pattern, text)

        token_id = []
        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                token_id.append(self.vocab_to_id[chunk.encode("utf-8")])
            else:
                ans = PAT.finditer(chunk)
                for item in ans:
                    utf8_text = item.group(0).encode("utf-8")
                    subword_in_bytes = list(bytes([i]) for i in utf8_text)

                    while True:
                        min_token_id = float("inf")
                        merge_idx = -1
                        merge_pair = None
                        for i in range(len(subword_in_bytes) - 1):
                            current_pair = (subword_in_bytes[i], subword_in_bytes[i + 1])
                            if current_pair in self.merges:
                                current_token_id = self.vocab_to_id[current_pair]
                                if current_token_id < min_token_id:
                                    min_token_id = current_token_id
                                    merge_idx = i
                                    merge_pair = subword_in_bytes[i] + subword_in_bytes[i + 1]
                        if merge_idx == -1:
                            break
                        subword_in_bytes = subword_in_bytes[:merge_idx] + [merge_pair] + subword_in_bytes[merge_idx + 2:]
                    token_id.extend(self.vocab_to_id[i] for i in subword_in_bytes)
        return token_id

    def encode_iterable(self, iterable: Iterable[str])->Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids:list[int])-> str:
        return b''.join([self.vocab[id] for id in ids]).decode('utf-8',errors='replace')