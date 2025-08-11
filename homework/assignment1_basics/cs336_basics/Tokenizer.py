import regex as re
import os
from typing import BinaryIO
from collections import Counter, defaultdict
from multiprocessing import Pool
import time
import config

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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


def pre_token(parameters_dict: dict) -> tuple[list, dict, dict]:
    start, end = parameters_dict["chunks_index"]
    input_path = parameters_dict["input_path"]
    text = ""
    with open(input_path, "rb") as f:
        f.seek(start)
        text += f.read(end - start).decode("utf-8", errors="ignore")
    pattern = "|".join(map(re.escape, "<|endoftext|>"))
    chunks = re.split(pattern, text)
    frequency_table = defaultdict(int)
    chunk_list = []
    for chunk in chunks:
        ans = re.finditer(PAT, chunk)
        for item in ans:
            utf8_text = tuple(item.group().encode("utf-8"))
            frequency_table[utf8_text] += 1
            chunk_list.append(utf8_text)

    pair_table = defaultdict(int)
    for key, value in frequency_table.items():
        for i in range(len(key) - 1):
            pair_table[key[i:i + 2]] += 1
    return chunk_list, frequency_table, pair_table


def train_bpe_tokenizer(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    now = time.time()
    vocabulary_dict = {}
    vocabulary_num = 257
    for special_token in special_tokens:
        vocabulary_dict[vocabulary_num] = special_token.encode("utf-8")
        vocabulary_num += 1

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))
        chunks_index = [(start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]

    parameters_dict = {"chunks_index": chunks_index, "input_path": input_path}
    with Pool(processes=num_processes) as pool:
        chunks, frequency_table, pair_table = pool.imap_unordered(pre_token, parameters_dict)
    while vocabulary_num < vocab_size:
        merge_pairs = max(pair_table.items(), key=lambda x: (x[1], x[0]))[0]
        vocabulary_dict[vocabulary_num] = merge_pairs
        new_token_id = vocabulary_num
        vocabulary_num += 1
        l, r = merge_pairs
        del (pair_table[merge_pairs])

        new_chunk = []
        for chunk in chunks:
            i = 0
            merge_chunk = []
            while i < len(chunk):
                if i < len(chunk) - 1 and chunk[i] == l and chunk[i + 1] == r:
                    if i > 0:
                        new_pair = (chunk[i - 1], new_token_id)
                        pair_table[new_pair] += 1

                        left_pair = (chunk[i - 1], l)
                        pair_table[left_pair] -= 1
                        if pair_table[left_pair] <= 0:
                            del pair_table[left_pair]

                    if i < len(chunk) - 2:
                        new_pair = (new_token_id, chunk[i + 2])
                        pair_table[new_pair] += 1

                        right_pair = (r, chunk[i + 2])
                        pair_table[right_pair] -= 1
                        if pair_table[right_pair] <= 0:
                            del pair_table[right_pair]

                    i += 2
                    merge_chunk.append(new_token_id)
                else:
                    merge_chunk.append(chunk[i])
                    i += 1

            new_chunk.append(tuple(merge_chunk))
        chunks = new_chunk
    print(time.time() - now)


if __name__ == "__main__":
    vocab_size = 1000
    num_processes = 12
    input_path = r"D:\Learn-Stanford-CS336\homework\assignment1_basics\data\TinyStories-valid.txt"
    special_tokens = ["<|endoftext|>"]
    chunks = ["low", "low", "low", "low", "low", "lower", "lower", "widest", "widest", "widest", "newest", "newest",
              "newest", "newest", "newest", "newest"]
    chunks = [tuple(chunk.encode("utf-8")) for chunk in chunks]
    pair_table = defaultdict(int)
    frequency_table = Counter(chunks)
