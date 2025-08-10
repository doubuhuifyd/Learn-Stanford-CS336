import regex as re
import os
from typing import BinaryIO
from collections import defaultdict
from multiprocessing import Pool
from functools import partial
import time 

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

def pre_token(index:tuple) -> tuple[dict, dict]:
    start, end =index
    input_path = r"C:\Users\A016512\Desktop\CS336\assignment1-basics-main\data\TinyStories-train.txt"
    text = ""
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    special_tokens = ["<|endoftext|>"]
    pattern = "|".join(map(re.escape, special_tokens))
    chunks = re.split(pattern, text)
    pre_token_table = defaultdict(int)
    for chunk in chunks:
        ans=re.finditer(PAT, chunk)
        for item in ans:
            pre_token_table[item.group()] += 1

    pair_table = defaultdict(int)
    for key,value in pre_token_table.items():
        for i in range(len(key) - 1):
            pair_table[key[i:i+2]] += 1
    return pre_token_table

if __name__ == "__main__":
    # now = time.time()
    # vocab_size = 1000
    # num_processes = 12
    # input_path = r"C:\Users\A016512\Desktop\CS336\assignment1_basics-main\data\TinyStories-train.txt"
    # special_tokens = ["<|endoftext|>"]

    # with open(input_path, "rb") as f:
    #     boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))

    #     chunks_index = [(start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    # with Pool(processes = num_processes) as pool:
    #     pre_token_table = pool.imap_unordered(pre_token, chunks_index)
    #     results = list(pre_token_table)
    # print(time.time() - now)
    chunks=["low", "low", "low", "low", "low", "lower", "lower", "widest", "widest", "widest", "newest", "newest", "newest", "newest", "newest", "newest"]
    frequency_table = defaultdict(int)
    pair_table = defaultdict(int)
    vocabulary_dict = {}
    vocabulary_num = 256
    for chunk in chunks:
        frequency_table[tuple(chunk.encode("utf-8"))] += 1
    for byte, num in frequency_table.items():
        for i in range(len(byte) - 1):
            pair_table[byte[i:i+2]] += num
    merge_pairs = sorted(pair_table.items(), key=lambda x: x[1], reverse=True)
    print(merge_pairs)
    print(pair_table)

