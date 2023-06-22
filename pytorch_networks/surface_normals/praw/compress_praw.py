import shutil
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import MISSING
from tqdm import tqdm

from praw_reader import xor_encryption
from config import get_omegaconf_config
from file_io import find_praws
from path import create_dir

CURRENT_VERSION = 1


@dataclass
class CompressPrawConfig:
    input_dir: str = MISSING
    output_dir: str = MISSING
    rm_output_dir_if_exists: bool = False
    copy_already_compressed: bool = False


def process_praw(source_praw: Path, config: CompressPrawConfig):
    output_praw = Path(config.output_dir) / source_praw.name

    with open(source_praw, 'rb') as f:
        version = np.fromfile(f, dtype=np.int32, count=1)[0]
        if version > CURRENT_VERSION:
            raise RuntimeError("New version of praw file not supported")

        initial_header = np.fromfile(f, dtype=bool, count=5)
        # use_binary_body = initial_header[0]
        compress_head = initial_header[1]
        compress_body = initial_header[2]
        encrypt_head = initial_header[3]
        encrypt_body = initial_header[4]

        if compress_head and compress_body:
            if config.copy_already_compressed:
                shutil.copy(source_praw, output_praw)
            return

        head_size = np.fromfile(f, dtype=np.uint64, count=1)[0]
        head_binary = np.fromfile(f, dtype=np.uint8, count=head_size)
        if not compress_head:
            if encrypt_head:
                xor_encryption(head_binary)

            head_binary = zlib.compress(head_binary.data, level=2)
            head_binary = np.frombuffer(head_binary, dtype='uint8').copy()

            if encrypt_head:
                xor_encryption(head_binary)

            compress_head = True
            initial_header[1] = compress_head

        body_size = np.fromfile(f, dtype=np.uint64, count=1)[0]
        body_binary = np.fromfile(f, dtype=np.uint8, count=body_size)
        if not compress_body and body_size:
            if encrypt_body:
                xor_encryption(body_binary)

            body_binary = zlib.compress(body_binary.data, level=2)
            body_binary = np.frombuffer(body_binary, dtype='uint8').copy()

            if encrypt_body:
                xor_encryption(body_binary)

            compress_body = True
            initial_header[2] = compress_body

    with open(output_praw, 'wb') as f:
        f.write(version)
        f.write(initial_header)

        head_size = np.array([head_binary.nbytes], dtype=np.uint64)
        f.write(head_size.data)
        f.write(head_binary.data)

        body_size = np.array([body_binary.nbytes], dtype=np.uint64)
        f.write(body_size.data)
        f.write(body_binary.data)


def compress_praw(config: CompressPrawConfig):
    praws = find_praws(config.input_dir)

    create_dir(config.output_dir, clean_dir=config.rm_output_dir_if_exists, exist_ok=True)
    for path in tqdm(praws):
        process_praw(path, config)


def _main():
    compress_praw(get_omegaconf_config(CompressPrawConfig))


if __name__ == "__main__":
    _main()
