import os
from itertools import chain
from pathlib import Path
from typing import Iterable, Union

PathLike = Union[str, bytes, os.PathLike]


def clean_directory(root, remove_top_dir=False):
    for p in root.iterdir():
        if p.is_dir():
            clean_directory(p, remove_top_dir=True)
        else:
            p.unlink()

    if remove_top_dir:
        root.rmdir()


def create_dir(new_dir, create_parents=False, clean_dir=False, exist_ok=None):
    new_dir = Path(new_dir)
    exist_ok = exist_ok if exist_ok is not None else clean_dir
    new_dir.mkdir(parents=create_parents, exist_ok=exist_ok)

    if clean_dir:
        clean_directory(new_dir)


def create_dirs(dir_list, create_parents=False, clean_dir=False):
    for new_dir in dir_list:
        create_dir(new_dir, create_parents, clean_dir)


def glob_multiple_extensions(path: Path, extensions: Iterable[str]):
    def remove_period(extension):
        return extension[1:] if extension[0] == '.' else extension

    normalized_extensions = map(remove_period, extensions)

    generators = (path.glob(f'*.{extension}') for extension in normalized_extensions)
    chained_generator = chain(*generators)

    return chained_generator


def cut_path_to_closest_parent(original_path, output_type=str):
    path = Path(original_path)
    new_path = path.relative_to(path.parents[1])
    output_type_path = output_type(new_path)

    return output_type_path
