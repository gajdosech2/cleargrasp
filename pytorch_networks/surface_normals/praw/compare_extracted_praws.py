from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from omegaconf import MISSING
from tqdm import tqdm

from config import get_omegaconf_config
from file_io import load_normals, load_image, load_depth, load_points, \
    load_calibration, find_praws


@dataclass
class CompareExtractedPrawsConfig:
    praws_dir: str = MISSING
    dir1: str = MISSING
    dir2: str = MISSING
    texture: bool = False
    texture_raw: bool = False
    depth_raw: bool = False
    normals_uint8: bool = False
    normals_raw: bool = False
    points_raw: bool = False
    calibration: bool = False


def compare(config: CompareExtractedPrawsConfig):
    praws = find_praws(config.praws_dir)
    for path in tqdm(praws):
        image_path1 = Path(config.dir1) / Path(path.name).with_suffix(".png")
        image_path2 = Path(config.dir2) / Path(path.name).with_suffix(".png")

        if config.texture:
            array1 = load_image(image_path1, "ANY")
            array2 = load_image(image_path2, "ANY")
            if not np.all(array1 == array2):
                print(path)
                print("texture")

        if config.texture_raw:
            path1 = Path(config.dir1) / "texture_raw" / Path(path.name).with_suffix(".tiff")
            array1 = cv2.imread(str(path1), cv2.IMREAD_UNCHANGED)
            if array1 is None:
                raise RuntimeError(f"Could not load {path1}")
            path2 = Path(config.dir2) / "texture_raw" / Path(path.name).with_suffix(".tiff")
            array2 = cv2.imread(str(path2), cv2.IMREAD_UNCHANGED)
            if array2 is None:
                raise RuntimeError(f"Could not load {path2}")
            if not np.all(array1 == array2):
                print(path)
                print("texture_raw")

        if config.depth_raw:
            array1 = load_depth(image_path1)
            array2 = load_depth(image_path2)
            if not np.all(array1 == array2):
                print(path)
                print("texture")

        if config.normals_raw:
            array1 = load_normals(image_path1, raw_normals=True)
            array2 = load_normals(image_path2, raw_normals=True)
            if not np.all(array1 == array2):
                print(path)
                print("normals_raw")

        if config.normals_uint8:
            array1 = load_normals(image_path1, raw_normals=False)
            array2 = load_normals(image_path2, raw_normals=False)
            if not np.all(array1 == array2):
                print(path)
                print("normals_uint8")

        if config.points_raw:
            array1 = load_points(image_path1)
            array2 = load_points(image_path2)
            if not np.all(array1 == array2):
                print(path)
                print("points")

        if config.calibration:
            calibration1 = load_calibration(image_path1)
            calibration2 = load_calibration(image_path2)
            if not np.all(calibration1.distortion_coefficients == calibration2.distortion_coefficients):
                print(path)
                print("distortion coefficients")
            if not np.all(calibration1.camera_matrix == calibration2.camera_matrix):
                print(path)
                print("camera matrix")
            if not np.all(calibration1.transformation_c2w == calibration2.transformation_c2w):
                print(path)
                print("camera matrix c2w")


def _main():
    compare(get_omegaconf_config(CompareExtractedPrawsConfig))


if __name__ == "__main__":
    _main()
