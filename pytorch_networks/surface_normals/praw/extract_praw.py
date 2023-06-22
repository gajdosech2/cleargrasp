from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np
from omegaconf import MISSING
from tqdm import tqdm

from praw_reader import get_data_manager, get_scene
from config import get_omegaconf_config
from file_io import find_praws, Scene


@dataclass
class ExtractPrawConfig:
    input_dir: str = MISSING
    output_dir: Optional[str] = None
    rm_output_dir_if_exists: bool = False
    texture_in_output_dir: bool = True
    texture: bool = False
    texture_raw: bool = False
    texture_quantile: float = 0.99
    depth_raw: bool = False
    normals_uint8: bool = False
    normals_raw: bool = False
    points_raw: bool = False
    calibration: bool = False
    recursive: bool = False


def write_image(path: Path, image):
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    if not cv2.imwrite(str(path), image):
        RuntimeError(f"Could not write image {path}")


def save_scene(scene: Scene, raw_texture: np.ndarray, file_name: Path, config: ExtractPrawConfig):
    if config.output_dir is None:
        output_dir = file_name.parent
    else:
        if config.recursive:
            output_dir = Path(config.output_dir) / file_name.relative_to(config.input_dir).parent
        else:
            output_dir = Path(config.output_dir)

    file_name = Path(file_name.name)

    if config.texture_raw:
        texture_path = output_dir / "texture_raw" / file_name.with_suffix(".tiff")
        if raw_texture.shape[2] == 3:
            raw_texture = cv2.cvtColor(raw_texture, cv2.COLOR_RGB2BGR)
        write_image(texture_path, raw_texture)

    if config.texture:
        if config.texture_in_output_dir:
            texture_path = output_dir / file_name.with_suffix(".png")
        else:
            texture_path = output_dir / "texture" / file_name.with_suffix(".png")
        if raw_texture.shape[2] == 3:
            texture = cv2.cvtColor(scene.image, cv2.COLOR_RGB2BGR)
        else:
            texture = scene.image
        write_image(texture_path, texture)

    if config.depth_raw:
        depth_raw_path = output_dir / "depth_raw" / file_name.with_suffix(".exr")
        write_image(depth_raw_path, scene.depth)

    if config.normals_raw:
        normals_raw_path = output_dir / "normals_raw" / file_name.with_suffix(".exr")
        write_image(normals_raw_path, scene.normals)

    if config.normals_uint8:
        normals_path = output_dir / "normals_uint8" / file_name.with_suffix(".png")
        normals = cv2.convertScaleAbs(scene.normals, alpha=128, beta=np.nextafter(127.5, 128))
        write_image(normals_path, normals)

    if config.points_raw:
        point_cloud_path = output_dir / "points_raw" / file_name.with_suffix(".exr")
        write_image(point_cloud_path, scene.point_cloud)

    if config.calibration:
        calibration_path = output_dir / "calibration_raw" / file_name.with_suffix(".bin")
        if not calibration_path.parent.exists():
            calibration_path.parent.mkdir(parents=True)
        raw_calibration = scene.calibration.raw_calibration
        with open(calibration_path, "wb") as f:
            f.write(raw_calibration.camera_matrix.data)
            f.write(raw_calibration.distortion_coefficients.data)
            f.write(raw_calibration.transformation_c2w.data)
            f.write(raw_calibration.actual_resolution.data)


def extract(praw_path, config: ExtractPrawConfig):
    data_manager = get_data_manager(praw_path)
    scene, raw_texture = get_scene(data_manager, config.texture_quantile, return_raw_texture=True)
    save_scene(scene, raw_texture, praw_path, config)


def extract_praw(config: ExtractPrawConfig, update_progress_callback: Callable[[int, int], None] = None):
    praws = find_praws(config.input_dir, False, config.recursive)

    for praws_index, path in enumerate(tqdm(praws)):
        try:
            extract(path, config)
            if update_progress_callback is not None:
                update_progress_callback(praws_index, len(praws))

        except Exception as e:
            print(f"Invalid praw file {e}:", path)


def _main():
    extract_praw(get_omegaconf_config(ExtractPrawConfig))


if __name__ == "__main__":
    _main()
