import warnings
from copy import copy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Iterable, Tuple

import cv2
import numpy as np

from bounding_box import BoundingBox3D
from calibration import RawCalibrationOpenCV, CalibrationOpenCV
from path import PathLike


# TODO not really tested and used properly
class TextureFormat(Enum):
    L = 0
    RGB = 1
    BGR = 2
    ANY = 3


@dataclass
class InputFormat:
    texture: bool = False
    texture_format: TextureFormat = TextureFormat.L
    depth: bool = False
    black_frame: bool = False
    normals: bool = False
    point_cloud: bool = False
    calibration: bool = False


def crop_array(array, crop, overlay=None):
    array_crop = array[crop[1]:crop[3], crop[0]:crop[2]]
    if overlay is not None:
        assert overlay.dtype == bool

        array_crop = array_crop.copy()
        array_crop[overlay] = 0

    return array_crop


@dataclass
class Scene:
    # TODO use texture from color camera directly?
    image: Optional[np.ndarray] = None  # Grayscale or RGB
    depth: Optional[np.ndarray] = None
    black_frame: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    point_cloud: Optional[np.ndarray] = None
    calibration: Optional[CalibrationOpenCV] = None
    uri: Optional[str] = None  # Uniform Resource Identifier
    original_shape: Optional[Tuple[int, int]] = None

    @classmethod
    def create_empty(cls,
                     height: int,
                     width: int,
                     input_format: InputFormat = InputFormat()):
        return cls(
            image=np.empty((height, width, 1), dtype=np.uint8) if input_format.texture else None,
            depth=np.empty((height, width), dtype=np.float32) if input_format.depth else None,
            black_frame=np.empty((height, width), dtype=np.uint8) if input_format.black_frame else None,
            normals=np.empty((height, width, 3), dtype=np.float32) if input_format.normals else None,
            point_cloud=np.empty((height, width, 3), dtype=np.float32) if input_format.point_cloud else None,
            calibration=CalibrationOpenCV.create_empty() if input_format.calibration else None,
        )

    def check_scene_requirements(self, input_format: InputFormat):
        for attr in ['texture', 'depth', 'black_frame', 'normals', 'point_cloud', 'calibration']:
            scene_attr = "image" if attr == "texture" else attr
            if getattr(input_format, attr) and getattr(self, scene_attr) is None:
                raise RuntimeError(f"Scene requires '{attr}'")

    @staticmethod
    def get_camera_matrix_cropped(camera_matrix: np.ndarray, crop: np.ndarray):
        camera_matrix = camera_matrix.copy()
        top_width, top_height = crop[:2]

        camera_matrix[0, 2] -= top_width
        camera_matrix[1, 2] -= top_height

        return camera_matrix

    def crop_2d(self, input_format: InputFormat, crop: np.ndarray, mask=None):
        self.check_scene_requirements(input_format)

        new_scene = Scene()
        if input_format.texture:
            new_scene.image = crop_array(self.image, crop, mask)
        if input_format.depth:
            new_scene.depth = crop_array(self.depth, crop, mask)
        if input_format.black_frame:
            new_scene.black_frame = crop_array(self.black_frame, crop, mask)
        if input_format.normals:
            new_scene.normals = crop_array(self.normals, crop, mask)
        if input_format.point_cloud:
            new_scene.point_cloud = crop_array(self.point_cloud, crop, mask)
        if input_format.calibration:
            # https://stackoverflow.com/questions/59477398/how-does-cropping-an-image-affect-camera-calibration-intrinsic
            new_scene.calibration = copy(self.calibration)
            new_scene.calibration.camera_matrix = self.get_camera_matrix_cropped(self.calibration.camera_matrix, crop)

        return new_scene

    def _get_convex_contour(self, bounding_box_3d: BoundingBox3D):
        bbox_3d_points = bounding_box_3d.get_corners()
        image_points = self.calibration.project_to_image_plane(bbox_3d_points)
        return cv2.convexHull(image_points.astype(np.float32)).astype(np.int32)

    def get_crop_2d(self,
                    bounding_box_3d: BoundingBox3D, stride: int = 1,
                    return_inverse_mask=False,
                    return_convex_hull_2d=False):
        convex_hull = self._get_convex_contour(bounding_box_3d) // stride

        top_left = np.min(convex_hull[:, 0], axis=0)
        top_left = np.clip(top_left, 0, None)
        bottom_right = np.max(convex_hull[:, 0], axis=0)
        height, width = self.shape()[:2]
        bottom_right = np.clip(bottom_right, None, (width // stride, height // stride))
        box_2d = np.concatenate((top_left, bottom_right), axis=None).astype(np.int32)

        if not return_inverse_mask:
            if not return_convex_hull_2d:
                return box_2d
            else:
                return box_2d, convex_hull

        inverse_mask = np.ones((bottom_right - top_left)[::-1], dtype=np.uint8)
        inverse_mask = cv2.fillPoly(inverse_mask, [convex_hull - top_left], 0)

        if not return_convex_hull_2d:
            return box_2d, inverse_mask
        else:
            return box_2d, inverse_mask, convex_hull

    def crop_3d(self, input_format: InputFormat, box_3d: BoundingBox3D):
        box_2d, inverse_mask = self.get_crop_2d(box_3d, return_inverse_mask=True)
        new_scene = self.crop_2d(input_format, box_2d, inverse_mask.view(bool))

        if input_format.point_cloud:
            # TODO filter points outside BoundingBox3D
            raise NotImplementedError()

        return new_scene

    def shape(self) -> Tuple[int, int]:
        if self.image is not None:
            return self.image.shape[:2]
        if self.depth is not None:
            return self.depth.shape[:2]
        if self.black_frame is not None:
            return self.black_frame.shape[:2]
        if self.normals is not None:
            return self.normals.shape[:2]
        if self.point_cloud is not None:
            return self.point_cloud.shape[:2]
        if self.calibration is not None:
            return tuple(self.calibration.raw_calibration.actual_resolution)

        raise RuntimeError("Scene is empty")


def load_praw_extracted(image_path: PathLike, input_config: InputFormat, raw_normals=False):
    scene = Scene(
        uri=str(Path(image_path).absolute()),
        image=load_image(image_path, input_config.texture_format.name) if input_config.texture else None,
        depth=load_depth(image_path) if input_config.depth else None,
        normals=load_normals(image_path, raw_normals=raw_normals) if input_config.normals else None,
        point_cloud=load_points(image_path) if input_config.point_cloud else None,
        calibration=load_calibration(image_path) if input_config.calibration else None,
    )

    return scene


def load_image(path, image_format="BGR", check_existence=True):
    """Load image.

    Args:
        path: path to image
        image_format: one of "L", "RGB", "BGR", "ANY"
        check_existence: raise Exception if True when file does not exist

    Returns:
        np.ndarray: image of shape (H,W,C)
    """
    if not Path(path).is_file():
        if check_existence:
            raise RuntimeError(
                f"File {path} does not exist"
            )
        else:
            return None

    path = str(path)
    if image_format == "L":
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    elif image_format == "BGR":
        image = cv2.imread(path, cv2.IMREAD_COLOR)
    elif image_format == "RGB":
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image is not None else None
    elif image_format == "ANY":
        image = cv2.imread(path, cv2.IMREAD_ANYCOLOR)
    else:
        raise RuntimeError(
            f"Unsupported image format {image_format}"
        )

    if image.ndim == 2:
        image = np.expand_dims(image, -1)
    return image


def load_normals(path, check_existence=True, raw_normals=False):
    """Load normals of image.

    Args:
        path: path to image
        check_existence: raise Exception if True when file does not exist
        raw_normals: whether to load float32 normals or uint8 normals

    Returns:
        np.ndarray: image of shape (H,W,3), "BGR" color order
    """
    path = Path(path)
    if raw_normals:
        normals_path = path.parent / "normals_raw" / (path.stem + ".exr")
    else:
        normals_path = path.parent / "normals_uint8" / (path.stem + ".png")

    if not Path(normals_path).is_file():
        if check_existence:
            raise RuntimeError(
                f"File {normals_path} does not exist"
            )
        else:
            return None

    normals = cv2.imread(str(normals_path), cv2.IMREAD_UNCHANGED)
    return normals


def load_depth(path, check_existence=True):
    """Load depth of image.

    Args:
        path: path to image
        check_existence: raise Exception if True when file does not exist

    Returns:
        np.ndarray: image of shape (H,W)
    """
    path = Path(path)
    depth_path = path.parent / "depth_raw" / (path.stem + ".exr")

    if not Path(depth_path).is_file():
        if check_existence:
            raise RuntimeError(
                f"File {depth_path} does not exist"
            )
        else:
            return None

    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    return depth


def load_points(path, check_existence=True):
    path = Path(path)
    point_cloud_path = path.parent / "points_raw" / (path.stem + ".exr")

    if not Path(point_cloud_path).is_file():
        if check_existence:
            raise RuntimeError(
                f"File {point_cloud_path} does not exist"
            )
        else:
            return None

    point_cloud = cv2.imread(str(point_cloud_path), cv2.IMREAD_UNCHANGED)
    return point_cloud


def load_calibration(path, check_existence=True):
    path = Path(path)
    calibration_path = path.parent / "calibration_raw" / (path.stem + ".bin")

    if not Path(calibration_path).is_file():
        if check_existence:
            raise RuntimeError(
                f"File {calibration_path} does not exist"
            )
        else:
            return None

    with open(calibration_path, 'rb') as f:
        camera_matrix = np.fromfile(f, dtype=np.float64, count=9).reshape((3, 3))
        distortion_coefficients = np.fromfile(f, dtype=np.float64, count=14)
        transformation_camera_to_world = np.fromfile(f, dtype=np.float64, count=12).reshape((3, 4))
        actual_resolution = np.fromfile(f, dtype=np.int32, count=2)

    raw_calibration = RawCalibrationOpenCV(camera_matrix,
                                           distortion_coefficients,
                                           transformation_camera_to_world,
                                           actual_resolution)
    return CalibrationOpenCV.from_raw_calibration(raw_calibration)


def save_image(original_path, image, raise_error_if_problem=True, save_as_jpg=True, mkdir_parents=True):
    if save_as_jpg:
        path = Path(original_path).with_suffix(".jpg")
        params = (cv2.IMWRITE_JPEG_QUALITY, 85)
    else:
        path = original_path
        params = ...

    if mkdir_parents and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(path), image, params):
        message = f"Image could not be written ({path})"
        if raise_error_if_problem:
            raise RuntimeError(message)
        else:
            warnings.warn(message)


def create_or_overwrite_symlink(source_path, target_path):
    """Create or overwrite symlink without temporary deleting it.

    Avoids image viewer to treat file as deleted, so if supported,
    image viewer can continuously update image.
    """
    source_path_tmp = source_path.parent / (source_path.name + '.tmp')
    source_path_tmp.symlink_to(target_path)
    source_path_tmp.replace(source_path)


def find_files(path: PathLike, extensions: Iterable, raise_if_not_found=True, recursive=False) -> List[Path]:
    path = Path(path)

    files = []
    for extension in extensions:
        files.extend(path.rglob(f"*.{extension}") if recursive else path.glob(f"*.{extension}"))

    if raise_if_not_found and not files:
        raise RuntimeError(f"No {extensions} files found ({path})")

    files.sort()
    return files


def find_images(path: PathLike, raise_if_not_found=True, recursive=False) -> List[Path]:
    return find_files(path, ('png', 'jpg', 'jpeg'), raise_if_not_found, recursive)


def find_praws(path: PathLike, raise_if_not_found=True, recursive=False) -> List[Path]:
    return find_files(path, ('praw',), raise_if_not_found, recursive)


def merge_input_formats(input_formats: List[InputFormat]) -> InputFormat:
    input_formats = [x for x in input_formats if x is not None]

    merged = InputFormat(texture_format=TextureFormat.ANY)
    for input_format in input_formats:
        for attr in ['texture', 'depth', 'normals', 'point_cloud', 'calibration']:
            setattr(merged, attr, getattr(input_format, attr) or getattr(merged, attr))

    texture_formats = [input_format.texture_format for input_format in input_formats
                       if input_format.texture and (input_format.texture_format != TextureFormat.ANY)]
    if texture_formats:
        if not all(texture_format == texture_formats[0] for texture_format in texture_formats):
            raise RuntimeError("Incompatible texture formats")
        merged.texture_format = texture_formats[0]
    else:
        merged.texture_format = TextureFormat.ANY

    return merged


def get_merged_input_format(objects: List):
    return merge_input_formats([
        x.get_input_format() for x in objects if x is not None
    ])
