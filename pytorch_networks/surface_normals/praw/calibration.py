from dataclasses import dataclass

import cv2
import numpy as np

SCANNER_RESOLUTIONS = ((2064, 1544), (1032, 772))
SCANNER_ALPHA_RESOLUTIONS = ((1440, 1080),)
MOTION_CAM_RESOLUTIONS = ((1680, 1200), (1120, 800), (560, 800))


def invert_affine_transform(transformation: np.ndarray):
    if transformation.shape != (3, 4):
        raise RuntimeError(f"Wrong shape: got {transformation.shape}, should be (3, 4)")
    if transformation.dtype != np.float64:
        raise RuntimeError(f"Wrong dtype: got {transformation.dtype}, should be np.float64")

    rotation_inv = transformation[:3, :3].T
    translation_inv = - rotation_inv @ transformation[:, 3:]

    transform_inv = np.empty((3, 4), dtype=np.float32)
    transform_inv[:3, :3] = rotation_inv
    transform_inv[:, 3:] = translation_inv

    return transform_inv


def get_max_resolution(width, height):
    if (width, height) in MOTION_CAM_RESOLUTIONS:
        return MOTION_CAM_RESOLUTIONS[0]
    elif (width, height) in SCANNER_ALPHA_RESOLUTIONS:
        return SCANNER_ALPHA_RESOLUTIONS[0]
    elif (width, height) in SCANNER_RESOLUTIONS:
        return SCANNER_RESOLUTIONS[0]
    else:
        raise RuntimeError("Unsupported resolution")


def get_actual_camera_matrix(raw_camera_matrix: np.ndarray, width: int, height: int):
    max_width, max_height = get_max_resolution(width, height)
    camera_matrix = raw_camera_matrix.copy()

    if max_width != width:
        scale = width / max_width
        camera_matrix[0][0] *= scale
        camera_matrix[0][2] = (camera_matrix[0][2] + 0.5) * scale - 0.5
    if max_height != height:
        scale = height / max_height
        camera_matrix[1][1] *= scale
        camera_matrix[1][2] = (camera_matrix[1][2] + 0.5) * scale - 0.5

    return camera_matrix.astype(np.float32)


@dataclass
class RawCalibrationOpenCV:
    camera_matrix: np.ndarray  # np.float64, opencv 3x3 camera matrix
    distortion_coefficients: np.ndarray  # np.float64, opencv distortion coefficients 14 vector
    transformation_c2w: np.ndarray  # np.float64, camera to world affine transform 3x4 matrix
    actual_resolution: np.ndarray  # np.int32, [height, width], resolution of scan, not scanner native resolution

    @classmethod
    def create_empty(cls):
        return cls(
            camera_matrix=np.empty((3, 3), dtype=np.float64),
            distortion_coefficients=np.empty(14, dtype=np.float64),
            transformation_c2w=np.empty((3, 4), dtype=np.float64),
            actual_resolution=np.empty(2, dtype=np.int32)
        )


@dataclass
class CalibrationOpenCV:
    camera_matrix: np.ndarray  # np.float32, opencv 3x3 camera matrix
    distortion_coefficients: np.ndarray  # np.float32, opencv distortion coefficients 14 vector
    # row-major matrix convention, used for multiplying column vectors
    # [ Rxx Ryx Rzx Tx
    #   Rxy Ryy Rzy Ty
    #   Rxz Ryz Rzz Tz ]
    # more info:
    # https://math.stackexchange.com/questions/3290237/rotation-matrix-difference-between-row-vs-column-representations
    transformation_c2w: np.ndarray  # np.float32, camera to world affine transform 3x4 matrix
    transformation_w2c: np.ndarray  # np.float32, world to camera affine transform 3x4 matrix
    raw_calibration: RawCalibrationOpenCV

    @classmethod
    def create_empty(cls):
        return cls(
            camera_matrix=np.empty((3, 3), dtype=np.float32),
            distortion_coefficients=np.empty(14, dtype=np.float32),
            transformation_c2w=np.empty((3, 4), dtype=np.float32),
            transformation_w2c=np.empty((3, 4), dtype=np.float32),
            raw_calibration=RawCalibrationOpenCV.create_empty()
        )

    @classmethod
    def from_raw_calibration(cls, raw_calibration: RawCalibrationOpenCV):
        return cls(
            camera_matrix=get_actual_camera_matrix(raw_calibration.camera_matrix,
                                                   raw_calibration.actual_resolution[1],
                                                   raw_calibration.actual_resolution[0]),
            distortion_coefficients=raw_calibration.distortion_coefficients.astype(np.float32),
            transformation_c2w=raw_calibration.transformation_c2w.astype(np.float32),
            transformation_w2c=invert_affine_transform(raw_calibration.transformation_c2w),
            raw_calibration=raw_calibration
        )

    def project_to_image_plane(self, points):
        pick_points_2d, _ = cv2.projectPoints(
            points,
            self.transformation_w2c[:3, :3].astype(np.float64),  # must be float64
            self.transformation_w2c[:, 3:],
            self.camera_matrix,
            self.distortion_coefficients
        )
        return pick_points_2d
