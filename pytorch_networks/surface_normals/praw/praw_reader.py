import warnings
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from calibration import RawCalibrationOpenCV, CalibrationOpenCV
from file_io import Scene

CURRENT_VERSION = 1
PASSWORD = np.frombuffer(b"CdtQywwV68P0AI4YFoy1hip3Cgze5BmSk4V5FDAASgXcbif", dtype='uint8')


def _get_child(node, tag):
    for child in node:
        if child.tag == tag:
            return child

    raise RuntimeError(f"tag {tag} not found")


def _get_node_by_name(nodes_element, name):
    for child in nodes_element:
        if _get_child(child, "NodeName").text in (name, f'"{name}"'):
            return _get_child(child, "NodeData")

    raise RuntimeError(f"name {name} not found")


def _get_leaf_by_name(nodes_element, name):
    for child in nodes_element:
        if _get_child(child, "LeafName").text in (name, f'"{name}"'):
            return _get_child(child, "LeafData")

    raise RuntimeError(f"name {name} not found")


def _find_node(node_data, address):
    nodes_element = _get_child(node_data, "Nodes")
    return _get_node_by_name(nodes_element, address)


def _find_leaf(node_data, address):
    nodes_element = _get_child(node_data, "Leafs")
    return _get_leaf_by_name(nodes_element, address)


class DataManager:
    DATA_TYPE = {
        'u': (np.uint8, 1),
        '"2u"': (np.uint8, 2),
        'w': (np.uint16, 1),
        '"3w"': (np.uint16, 3),
        'f': (np.float32, 1),
        '"2f"': (np.float32, 2),
        '"3f"': (np.float32, 3),
        'd': (np.float64, 1),
    }

    def __init__(self, xml_data, blobs):
        self.root_node_data = ET.fromstring(xml_data)[0]
        self.parent_map = {c: p for p in self.root_node_data.iter() for c in p}
        self.blobs = blobs

    def _get_element_by_leaf_data_address(self, leaf_data_address: int):
        for elem in self.root_node_data.iter():
            if (elem.tag == "LeafDataAddress") and (int(elem.text) == leaf_data_address):
                leaf_data = _get_child(self.parent_map[elem], "LeafData")
                if leaf_data.text == "Link":
                    continue
                return leaf_data

    def _get_value(self, leaf_data):
        if leaf_data.text == "Link":
            leaf_data_address = int(_get_child(self.parent_map[leaf_data], "LeafDataAddress").text)
            leaf_data = self._get_element_by_leaf_data_address(leaf_data_address)

        storage_type = _get_child(leaf_data, "StorageType").text
        if storage_type == '"AHBox<int>"':
            return int(_get_child(_get_child(leaf_data, "Value"), "Value").text)
        if storage_type == '"AHBox<double>"':
            return float(_get_child(_get_child(leaf_data, "Value"), "Value").text)
        if storage_type == '"AHBox<string>"':
            return _get_child(_get_child(leaf_data, "Value"), "Value").text
        if storage_type == "Mat":
            value_node = _get_child(leaf_data, "Value")
            rows = int(_get_child(value_node, "rows").text)
            cols = int(_get_child(value_node, "cols").text)
            data_node = _get_child(value_node, "data")
            if not data_node.text:
                data_type, channels = self.DATA_TYPE[_get_child(value_node, "dt").text]
                data = self.blobs[int(_get_child(leaf_data, "BinaryDataInfoAddress").text)]
                return data.view(dtype=data_type).reshape(rows, cols, channels)
            else:
                data_type, channels = self.DATA_TYPE[_get_child(value_node, "dt").text]
                data = np.fromstring(data_node.text, dtype=data_type, sep=" ")
                if channels > 1:
                    raise NotImplementedError()
                return data.reshape(rows, cols)

        raise NotImplementedError("Access type handler not implemented")

    def _get_leaf_data(self, path):
        path = Path(path)

        node_data = self.root_node_data
        for address in path.parts[:-1]:
            node_data = _find_node(node_data, address)

        return _find_leaf(node_data, path.name)

    def get_data_member_value(self, path):
        leaf_data = self._get_leaf_data(path)
        return self._get_value(leaf_data)

    def get_data_member_value_vector(self, path, length):
        data = np.empty(length, dtype=np.float64)

        for i in range(length):
            data[i] = self.get_data_member_value(f"{path}{i}.AHBox<double>")

        return data


def xor_encryption(array: np.ndarray):
    tail_length = len(array) % len(PASSWORD)
    if tail_length:
        array_main = array[:-tail_length].reshape(-1, len(PASSWORD))
        np.bitwise_xor(array_main, PASSWORD[None, :], out=array_main)
        array_tail = array[-tail_length:]
        np.bitwise_xor(array_tail, PASSWORD[:tail_length], out=array_tail)
    else:
        array_main = array.reshape(-1, len(PASSWORD))
        np.bitwise_xor(array_main, PASSWORD[None, :], out=array_main)


def get_data_manager(praw_file) -> DataManager:
    with open(praw_file, 'rb') as f:
        version = np.fromfile(f, dtype=np.int32, count=1)[0]
        if version > CURRENT_VERSION:
            raise RuntimeError("New version of praw file not supported")

        initial_header = np.fromfile(f, dtype=bool, count=5)
        use_binary_body = initial_header[0]
        compress_head = initial_header[1]
        compress_body = initial_header[2]
        encrypt_head = initial_header[3]
        encrypt_body = initial_header[4]

        head_size = np.fromfile(f, dtype=np.uint64, count=1)[0]
        head_binary = np.fromfile(f, dtype=np.uint8, count=head_size)
        if encrypt_head:
            xor_encryption(head_binary)
        if compress_head:
            head_binary = zlib.decompress(head_binary.data)
        else:
            head_binary = head_binary.tobytes()

        body_size = np.fromfile(f, dtype=np.uint64, count=1)[0]
        body_binary = np.fromfile(f, dtype=np.uint8, count=body_size)
        if encrypt_body and body_size:
            xor_encryption(body_binary)
        if compress_body and body_size:
            body_binary = zlib.decompress(body_binary.data)

        if use_binary_body:
            items_count = np.frombuffer(body_binary, dtype=np.uint32, count=1)[0]
            offset = 4

            items_sizes = np.frombuffer(body_binary, dtype=np.uint32, count=items_count, offset=offset)
            offset += 4 * items_count

            blobs = []
            for item_size in items_sizes:
                blobs.append(np.frombuffer(body_binary, dtype=np.uint8, count=item_size, offset=offset))
                offset += item_size
        else:
            blobs = None

        return DataManager(head_binary.decode("utf-8"), blobs)


def get_scaled_texture(texture: np.ndarray, quantile: float):
    if texture.shape[0] > 1000:
        stride = 3
    else:
        stride = 2

    pixels_to_fetch = round(texture.size / (stride ** 2) * quantile)
    counts = np.bincount(texture[::stride, ::stride].flatten(), minlength=np.iinfo(np.uint16).max)
    quantile_value = np.argmax(np.cumsum(counts) >= pixels_to_fetch)

    scaled_texture = cv2.convertScaleAbs(texture, alpha=255 / quantile_value)
    if scaled_texture.ndim == 2:
        scaled_texture = np.expand_dims(scaled_texture, -1)

    return scaled_texture


def get_calibration(data_manager: DataManager, shape):
    camera_matrix = data_manager.get_data_member_value_vector(
        "Global/Cameras/0/Calibration/CameraMatrix/", 9).reshape(3, 3)
    distortion_coefficients = data_manager.get_data_member_value_vector(
        "Global/Cameras/0/Calibration/DistortionCoefficients/", 14)
    transformation_camera_to_world = data_manager.get_data_member_value(
        "User/Coordinates/CoordinateTransformation.Mat").reshape(3, 4)
    height, width = shape[:2]
    actual_resolution = np.array([height, width], dtype=np.int32)

    raw_calibration = RawCalibrationOpenCV(camera_matrix,
                                           distortion_coefficients,
                                           transformation_camera_to_world,
                                           actual_resolution)
    return CalibrationOpenCV.from_raw_calibration(raw_calibration)


def check_compatibility(data_manager: DataManager):
    try:
        fw_version: str = data_manager.get_data_member_value("Global/SystemInfo/FirmwareVersion.AHBox<string>")
    except RuntimeError:
        # in case of color texture, praw has BGR image, not RGB
        warnings.warn("Praw not supported [device with too old firmware or non production model]")
    else:
        version = fw_version.split(".")
        if int(version[1]) == 3:
            # does not have leaf "OperationMode.AHBox<string>"
            warnings.warn("Praw not supported [non production camera]")


def get_scene(data_manager: DataManager, texture_quantile: float, return_raw_texture=False):
    texture_raw = data_manager.get_data_member_value("User/Texture.Mat")
    texture = get_scaled_texture(texture_raw, texture_quantile)
    color = data_manager.get_data_member_value("User/ColorCameraImage.Mat")
    color = get_scaled_texture(color, texture_quantile)
    black_frame = data_manager.get_data_member_value("Global/Cameras/0/RawImagesMat/Image0.Mat")
    black_frame = get_scaled_texture(black_frame, texture_quantile)
    #normals = data_manager.get_data_member_value("User/NormalMap.Mat")
    normals = None
    depth = data_manager.get_data_member_value("User/DepthMap.Mat").squeeze(2)
    points = data_manager.get_data_member_value("User/PointCloud.Mat")
    calibration = get_calibration(data_manager, points.shape)

    check_compatibility(data_manager)

    scene = Scene(image=color,
                  depth=depth,
                  black_frame = black_frame,
                  normals=normals,
                  point_cloud=points,
                  calibration=calibration)
    if return_raw_texture:
        return scene, texture_raw

    return scene


def load_praw(praw_path, texture_quantile=0.99):
    data_manager = get_data_manager(praw_path)
    scene = get_scene(data_manager, texture_quantile)
    scene.uri = str(praw_path)
    return scene


if __name__ == "__main__":
    scene = load_praw("D:/robot_chalk_datasets_part/poses_a/bubbles/chalk/0_scan_inter_ultra.praw")
    print(scene.depth.shape)
    plt.imshow(scene.depth)
    plt.show()
    print(scene.image.shape)
    plt.imshow(scene.image)
    plt.show()
    print(scene.black_frame.shape)
    plt.imshow(scene.black_frame)
    plt.show()
    print()