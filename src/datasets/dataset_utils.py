from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from enum import Enum


class PoseMode(Enum):
    MAT4x4 = "mat4x4"  # 4x4 transformation matrix
    TUM = "tum"  # tx, ty, tz, qx, qy, qz, qw


def correct_intrinsic_scale(K, scale_x, scale_y):
    """Given an intrinsic matrix (3x3) and two scale factors, returns the new intrinsic matrix corresponding to
    the new coordinates x' = scale_x * x; y' = scale_y * y
    Source: https://dsp.stackexchange.com/questions/6055/how-does-resizing-an-image-affect-the-intrinsic-camera-matrix
    """

    transform = torch.eye(3)
    transform[0, 0] = scale_x
    transform[0, 2] = scale_x / 2 - 0.5
    transform[1, 1] = scale_y
    transform[1, 2] = scale_y / 2 - 0.5
    Kprime = transform @ K

    return Kprime


def read_intrinsics(path_intrinsics, resize=None):
    with Path(path_intrinsics).open("r") as f:
        for line in f.readlines():
            if "#" in line:
                continue
            if len(line.strip()) == 0:
                continue

            line = line.strip().split(" ")
            img_name = line[0]
            fx, fy, cx, cy, W, H = map(float, line)

            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            if resize is not None:
                print(f"Resizing intrinsics by {resize/np.array([W, H])}")
                K = correct_intrinsic_scale(K, resize[0] / W, resize[1] / H).numpy()
    return K, W, H


def load_tum_poses(poses_file, return_mode=PoseMode.MAT4x4):
    """
    Load TUM pose format file: timestamp, tx, ty, tz, qx, qy, qz, qw.
    Returns:
        - timestamps: List of timestamps
        - poses: List of 4x4 transformation matrices (numpy arrays)
    """
    timestamps = []
    poses = []
    with open(poses_file, "r") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                timestamp = float(parts[0])
                tx, ty, tz = map(float, parts[1:4])
                qx, qy, qz, qw = map(float, parts[4:8])
                timestamps.append(timestamp)

                if return_mode == PoseMode.MAT4x4:
                    rotation_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()
                    transform = np.eye(4)
                    transform[:3, :3] = rotation_matrix
                    transform[:3, 3] = [tx, ty, tz]

                elif return_mode == PoseMode.TUM:
                    transform = [tx, ty, tz, qx, qy, qz, qw]

                poses.append(transform)
    poses = np.array(poses)
    return timestamps, poses


def backproject_depth(depth, intrinsics):
    """Convert depth map to 3D points using camera intrinsics."""
    h, w = depth.shape
    fx, fy, cx, cy = (
        intrinsics[0, 0],
        intrinsics[1, 1],
        intrinsics[0, 2],
        intrinsics[1, 2],
    )
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth / 1000.0  # Convert from millimeters to meters
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    return points[z.flatten() > 0]  # Keep only valid depth points
