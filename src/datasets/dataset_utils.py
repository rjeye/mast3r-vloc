from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from enum import Enum
import pickle
import cv2
import matplotlib.pyplot as plt


class PoseMode(Enum):
    MAT4x4 = "mat4x4"  # 4x4 transformation matrix
    TUM = "tum"  # timestamp tx, ty, tz, qx, qy, qz, qw
    TQ = "tq"  # tx, ty, tz, qx, qy, qz


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
    return {
        "K3x3": K,
        "width": W,
        "height": H,
    }


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

                elif return_mode == PoseMode.TUM or return_mode == PoseMode.TQ:
                    transform = [tx, ty, tz, qx, qy, qz, qw]
                poses.append(transform)
    poses = np.array(poses)

    if return_mode == PoseMode.TUM:
        # concat timestamps to poses
        poses = np.concatenate((np.array(timestamps)[:, None], poses), axis=1)
        return poses

    return timestamps, poses


def scale_matches(
    matches: np.ndarray, orig_size: tuple, resize_size: tuple
) -> np.ndarray:
    """Scale match coordinates from resized to original image dimensions."""
    scale = np.array(
        [orig_size[1] / resize_size[1], orig_size[0] / resize_size[0]]  # width scale
    )  # height scale
    return matches * scale[None, :]


def plot_mast3r_inliers(path_mast3r_inliers, view1, view2, TOP_K=15):

    # check if inliers file exists otherwise return None
    if not Path(path_mast3r_inliers).exists():
        return None

    with open(path_mast3r_inliers, "rb") as f:
        inliers = pickle.load(f)

    # resize matches_im0 and matches_im1
    matches_im0 = inliers["matches_im0"]
    matches_im1 = inliers["matches_im1"]
    H0, W0 = inliers["H0W0"]
    H1, W1 = inliers["H1W1"]

    H0_RESIZE, W0_RESIZE = inliers["H0W0_RESIZE"]
    H1_RESIZE, W1_RESIZE = inliers["H1W1_RESIZE"]
    xyz_0_inliers = inliers["xyz_0"]

    # resize 2d image matches to origina size using the scaling factor H0/H0_RESIZE and W0/W0_RESIZE
    matches_im0 = scale_matches(np.array(matches_im0), (H0, W0), (H0_RESIZE, W0_RESIZE))
    matches_im1 = scale_matches(np.array(matches_im1), (H1, W1), (H1_RESIZE, W1_RESIZE))

    # # resize view1 and view2 to resized size
    # view1 = cv2.resize(view1, (W0_RESIZE, H0_RESIZE))
    # view2 = cv2.resize(view2, (W1_RESIZE, H1_RESIZE))

    # visualize a few matches
    n_viz = TOP_K
    num_matches = matches_im0.shape[0]
    match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = (
        matches_im0[match_idx_to_viz],
        matches_im1[match_idx_to_viz],
    )

    viz_imgs = [view1, view2]

    H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
    img0 = np.pad(
        viz_imgs[0],
        ((0, max(H1 - H0, 0)), (0, 0), (0, 0)),
        "constant",
        constant_values=0,
    )
    img1 = np.pad(
        viz_imgs[1],
        ((0, max(H0 - H1, 0)), (0, 0), (0, 0)),
        "constant",
        constant_values=0,
    )
    stitched_img = np.concatenate((img0, img1), axis=1)

    # Create color array using similar colormap logic
    colors = [
        tuple(int(c * 255) for c in plt.cm.viridis(i / (TOP_K - 1))[:3])
        for i in range(TOP_K)
    ]

    # Draw matches
    for i in range(TOP_K):
        pt0 = tuple(map(int, viz_matches_im0[i]))
        pt1 = (int(viz_matches_im1[i][0] + W0), int(viz_matches_im1[i][1]))

        # Draw line
        cv2.line(stitched_img, pt0, pt1, colors[i], thickness=2)

        # Draw points
        cv2.circle(stitched_img, pt0, 2, colors[i], -1)
        cv2.circle(stitched_img, pt1, 2, colors[i], -1)

    return stitched_img
