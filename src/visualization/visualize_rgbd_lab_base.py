"""
Ensure you're running the Rerun Viewer and the Rerun TCP server before running this script.
This can be done by running the following command in a separate terminal on your remote machine:
        rerun --serve
Set up port forwarding by running the following command on your local machine:
        ssh -L 9876:localhost:9876 -L 9877:localhost:9877 -L 9090:localhost:9090 -N user@remote_server
Once the ports are forwarded, open the following URL in your browser to access the web viewer:
        http://localhost:9090?url=ws://localhost:9877
Now run this script to log images to the Rerun Viewer.
"""

import os
import time
import numpy as np
import cv2
import argparse
import rerun as rr
from pathlib import Path
from natsort import natsorted
from scipy.spatial.transform import Rotation as R

from src.datasets.dataset_utils import (
    read_intrinsics,
    load_tum_poses,
    backproject_depth,
    PoseMode,
)
from src.utils.tf_utils import compose_qt_tf

g_DEPTH_SCALE = 1000.0  # Divide to convert to meters


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch_local", action="store_true")

    return parser.parse_args()


def log_to_rerun(
    data_root, exp_name, subsample_factor, blueprint_path, launch_local=False
):
    # Load poses
    timestamps, poses = load_tum_poses(
        data_root / "poses_camera_tum.txt", PoseMode.MAT4x4
    )
    intrinsics_dict = read_intrinsics(data_root / "intrinsics.txt")
    K, W, H = (
        intrinsics_dict["K3x3"],
        intrinsics_dict["width"],
        intrinsics_dict["height"],
    )

    # Collect and sort RGB and depth files
    rgb_folder = data_root / "rgb"
    rgb_files = natsorted(Path(rgb_folder).glob("*.png"))

    depth_folder = data_root / "aligned_depth"
    depth_files = natsorted(Path(depth_folder).glob("*.png"))

    # add subsampling
    rgb_files = rgb_files[::subsample_factor]
    depth_files = depth_files[::subsample_factor]
    poses = poses[::subsample_factor]

    assert len(rgb_files) == len(
        depth_files
    ), f"Mismatch between RGB {len(rgb_files)} and depth frames {len(depth_files)}"
    assert len(rgb_files) == len(
        poses
    ), f"Mismatch between RGB {len(rgb_files)} and poses {len(poses)}"
    print(f"Subsampled {len(rgb_files)} frames")

    if launch_local:
        rr.init(f"Trajectory Viewer {exp_name}", spawn=True)
    else:
        # By Default initialize Rerun and connect to the running Rerun TCP server
        rr.init(f"Trajectory Viewer {exp_name}", spawn=False)
        rr.connect_tcp()  # Connect to the TCP server

    if blueprint_path is not None:
        rr.log_file_from_path(blueprint_path)

    rr.set_time_sequence("frame_nr", 0)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    rr.log(
        "world/xyz",
        rr.Arrows3D(
            vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
    )

    # Log the full trajectory as a line
    positions = np.array([pose[:3, 3] for pose in poses])
    rr.log(
        "world/trajectory",
        rr.LineStrips3D(
            positions,
            colors=[65, 105, 225],
            radii=0.02,  # Adjust line thickness as needed
        ),
    )

    # Log camera positions as points
    rr.log(
        "world/camera_positions",
        rr.Points3D(
            positions,
            colors=[190, 150, 30],  # Red color for camera positions
            radii=0.05,  # Adjust point size as needed
        ),
    )

    for i, (rgb_file, depth_file, pose) in enumerate(
        zip(rgb_files, depth_files, poses)
    ):
        # Load RGB and depth images
        rgb = cv2.imread(str(rgb_file))
        depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED).astype(np.float32)

        # remove the background
        depth[depth == 0] = np.nan

        rr.set_time_sequence("frame_nr", i)

        tf_w2c = pose

        rr.log(
            f"world/camera",
            rr.Transform3D(
                translation=tf_w2c[:3, 3],
                mat3x3=tf_w2c[:3, :3],
            ),
        )

        rr.log(
            f"world/camera/image",
            rr.Pinhole(
                resolution=[W, H],
                focal_length=[K[0, 0], K[1, 1]],
                principal_point=[K[0, 2], K[1, 2]],
            ),
        )

        rr.log(
            f"world/camera/image/rgb",
            rr.Image(rgb, color_model="BGR").compress(jpeg_quality=95),
        )
        rr.log(f"world/camera/image/depth", rr.DepthImage(depth, meter=g_DEPTH_SCALE))

    print("Disconnecting...")
    rr.disconnect()  # Disconnect gracefully on script termination


if __name__ == "__main__":
    args = parse_args()

    EXP_NAME = "run-3-wheelchair-query"
    data_root = Path("data/rrc-lab-data/wheelchair-runs-20241220/") / EXP_NAME
    subsample_factor = 30  # Set the subsample factor to 1 to log all images
    blueprint_path = Path("results/mast3rvloc-rrclab/trajectory-viewer.rbl")
    log_to_rerun(
        data_root,
        EXP_NAME,
        subsample_factor,
        blueprint_path,
        launch_local=args.launch_local,
    )
