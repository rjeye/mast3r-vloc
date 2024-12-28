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
import rerun as rr
from pathlib import Path
from natsort import natsorted
from scipy.spatial.transform import Rotation as R

from src.datasets.dataset_utils import (
    read_intrinsics,
    load_tum_poses,
)
from src.utils.tf_utils import compose_qt_tf


def log_to_rerun(data_root, ply_path, subsample_factor, depth_scale=1000):
    # Load poses
    timestamps, poses = load_tum_poses(data_root / "traj_gt.txt")
    fx, fy, cx, cy = (
        613.100554293744,
        613.903691840614,
        638.086832160361,
        378.314715743037,
    )
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    W, H = 1280, 720

    # Collect and sort RGB and depth files
    rgb_folder = data_root / "image_right"
    rgb_files = natsorted(Path(rgb_folder).glob("*.png"))

    depth_folder = data_root / "depth_right"
    depth_files = natsorted(Path(depth_folder).glob("*.png"))

    # add subsampling
    rgb_files = rgb_files[::subsample_factor]
    depth_files = depth_files[::subsample_factor]
    poses = poses[::subsample_factor]

    qvec_l2c = np.array([0.41406507, -0.6100328, 0.57049433, -0.3618651])
    tvec_l2c = np.array([0.07686256, -0.15441064, -0.1026438])
    tf_l2c = compose_qt_tf(qvec_l2c, tvec_l2c, in_xyzw=True)

    assert len(rgb_files) == len(
        depth_files
    ), f"Mismatch between RGB {len(rgb_files)} and depth frames {len(depth_files)}"
    assert len(rgb_files) == len(
        poses
    ), f"Mismatch between RGB {len(rgb_files)} and poses {len(poses)}"
    print(f"Subsampled {len(rgb_files)} frames")

    # Initialize Rerun and connect to the running Rerun TCP server
    rr.init("Trajectory Viewer", spawn=False)
    rr.connect_tcp()  # Connect to the TCP server

    rr.set_time_sequence("frame_nr", 0)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    rr.log(
        "world/xyz",
        rr.Arrows3D(
            vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
    )

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

        rr.set_time_sequence("frame_nr", i)

        tf_w2l = pose
        tf_w2c = tf_w2l @ tf_l2c

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
        rr.log(f"world/camera/image/depth", rr.DepthImage(depth, meter=depth_scale))

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Disconnecting...")
        rr.disconnect()  # Disconnect gracefully on script termination


if __name__ == "__main__":
    data_root = Path("data/TorWIC-SLAM/Jun15/Aisle_CCW_Run_1")
    # ply_path = Path(
    #     "/home/onyx/work_dirs/rjayanti/mast3r-vloc/data/TorWIC-SLAM/groundtruth_map.ply"
    # )
    subsample_factor = 20  # Set the subsample factor to 1 to log all images

    log_to_rerun(data_root, None, subsample_factor)
