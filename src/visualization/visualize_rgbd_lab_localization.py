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

from src.datasets.dataset_utils import read_intrinsics, load_tum_poses, PoseMode
from src.utils.common_viz_utils import ColorSelector, get_complement, ErrorCmap
from src.utils.tf_utils import calculate_tf_error

g_DEPTH_SCALE = 1000.0  # Divide to convert to meters
g_TRANSLATION_ERROR_THRESHOLD = 1.0  # meters


def log_trajectory(poses, label, traj_color):
    # Log the full trajectory as a line
    positions = np.array([pose[:3, 3] for pose in poses])
    rr.log(
        f"world/trajectory_{label}",
        rr.LineStrips3D(
            positions,
            colors=traj_color,
            radii=0.02,  # Adjust line thickness as needed
        ),
    )

    # Log camera positions as points
    rr.log(
        f"world/camera_positions_{label}",
        rr.Points3D(
            positions,
            colors=get_complement(traj_color),  # Complementary color
            radii=0.05,  # Adjust point size as needed
        ),
    )

    # Add start marker (green sphere)
    rr.log(
        f"world/start_marker_{label}",
        rr.Points3D(
            positions[0:1],  # Just the first position
            colors=[0, 255, 0],  # Green
            radii=0.15,  # Larger radius for visibility
            # labels=[f"START_{label}"],  # Add text label
        ),
    )

    # Add end marker (red sphere)
    rr.log(
        f"world/end_marker_{label}",
        rr.Points3D(
            positions[-1:],  # Just the last position
            colors=[255, 0, 0],  # Red
            radii=0.15,  # Larger radius for visibility
            # labels=[f"END_{label}"],  # Add text label
        ),
    )


def log_posed_rgbd(
    rgb, depth, pose_w2c, K, W, H, frame_id, text_label=None, text_color=None
):
    if pose_w2c is not None:
        rr.log(
            f"world/camera_{frame_id}",
            rr.Transform3D(
                translation=pose_w2c[:3, 3],
                mat3x3=pose_w2c[:3, :3],
            ),
        )

        rr.log(
            f"world/camera_{frame_id}/image",
            rr.Pinhole(
                resolution=[W, H],
                focal_length=[K[0, 0], K[1, 1]],
                principal_point=[K[0, 2], K[1, 2]],
            ),
        )

    if rgb is not None:
        rr.log(
            f"world/camera_{frame_id}/image/rgb",
            rr.Image(rgb, color_model="BGR").compress(jpeg_quality=95),
        )

    if depth is not None:
        rr.log(
            f"world/camera_{frame_id}/image/depth",
            rr.DepthImage(depth, meter=g_DEPTH_SCALE),
        )

    if text_label is not None:
        # log error of the predicted pose as a miniscule point at the camera position
        rr.log(
            f"world/camera_{frame_id}/label",
            rr.Points3D(
                [0, -2.0, 0],
                colors=text_color,
                radii=0.001,
                labels=[text_label],
            ),
        )


def log_to_rerun(ref_data_root, query_data_root, exp_root, exp_name):
    K, W, H = read_intrinsics(query_data_root / "intrinsics.txt")

    # Collect and sort RGB and depth files
    ref_rgb_files = natsorted(Path(ref_data_root / "rgb").glob("*.png"))
    ref_depth_files = natsorted(Path(ref_data_root / "aligned_depth").glob("*.png"))
    query_rgb_files = natsorted(Path(query_data_root / "rgb").glob("*.png"))
    query_depth_files = natsorted(Path(query_data_root / "aligned_depth").glob("*.png"))

    ref_indices, ref_poses = load_tum_poses(
        exp_root / "retrieved_ref_poses_tum.txt", PoseMode.MAT4x4
    )
    query_indices, query_gt_poses = load_tum_poses(
        exp_root / "filtered_query_poses_tum.txt", PoseMode.MAT4x4
    )
    _, query_pred_poses = load_tum_poses(
        exp_root / "pred_poses_tum_query_down.txt", PoseMode.MAT4x4
    )

    color_selector = ColorSelector()
    error_cmap = ErrorCmap(g_TRANSLATION_ERROR_THRESHOLD)

    # Initialize Rerun and connect to the running Rerun TCP server
    rr.init(f"Localization Viewer {exp_name}", spawn=False)
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

    # log_trajectory(ref_poses, "ref", color_selector.get_color("royalblue"))
    log_trajectory(query_gt_poses, "query_gt", color_selector.get_color("orange"))
    log_trajectory(query_pred_poses, "query_pred", color_selector.get_color("green"))

    for i, (
        ref_index,
        query_index,
        ref_pose,
        query_gt_pose,
        query_pred_pose,
    ) in enumerate(
        zip(ref_indices, query_indices, ref_poses, query_gt_poses, query_pred_poses)
    ):
        rr.set_time_sequence("frame_nr", i)
        # Load RGB and depth images
        ref_rgb = cv2.imread(str(ref_rgb_files[int(ref_index)]), cv2.IMREAD_COLOR)
        # ref_depth = cv2.imread(
        #     str(ref_depth_files[int(ref_index)]), cv2.IMREAD_UNCHANGED
        # )
        query_rgb = cv2.imread(str(query_rgb_files[int(query_index)]), cv2.IMREAD_COLOR)
        # query_depth = cv2.imread(
        #     str(query_depth_files[int(ref_index)]), cv2.IMREAD_UNCHANGED
        # )

        tf_ref_w2c = ref_pose
        tf_query_gt_w2c = query_gt_pose
        tf_query_pred_w2c = query_pred_pose

        rotation_error, translation_error = calculate_tf_error(
            tf_query_gt_w2c, tf_query_pred_w2c
        )
        rotation_delta, _ = calculate_tf_error(tf_ref_w2c, tf_query_gt_w2c)
        error_color = error_cmap.get_error_color(translation_error)

        log_posed_rgbd(ref_rgb, None, None, K, W, H, "ref")
        log_posed_rgbd(query_rgb, None, tf_query_gt_w2c, K, W, H, "query_gt")
        log_posed_rgbd(
            None,
            None,
            tf_query_pred_w2c,
            K,
            W,
            H,
            "query_pred",
            text_label=f"{translation_error:.2f}m, {rotation_error:.2f}°",
            text_color=error_color,
        )

        rr.log(
            f"rotation_error",
            rr.Scalar(
                rotation_error,
            ),
        )
        rr.log(
            f"rotation_delta",
            rr.Scalar(
                rotation_delta,
            ),
        )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Disconnecting...")
        rr.disconnect()  # Disconnect gracefully on script termination


if __name__ == "__main__":
    ref_data_root = Path(
        "data/rrc-lab-data/wheelchair-runs-20241220/run-1-wheelchair-mapping"
    )
    query_data_root = Path(
        "data/rrc-lab-data/wheelchair-runs-20241220/run-2-wheelchair-query"
    )
    EXP_NAME = "run-2-query-min-r-45-t-2"
    exp_root = Path("results/mast3rvloc-rrclab/") / EXP_NAME

    log_to_rerun(ref_data_root, query_data_root, exp_root, EXP_NAME)
