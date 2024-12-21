from pathlib import Path
import numpy as np
import torch


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


def read_intrinsics(path_intrinsics, resize):
    Ks = {}
    with Path(path_intrinsics).open("r") as f:
        for line in f.readlines():
            if "#" in line:
                continue

            line = line.strip().split(" ")
            img_name = line[0]
            fx, fy, cx, cy, W, H = map(float, line[1:])

            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            if resize is not None:
                K = correct_intrinsic_scale(K, resize[0] / W, resize[1] / H).numpy()
            Ks[img_name] = K
    return Ks
