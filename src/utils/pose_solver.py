import cv2
import numpy as np


def backproject_3d(uv, depth, K):
    """
    Backprojects 2d points given by uv coordinates into 3D using their depth values and intrinsic K
    :param uv: array [N,2]
    :param depth: array [N]
    :param K: array [3,3]
    :return: xyz: array [N,3]
    """

    uv1 = np.concatenate([uv, np.ones((uv.shape[0], 1))], axis=1)
    xyz = depth.reshape(-1, 1) * (np.linalg.inv(K) @ uv1.T).T
    return xyz


class PnPSolver:
    """Estimate relative pose (metric) using Perspective-n-Point algorithm (2D-3D) correspondences"""

    def __init__(self):
        # PnP RANSAC parameters
        self.ransac_iterations = 1000
        self.reprojection_inlier_threshold = 3
        self.confidence = 0.9999

    def estimate_pose(self, pts0, pts1, depth_0, K_common):
        # uses nearest neighbour
        pts0 = np.int32(pts0)

        if len(pts0) < 4:
            return np.full((3, 3), np.nan), np.full((3, 1), np.nan), 0

        # get depth at correspondence points
        depth_pts0 = depth_0[pts0[:, 1], pts0[:, 0]]

        # remove invalid pts (depth == 0)
        valid = depth_pts0 > depth_0.min()
        if valid.sum() < 4:
            return np.full((3, 3), np.nan), np.full((3, 1), np.nan), 0
        pts0 = pts0[valid]
        pts1 = pts1[valid].astype(np.float32)
        depth_pts0 = depth_pts0[valid]

        # backproject points to 3D in each sensors' local coordinates
        K0 = K_common
        K1 = K_common
        xyz_0 = backproject_3d(pts0, depth_pts0, K0)

        # get relative pose using PnP + RANSAC
        succ, rvec, tvec, inliers = cv2.solvePnPRansac(
            xyz_0,
            pts1,
            K1,
            None,
            iterationsCount=self.ransac_iterations,
            reprojectionError=self.reprojection_inlier_threshold,
            confidence=self.confidence,
            flags=cv2.SOLVEPNP_P3P,
        )

        # refine with iterative PnP using inliers only
        if succ and len(inliers) >= 6:
            succ, rvec, tvec, _ = cv2.solvePnPGeneric(
                xyz_0[inliers],
                pts1[inliers],
                K1,
                None,
                useExtrinsicGuess=True,
                rvec=rvec,
                tvec=tvec,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            rvec = rvec[0]
            tvec = tvec[0]

        # avoid degenerate solutions
        if succ:
            if np.linalg.norm(tvec) > 1000:
                succ = False

        if succ:
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.reshape(3, 1)
        else:
            R = np.full((3, 3), np.nan)
            t = np.full((3, 1), np.nan)
            inliers = []

        return R, t, len(inliers)
