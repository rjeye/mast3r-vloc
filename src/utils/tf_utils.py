import numpy as np


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def compose_qt_tf(qvec, tvec, in_xyzw=False, return_Rt=False):

    if in_xyzw:
        # convert [qx, qy, qz, qw] to [qw, qx, qy, qz]
        qvec = np.array([qvec[3], qvec[0], qvec[1], qvec[2]])

    rmat = qvec2rotmat(qvec)

    tvec = np.array(tvec)

    if return_Rt:
        return rmat, tvec

    T = np.eye(4)
    T[:3, :3] = rmat
    T[:3, 3] = tvec

    return T


def decompose_tf_qt(T, in_xyzw=False):
    rmat = T[:3, :3]
    tvec = T[:3, 3]

    qvec = rotmat2qvec(rmat)

    if in_xyzw:
        # convert [qw, qx, qy, qz] to [qx, qy, qz, qw]
        qvec = np.array([qvec[1], qvec[2], qvec[3], qvec[0]])

    return qvec, tvec


def calculate_tf_error(T_source, T_target):
    R_source = T_source[:3, :3]
    R_target = T_target[:3, :3]

    t_source = T_source[:3, 3]
    t_target = T_target[:3, 3]

    R_err = calculate_rot_error(R_source, R_target)
    t_err = calculate_translation_error(t_source, t_target)

    return R_err, t_err


def calculate_rot_error(R_source, R_target):
    R_err = np.dot(np.linalg.inv(R_source), R_target)

    # calculating rotation error in degrees
    angle_err = np.arccos((np.trace(R_err) - 1) / 2)
    angle_err = np.rad2deg(angle_err)

    return angle_err


def calculate_translation_error(t_source, t_target):
    t_err = t_target - t_source
    t_err = np.linalg.norm(t_err)

    return t_err
