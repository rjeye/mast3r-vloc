import numpy as np
from scipy.spatial.transform import Rotation
from typing import Union


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion back to a rotation matrix.

    Parameters:
    -----------
    qvec : np.ndarray
        Quaternion in [w, x, y, z] format

    Returns:
    --------
    np.ndarray
        3x3 rotation matrix
    """
    # Convert from [w, x, y, z] to scipy's [x, y, z, w] format
    quat_scipy = np.roll(qvec, -1)
    rot = Rotation.from_quat(quat_scipy)
    return rot.as_matrix()


def rotmat2qvec(R: Union[np.ndarray, list]) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a quaternion using scipy's robust implementation.

    Parameters:
    -----------
    R : Union[np.ndarray, list]
        3x3 rotation matrix. Can be provided as a numpy array or nested list.

    Returns:
    --------
    np.ndarray
        Quaternion as [w, x, y, z]

    Raises:
    -------
    ValueError
        If input matrix is not 3x3 or is not a valid rotation matrix

    Examples:
    --------
    >>> R = np.eye(3)  # Identity rotation matrix
    >>> q = rotation_matrix_to_quaternion(R)
    >>> print(q)  # Should print approximately [1, 0, 0, 0]
    """
    # Convert input to numpy array if needed
    R = np.asarray(R, dtype=np.float64)

    # Input validation
    if R.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got shape {R.shape}")

    # Check if it's a valid rotation matrix
    is_orthogonal = np.allclose(R @ R.T, np.eye(3), rtol=1e-05, atol=1e-08)
    det_is_one = np.allclose(np.linalg.det(R), 1.0, rtol=1e-05, atol=1e-08)

    if not (is_orthogonal and det_is_one):
        raise ValueError(f"Input is not a valid rotation matrix | R3x3={R}")

    # Convert to quaternion using scipy
    rot = Rotation.from_matrix(R)
    qvec = rot.as_quat()  # returns [x, y, z, w]
    return np.roll(qvec, 1)  # convert to [w, x, y, z] format


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
