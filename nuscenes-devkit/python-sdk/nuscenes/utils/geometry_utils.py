# nuScenes dev-kit.
# Code written by Oscar Beijbom and Alex Lang, 2018.

from enum import IntEnum
from typing import Tuple

import numpy as np
from pyquaternion import Quaternion


class BoxVisibility(IntEnum):
    """ Enumerates the various level of box visibility in an image """
    ALL = 0  # Requires all corners are inside the image.
    ANY = 1  # Requires at least one corner visible in the image.
    NONE = 2  # Requires no corners to be inside, i.e. box can be fully outside the image.


# def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
#     """
#     This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
#     orthographic projections. It first applies the dot product between the points and the view. By convention,
#     the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
#     normalization along the third dimension.

#     For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
#     For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
#     For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
#      all zeros) and normalize=False

#     :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
#     :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
#         The projection should be such that the corners are projected onto the first 2 axis.
#     :param normalize: Whether to normalize the remaining coordinate (along the third axis).
#     :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
#     """

#     assert view.shape[0] <= 4
#     assert view.shape[1] <= 4
#     assert points.shape[0] == 3

#     viewpad = np.eye(4)
#     viewpad[:view.shape[0], :view.shape[1]] = view

#     nbr_points = points.shape[1]

#     # Do operation in homogenous coordinates.
#     points = np.concatenate((points, np.ones((1, nbr_points))))
#     points = np.dot(viewpad, points)
#     points = points[:3, :]

#     if normalize:
#         points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

#     return points
def view_points(points: np.ndarray, view: np.ndarray, normalize: bool, dist_coeffs: np.ndarray = None) -> np.ndarray:
    if points.shape[0] != 3:
        raise ValueError("Input points should be of shape (3, n).")
    # if dist_coeffs is None:
    #     dist_coeffs = np.zeros(5)

    if dist_coeffs is not None and len(dist_coeffs) == 5:
        points = apply_fisheye_distortion(points, dist_coeffs,view)
    else:
        nbr_points = points.shape[1]
        # Apply the view transformation
        assert view.shape[0] <= 4
        assert view.shape[1] <= 4

        viewpad = np.eye(4)
        viewpad[:view.shape[0], :view.shape[1]] = view

        # Apply homogeneous coordinates transformation
        points = np.concatenate((points, np.ones((1, nbr_points))))
        points = np.dot(viewpad, points)
        points = points[:3, :]

        if normalize:
            points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points



def apply_fisheye_distortion(points: np.ndarray, dist_coeffs: np.ndarray, view: np.ndarray) -> np.ndarray:
    """
    Applies fisheye distortion correction to the projected 2D points, using a similar approach to the cam2image method.

    :param points: <np.float32: 3, n> The projected points (normalized), where each point is (x, y, z).
    :param distortion_coefficients: <np.ndarray: 9> Array of distortion coefficients [k1, k2, p1, p2, xi, gamma1, gamma2, u0, v0].
    :return: <np.float32: 3, n> Distorted points after applying fisheye correction.
    """
    
    # Extract distortion coefficients and projection parameters
    k1, k2, p1, p2, xi = dist_coeffs
    gamma1, gamma2, u0, v0 = view[0, 0], view[1, 1], view[0, 2], view[1, 2]

    # Transpose points array to (n, 3) for easier manipulation
    points = points.T

    # Calculate the norm (distance from origin) for each point
    norm = np.linalg.norm(points, axis=1)

    # Normalize x, y, z coordinates by the norm
    x = points[:, 0] / norm
    y = points[:, 1] / norm
    z = points[:, 2] / norm
    # z = points[:, 2] 

    # xi = xi -1

    # Apply xi parameter for non-linear mapping
    x /= z + xi
    y /= z + xi

    # Compute the radial distance squared (ro2 or r2)
    r2 = x * x + y * y
    # r2 = np.minimum(r2, 1.0)  # 限制最大半径距离

    alpha = 1  # 控制缩放程度的参数
    x_distorted = x * (1 + alpha * (k1 * r2 + k2 * r2**2))
    y_distorted = y * (1 + alpha * (k1 * r2 + k2 * r2**2))


    # Apply radial distortion using k1 and k2
    #x_distorted = x * (1 + k1 * r2 + k2 * r2**2)
    #y_distorted = y * (1 + k1 * r2 + k2 * r2**2)

    x_distorted += 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    y_distorted += p1 * (r2 + 2 * y * y) + 2 * p2 * x * y

    # Project the corrected coordinates back to the image plane using camera intrinsics
    x_final = gamma1 * x_distorted + u0
    y_final = gamma2 * y_distorted + v0

    # Stack u, v, and original z to form 3D points, keeping z's direction
    distorted_points = np.vstack((x_final, y_final, z))
    return distorted_points


def box_in_image(box, intrinsic: np.ndarray, imsize: Tuple[int, int], vis_level: int = BoxVisibility.ANY, dist_coeffs: np.ndarray = None) -> bool:
    """
    Check if a box is visible inside an image without accounting for occlusions.
    :param box: The box to be checked.
    :param intrinsic: <float: 3, 3>. Intrinsic camera matrix.
    :param imsize: (width, height).
    :param vis_level: One of the enumerations of <BoxVisibility>.
    :return True if visibility condition is satisfied.
    """

    corners_3d = box.corners()
    corners_img = view_points(corners_3d, intrinsic, normalize=True, dist_coeffs=dist_coeffs)[:2, :]

    visible = np.logical_and(corners_img[0, :] > 0, corners_img[0, :] < imsize[0])
    visible = np.logical_and(visible, corners_img[1, :] < imsize[1])
    visible = np.logical_and(visible, corners_img[1, :] > 0)
    visible = np.logical_and(visible, corners_3d[2, :] > 1)

    in_front = corners_3d[2, :] > 0.1  # True if a corner is at least 0.1 meter in front of the camera.

    if vis_level == BoxVisibility.ALL:
        return all(visible) and all(in_front)
    elif vis_level == BoxVisibility.ANY:
        return any(visible) and all(in_front)
    elif vis_level == BoxVisibility.NONE:
        return True
    else:
        raise ValueError("vis_level: {} not valid".format(vis_level))


def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                     rotation: Quaternion = Quaternion([1, 0, 0, 0]),
                     inverse: bool = False) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w ri rj rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)

    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))

    return tm


def points_in_box(box: 'Box', points: np.ndarray, wlh_factor: float = 1.0):
    """
    Checks whether points are inside the box.

    Picks one corner as reference (p1) and computes the vector to a target point (v).
    Then for each of the 3 axes, project v onto the axis and compare the length.
    Inspired by: https://math.stackexchange.com/a/1552579
    :param box: <Box>.
    :param points: <np.float: 3, n>.
    :param wlh_factor: Inflates or deflates the box.
    :return: <np.bool: n, >.
    """
    corners = box.corners(wlh_factor=wlh_factor)

    p1 = corners[:, 0]
    p_x = corners[:, 4]
    p_y = corners[:, 1]
    p_z = corners[:, 3]

    i = p_x - p1
    j = p_y - p1
    k = p_z - p1

    v = points - p1.reshape((-1, 1))

    iv = np.dot(i, v)
    jv = np.dot(j, v)
    kv = np.dot(k, v)

    mask_x = np.logical_and(0 <= iv, iv <= np.dot(i, i))
    mask_y = np.logical_and(0 <= jv, jv <= np.dot(j, j))
    mask_z = np.logical_and(0 <= kv, kv <= np.dot(k, k))
    mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)

    return mask
