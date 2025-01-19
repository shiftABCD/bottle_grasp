import cv2
import numpy as np
from numpy import ndarray
from typing import Tuple
from scipy.spatial.transform import Rotation as R


def compute_angle_with_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_rect = None
    max_area = 0
    for contour in contours:
        center, (w, h), angle = cv2.minAreaRect(contour)
        area = w * h
        if area > max_area:
            max_area = area
            min_rect = center, (w, h), angle
    center, (width, height), angle = min_rect
    if width > height:
        angle = -(90 - angle)
    return angle, center


def convert(x, y, z, x1, y1, z1, rx, ry, rz, rotation_matrix, translation_vector):
    rotation_matrix = rotation_matrix
    translation_vector = translation_vector
    obj_camera_coordinates = np.array([x, y, z])
    end_effector_pose = np.array([x1, y1, z1, rx, ry, rz])
    T_camera_to_end_effector = np.eye(4)
    T_camera_to_end_effector[:3, :3] = rotation_matrix
    T_camera_to_end_effector[:3, 3] = translation_vector
    position = end_effector_pose[:3]
    orientation = R.from_euler("xyz", end_effector_pose[3:], degrees=False).as_matrix()
    T_base_to_end_effector = np.eye(4)
    T_base_to_end_effector[:3, :3] = orientation
    T_base_to_end_effector[:3, 3] = position
    obj_camera_coordinates_homo = np.append(obj_camera_coordinates, [1])
    obj_end_effector_coordinates_homo = T_camera_to_end_effector.dot(obj_camera_coordinates_homo)
    obj_base_coordinates_homo = T_base_to_end_effector.dot(obj_end_effector_coordinates_homo)
    obj_base_coordinates = obj_base_coordinates_homo[:3]
    obj_orientation_matrix = T_base_to_end_effector[:3, :3].dot(rotation_matrix)
    obj_orientation_euler = R.from_matrix(obj_orientation_matrix).as_euler("xyz", degrees=False)
    obj_base_pose = np.hstack((obj_base_coordinates, obj_orientation_euler))
    obj_base_pose[3:] = rx, ry, rz
    return obj_base_pose


def euler_angles_to_rotation_matrix(rx, ry, rz):
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return R


def pose_to_homogeneous_matrix(pose):
    x, y, z, rx, ry, rz = pose
    R = euler_angles_to_rotation_matrix(rx, ry, rz)
    t = np.array([x, y, z]).reshape(3, 1)
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t[:, 0]
    return H

#-------根据物体和基座的其次变换矩阵 求得 物体z轴 0 0 num所在位置对应基座标系的位姿y轴补偿6cm----------
def chage_pose(pose, num):
    matrix = pose_to_homogeneous_matrix(pose)
    obj_init = np.array([0, 0, num])
    obj_init = np.append(obj_init, [1])
    obj_base_init = matrix.dot(obj_init)
    return [i for i in obj_base_init[:3]] + pose[3:]


def vertical_catch_main(
        mask: ndarray,
        depth_frame: ndarray,
        color_intr: dict,
        current_pose: list,
        arm_gripper_length: float,
        vertical_rx_ry_rz: list,
        rotation_matrix: list,
        translation_vector: list,
        use_point_depth_or_mean: bool = True,
) -> Tuple[list, list, list]:
    """
    :param center:  抓取的中心点位
    :param mask:    抓取物体的轮廓信息
    :param depth_frame:     物体的深度值信息
    :param color_intr:      相机的内参
    :param current_pose:    当前的位姿信息
    :param arm_gripper_length:      夹爪的长度
    :param vertical_rx_ry_rz:       正确的夹爪偏移角度
    :param rotation_matrix:         手眼标定的旋转矩阵
    :param translation_vector:      手眼标定的平移矩阵
    :param use_point_depth_or_mean:     使用一个点位的深度信息还是整个物体的平均深度

    :return:
    above_object_pose：      垂直抓取物体上方的位姿
    correct_angle_pose：     垂直抓取物体正确的角度位姿
    finally_pose：           垂直抓取最终下爪的抓取位姿
    """
    _, center = compute_angle_with_mask(mask)
    real_x, real_y = center[0], center[1]
    if not use_point_depth_or_mean:
        dis = depth_frame[real_y][real_x]
    else:
        depth_mask = depth_frame[mask == 255]
        non_zero_values = depth_mask[depth_mask != 0]
        sorted_values = np.sort(non_zero_values)
        top_20_percent_index = int(0.2 * len(sorted_values))
        top_20_percent_values = sorted_values[:top_20_percent_index]
        dis = np.mean(top_20_percent_values)
    x = int(dis * (real_x - color_intr["ppx"]) / color_intr["fx"])
    y = int(dis * (real_y - color_intr["ppy"]) / color_intr["fy"])
    dis = int(dis)
    x, y, z = (x) * 0.001, (y) * 0.001, (dis) * 0.001
    obj_pose = convert(x, y, z, *current_pose, rotation_matrix, translation_vector)
    obj_pose = [i for i in obj_pose]
    _z = min(obj_pose[2] * 0.8 + 0.10, 0.1 + 0.03)
    obj_pose[2] = obj_pose.copy()[2] + 0.10 + arm_gripper_length * 0.001
    obj_pose[3:] = vertical_rx_ry_rz
    above_object_pose = obj_pose.copy()
    _angle = obj_pose[5] - vertical_rx_ry_rz[2]
    angle_joint, _ = compute_angle_with_mask(mask)
    angle = (angle_joint / 180) * 3.14 - _angle
    catch_pose = obj_pose.copy()
    if obj_pose[5] - angle > 0:
        catch_pose[5] = obj_pose[5] - angle
    else:
        catch_pose[5] = obj_pose[5] - angle
    correct_angle_pose = catch_pose.copy()
    finally_pose = chage_pose(list(catch_pose), _z)
    finally_pose = finally_pose.copy()
    return above_object_pose, correct_angle_pose, finally_pose

