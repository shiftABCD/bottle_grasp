import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import numpy as np
from torchvision.transforms import functional as F
from torchvision.io import read_image
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
import os
import cv2

#------------------------------将imgs列表或单个img显示出来---------------------------------------------
def show(imgs):
    plt.rcParams["savefig.bbox"] = 'tight'
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        masks = get_mask("images","bottle")
    


#------------------------------将目录里的图片转化为tensor格式并保存在列表里-------------------------------
def imgs2list(images_dir):
    images_list=[]
    for filename in os.listdir(images_dir):
        # 检查是否为图片
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(images_dir, filename)
            image =read_image(file_path)
            images_list.append(image)
    return images_list


#-------------------------------剔除除指定类别之外和小于指定置信度的的mask--------------------------------------------------
def get_special_mask(outputs, category_index, proba_threshold=0.9):
    category_masks_=[]
    for output in outputs:
        labels = output["labels"]
        masks = output["masks"]
        cate_mask = masks[labels==category_index]
        bool_mask = cate_mask > proba_threshold
        bool_mask = bool_mask.squeeze(1)
        category_masks_.append(bool_mask)
    return category_masks_

#----------------------------------给定图片和物体类别得到mask------------------------------------------
def get_mask(images_dir, category):
    images_list = imgs2list(images_dir)
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    transforms = weights.transforms()
    images = [transforms(d) for d in images_list]
    model = maskrcnn_resnet50_fpn(weights=weights, progress=False)
    model = model.eval()
    output = model(images)
    category_index = weights.meta["categories"].index(category)
    masks = get_special_mask(output,category_index)
    return masks

#-------------------------------------将欧拉角转换为四元数-------------------------------------------
def euler2quaterion(euler_angles):
    rotation = R.from_euler('xyz', euler_angles)
    quaternion = rotation.as_quat()
    return quaternion

#-----------------------------------将pose中的欧拉角转换为四元数-----------------------------------------
def target_quaternion_pose_trans(pose):
    pose_quaternion = euler2quaterion(pose[3:6])
    new_pose = np.concatenate((pose[:3], pose_quaternion))
    return new_pose

import numpy as np

#-------------------------------------将四元数转换为欧拉角-------------------------------------------
def quaternion2euler(quaternion):
    """
    将四元数转换为欧拉角（弧度制）
    :param quaternion: 四元数，格式为 [x, y, z, w]
    :return: 欧拉角，格式为 [roll, pitch, yaw]（弧度制）
    """
    rotation = R.from_quat(quaternion)
    euler_angles = rotation.as_euler('xyz', degrees=False)  # 弧度制
    return euler_angles

#-----------------------------------将pose中的四元数转换为欧拉角-----------------------------------------
def target_euler_pose_trans(pose):
    """
    将pose中的四元数转换为欧拉角
    :param pose: 位姿，格式为 [x, y, z, qx, qy, qz, qw]
    :return: 转换后的位姿，格式为 [x, y, z, roll, pitch, yaw]
    """
    # 提取四元数部分
    quaternion = pose[3:7]
    
    # 将四元数转换为欧拉角
    euler_angles = quaternion2euler(quaternion)
    
    # 拼接位置和欧拉角
    new_pose = np.concatenate((pose[:3], euler_angles))
    return new_pose





def image_processor(image_path, h_low=15.0, h_high=40.0, s_low=100.0, s_high=255.0, v_low=100.0, v_high=255.0,
                    min_area=1000):
    """
    图像处理函数，检测黄色区域并返回处理后的图像。

    :param image: 输入图像（BGR 格式）
    :param h_low: HSV 中 H 通道的最小值
    :param h_high: HSV 中 H 通道的最大值
    :param s_low: HSV 中 S 通道的最小值
    :param s_high: HSV 中 S 通道的最大值
    :param v_low: HSV 中 V 通道的最小值
    :param v_high: HSV 中 V 通道的最大值
    :param min_area: 最小面积阈值
    :return: 处理后的图像（包含黄色区域的矩形框和标注）
    """

    image = cv2.imread(image_path)
    # 将图像从 BGR 转换到 HSV 色彩空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义黄色的 HSV 范围
    lower_yellow = np.array([h_low, s_low, v_low])
    upper_yellow = np.array([h_high, s_high, v_high])

    # 创建黄色区域的掩码
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 寻找轮廓
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 存储所有符合条件的轮廓
    valid_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]

    # 如果存在符合条件的轮廓，选择面积最大的那个
    if valid_contours:
        max_contour = max(valid_contours, key=cv2.contourArea)
        # 创建一个最大轮廓的掩码
        max_contour_mask = np.zeros_like(yellow_mask)
        cv2.drawContours(max_contour_mask, [max_contour], -1, 255, thickness=cv2.FILLED)

        # 最大轮廓的黄色区域掩码 = 黄色掩码 & 最大轮廓掩码
        max_contour_yellow_mask = cv2.bitwise_and(yellow_mask, max_contour_mask)
        
        # 在图像上绘制矩形框
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"{w}x{h}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 在框旁边标注当前的像素大小
        text = f"Size: {w}x{h}"
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 返回处理后的图像
    return image,max_contour_yellow_mask