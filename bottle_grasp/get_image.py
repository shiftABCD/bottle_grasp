#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import message_filters
import cv2


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        # 创建 CvBridge 对象，用于将 ROS 2 图像消息转换为 OpenCV 图像
        self.bridge = CvBridge()
        # 订阅彩色图像话题
        self.color_img_get_subscription = message_filters.Subscriber(self, Image,'/camera/camera/color/image_raw')
        # 订阅深度图像话题
        self.depth_img_get_subscription = message_filters.Subscriber(self, Image,'/camera/camera/aligned_depth_to_color/image_raw')
        # 创建时间同步器
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_img_get_subscription, self.depth_img_get_subscription],10,1)
        self.ts.registerCallback(self.get_img_callback)
        # 订阅命令话题
        self.saveimg_subscription = self.create_subscription(Bool,'/save_image_cmd',self.saveimg_callback,10)
        # 发布保存图像结果话题
        self.save_result_publisher = self.create_publisher(Bool,'/save_image_result',10)
        # 保存图像的标志
        self.save_image = False

    def get_img_callback(self,color_msg, depth_msg):
        try:
            # 将 ROS 2 图像消息转换为 OpenCV 图像
            color_image = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
            depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        except Exception as e:
            self.get_logger().error("CvBridge Error: {}".format(e))
            return

        # 显示图像
        cv2.imshow('Color Image', color_image)
        cv2.imshow('Depth Image', depth_image_normalized)
        cv2.waitKey(1)

        # 保存图像
        if self.save_image:
            cv2.imwrite("images/realsense_color_image.png", color_image)
            cv2.imwrite("realsense_depth_image.png", depth_image)
            self.get_logger().info("Color Image saved as realsense_color_image.png")
            self.get_logger().info("Depth Image saved as realsense_depth_image.png")
            msg = Bool()
            msg.data = True
            self.save_result_publisher.publish(msg)
            self.save_image = False
    def saveimg_callback(self,msg):
        if msg.data:
            self.save_image = True
            self.get_logger().info("Received Command: Save Image!")


def main(args=None):
    # 初始化 ROS 2
    rclpy.init(args=args)

    # 创建 ImageSubscriber 对象
    image_subscriber = ImageSubscriber()

    # 运行节点
    rclpy.spin(image_subscriber)

    # 销毁节点
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()