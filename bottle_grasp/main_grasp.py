

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rm_ros_interfaces.msg import Movejp, Armstate ,Gripperset, Gripperpick
from std_msgs.msg import Bool, Empty
import time
import cv2
import numpy as np

from bottle_grasp.utils import get_mask, target_quaternion_pose_trans,target_euler_pose_trans,image_processor
from bottle_grasp.grasp_compute import vertical_catch_main

class GraspDemoPub(Node):
    def __init__(self):
        super().__init__("Grasp_demo_pub_node")
        
        # 初始化状态变量
        self.first_run = True
        self.movej_p_state = False
        self.saveimg_state = False
        self.get_mask_state = False
        self.above_object_state = False
        self.correct_angle_state = False
        self.close_to_object_state = False
        self.move_away_state1 = False
        self.move_away_state2 = False
        self.close_jaw_state = False
        self.open_jaw_state = False
        self.movejp1 = False
        self.movejp2 = False
        self.movejp3 = False
        self.movejp4 = False
        self.movejp5 = False
        self.movejp6 = False
        self.num = 0

        # 初始化机械臂状态
        self.current_arm_state = [0.0] * 6
        self.above_object_pose = []
        self.correct_angle_pose = []
        self.finally_pose = []

        # 初始化发布器
        self.movej_p_publisher_ = self.create_publisher(Movejp, "/rm_driver/movej_p_cmd", 10)
        self.saveimg_publisher_ = self.create_publisher(Bool, "save_image_cmd", 10)
        self.get_arm_state_publisher = self.create_publisher(Empty, "/rm_driver/get_current_arm_state_cmd", 10)
        self.close_jaw_publisher = self.create_publisher(Gripperpick, "/rm_driver/set_gripper_pick_cmd", 10)
        self.open_jaw_publisher = self.create_publisher(Gripperset, "/rm_driver/set_gripper_position_cmd", 10)

        # 初始化订阅器
        self.movej_p_subscription_ = self.create_subscription(
            Bool, "/rm_driver/movej_p_result", self.Movej_p_Callback, 10
        )
        self.arm_state_subscription = self.create_subscription(
            Armstate, "/rm_driver/get_current_arm_state_result", self.get_arm_state_callback, 10
        )
        self.saveimg_subscription = self.create_subscription(
            Bool, "/save_image_result", self.saveimg_callback, 10
        )


        # 初始化定时器
        self.loop_pub_timer = self.create_timer(0.1, self.looppub_timer_callback)

    def saveimg_callback(self,msg):
        self.saveimg_state = msg.data
        if msg.data:
            self.get_logger().info("*******Save Image succeeded\n")
        else:
            self.get_logger().error("*******Save Image Failed\n")



    def Movej_p_Callback(self,msg):
        if self.movejp1:
            self.movej_p_state = msg.data
            self.movejp1 = False
            if msg.data:
                self.get_logger().info("*******Movej_p succeeded\n")
            else:
                self.get_logger().error("*******Movej_p Failed\n")
        if self.movejp2:
            self.above_object_state = msg.data
            self.movejp2 = False
            if msg.data:
                self.get_logger().info("*******Movej_p succeeded\n")
            else:
                self.get_logger().error("*******Movej_p Failed\n")
        if self.movejp3:
            self.correct_angle_state = msg.data
            self.movejp3 = False
            if msg.data:
                self.get_logger().info("*******Movej_p succeeded\n")
            else:
                self.get_logger().error("*******Movej_p Failed\n")
        if self.movejp4:
            self.close_to_object_state = msg.data
            self.movejp4 = False
            if msg.data:
                self.get_logger().info("*******Movej_p succeeded\n")
            else:
                self.get_logger().error("*******Movej_p Failed\n")
        if self.movejp5:
            self.move_away_state1 = msg.data
            self.movejp5 = False
            if msg.data:
                self.get_logger().info("*******Movej_p succeeded\n")
            else:
                self.get_logger().error("*******Movej_p Failed\n")
        if self.movejp6:
            self.move_away_state2 = msg.data
            self.movejp6 = False
            if msg.data:
                self.get_logger().info("*******Movej_p succeeded\n")
            else:
                self.get_logger().error("*******Movej_p Failed\n")

    def get_arm_state_callback(self, msg):
        pose_quaternion = []
        pose_quaternion.append(msg.pose.position.x)
        pose_quaternion.append(msg.pose.position.y)
        pose_quaternion.append(msg.pose.position.z)
        pose_quaternion.append(msg.pose.orientation.x)
        pose_quaternion.append(msg.pose.orientation.y)
        pose_quaternion.append(msg.pose.orientation.z)
        pose_quaternion.append(msg.pose.orientation.w)
        pose = target_euler_pose_trans(pose_quaternion)
        self.current_arm_state[0] = pose[0]
        self.current_arm_state[1] = pose[1]
        self.current_arm_state[2] = pose[2]
        self.current_arm_state[3] = pose[3]
        self.current_arm_state[4] = pose[4]
        self.current_arm_state[5] = pose[5]
        self.get_logger().info("get_arm_state")



    def looppub_timer_callback(self):

        if self.first_run:  # Step1: 移动机械臂水平
            self.get_logger().info("Step1: 移动机械臂水平")
            self.movejp1 = True
            movej_p_target1 = Movejp()
            movej_p_target1.pose.position.x = -0.18
            movej_p_target1.pose.position.y = 0.0
            movej_p_target1.pose.position.z = 0.45
            movej_p_target1.pose.orientation.x = 1.0
            movej_p_target1.pose.orientation.y = 0.0
            movej_p_target1.pose.orientation.z = 0.0
            movej_p_target1.pose.orientation.w = 0.0
            movej_p_target1.speed = 10
            movej_p_target1.block = True
            self.movej_p_publisher_.publish(movej_p_target1)
            
            self.first_run = False
            time.sleep(1)


        if self.movej_p_state:  # Step2: 获取相机的一帧和当前机械臂关节状态
            self.get_logger().info("Step2: 获取相机的一帧和当前机械臂关节状态")
            time.sleep(1)
            empty_msg = Empty()
            self.get_arm_state_publisher.publish(empty_msg)
            saveimg_bool = Bool()
            saveimg_bool.data = True
            self.saveimg_publisher_.publish(saveimg_bool)
            self.movej_p_state = False
            time.sleep(1)


        if self.saveimg_state:  # Step3: 获取目标物 mask 并计算位置
            self.get_logger().info("Step3: 获取目标物 mask 并计算位置")
            if(self.num == 0):
                _,mask_array = image_processor('/home/hhh/learning_space/ros2_workspaces/ros2_ws/src/bottle_grasp/bottle_grasp/images/realsense_color_image.png')
                mask_array = np.array(mask_array)
            if(self.num == 1):
                _,mask_array = image_processor('/home/hhh/learning_space/ros2_workspaces/ros2_ws/src/bottle_grasp/bottle_grasp/images/realsense_color_image.png',h_low=35.0, h_high=85.0)
                mask_array = np.array(mask_array)
            if(self.num == 2):
                _,mask_array = image_processor('/home/hhh/learning_space/ros2_workspaces/ros2_ws/src/bottle_grasp/bottle_grasp/images/realsense_color_image.png',h_low=100, h_high=140.0)
                mask_array = np.array(mask_array)
            if(self.num == 3):
                masks = get_mask('/home/hhh/learning_space/ros2_workspaces/ros2_ws/src/bottle_grasp/bottle_grasp/images', 'bottle')
                mask_array = np.array(masks[0][0])
                mask_array = (mask_array * 255).astype(np.uint8)
            print(mask_array)
            self.get_logger().info("get_mask")
            depth_frame = cv2.imread('/home/hhh/learning_space/ros2_workspaces/ros2_ws/src/bottle_grasp/bottle_grasp/realsense_depth_image.png', cv2.IMREAD_UNCHANGED)

            # 摄像头的内参
            color_intr = {"ppx": 301.348, "ppy": 251.109, "fx": 590.209, "fy": 597.667}
            # 机械臂夹爪的长度
            arm_gripper_length = 150.0
            vertical_rx_ry_rz = [3.14, 0, 0]
            # 手眼标定结果
            rotation_matrix = [[ 0.03620715,  0.99924033, -0.01441568],
                               [-0.9991173,   0.0365026,   0.02078883],
                               [ 0.02129925,  0.01365025,  0.99967996]]
            translation_vector =[-0.10298245, 0.03366391, 0.03288961]

            # self.above_object_pose, self.correct_angle_pose, self.finally_pose = vertical_catch(
            #     mask_array, depth_frame, color_intr, self.current_arm_state, arm_gripper_length,
            #     vertical_rx_ry_rz, rotation_matrix, translation_vector, True
            # )
            self.above_object_pose, self.correct_angle_pose, self.finally_pose = vertical_catch_main(
                mask_array, depth_frame, color_intr, self.current_arm_state, arm_gripper_length,
                vertical_rx_ry_rz, rotation_matrix, translation_vector, True
            )
            print(self.above_object_pose)
            print(self.correct_angle_pose)
            print(self.finally_pose)
            self.above_object_pose = target_quaternion_pose_trans(self.above_object_pose)
            self.correct_angle_pose = target_quaternion_pose_trans(self.correct_angle_pose)
            self.finally_pose = target_quaternion_pose_trans(self.finally_pose)
            print(self.above_object_pose)
            print(self.correct_angle_pose)
            print(self.finally_pose)
            self.saveimg_state = False
            self.get_mask_state = True
            time.sleep(1)


        if self.get_mask_state:  # Step4: 机械臂移动到目标物上方
            self.get_logger().info("Step4: 机械臂移动到目标物上方")
            self.movejp2 = True
            movej_p_target2 = Movejp()
            movej_p_target2.pose.position.x = self.above_object_pose[0]
            movej_p_target2.pose.position.y = self.above_object_pose[1]
            movej_p_target2.pose.position.z = self.above_object_pose[2]
            movej_p_target2.pose.orientation.x = self.above_object_pose[3]
            movej_p_target2.pose.orientation.y = self.above_object_pose[4]
            movej_p_target2.pose.orientation.z = self.above_object_pose[5]
            movej_p_target2.pose.orientation.w = self.above_object_pose[6]
            movej_p_target2.speed = 10
            movej_p_target2.block = True
            self.movej_p_publisher_.publish(movej_p_target2)

            self.get_mask_state = False
            time.sleep(1)


        if self.above_object_state:  # Step5: 调整机械臂角度
            self.get_logger().info("Step5: 调整机械臂角度")
            self.movejp3 = True
            movej_p_target3 = Movejp()
            movej_p_target3.pose.position.x = self.correct_angle_pose[0]
            movej_p_target3.pose.position.y = self.correct_angle_pose[1]
            movej_p_target3.pose.position.z = self.correct_angle_pose[2]
            movej_p_target3.pose.orientation.x = self.correct_angle_pose[3]
            movej_p_target3.pose.orientation.y = self.correct_angle_pose[4]
            movej_p_target3.pose.orientation.z = self.correct_angle_pose[5]
            movej_p_target3.pose.orientation.w = self.correct_angle_pose[6]
            movej_p_target3.speed = 10
            movej_p_target3.block = True
            self.movej_p_publisher_.publish(movej_p_target3)
            self.above_object_state = False
            time.sleep(1)


        if self.correct_angle_state:  # Step6: 接近物体
            self.get_logger().info("Step6: 接近物体")
            self.movejp4 = True
            movej_p_target4 = Movejp()
            movej_p_target4.pose.position.x = self.finally_pose[0]
            movej_p_target4.pose.position.y = self.finally_pose[1]
            movej_p_target4.pose.position.z = self.finally_pose[2]
            movej_p_target4.pose.orientation.x = self.finally_pose[3]
            movej_p_target4.pose.orientation.y = self.finally_pose[4]
            movej_p_target4.pose.orientation.z = self.finally_pose[5]
            movej_p_target4.pose.orientation.w = self.finally_pose[6]
            movej_p_target4.speed = 10
            movej_p_target4.block = True
            self.movej_p_publisher_.publish(movej_p_target4)
            self.correct_angle_state = False
            time.sleep(1)
        
        if self.close_to_object_state:  # Step7:闭合夹爪
            self.get_logger().info("Step7: 闭合夹爪")
            self.close_jaw_msg = Gripperpick()
            self.close_jaw_msg.speed = 500
            self.close_jaw_msg.force = 500
            self.close_jaw_msg.block = True
            self.close_jaw_publisher.publish(self.close_jaw_msg)
            self.close_to_object_state = False    
            time.sleep(2) 
            self.close_jaw_state = True

        if self.close_jaw_state: # Step8: 移动到目标物体放置位置
            self.get_logger().info("Step8: 移动到目标物体位置")
            self.movejp5 = True
            movej_p_target5 = Movejp()
            movej_p_target5.pose.position.x = -0.18
            movej_p_target5.pose.position.y = 0.0
            movej_p_target5.pose.position.z = 0.45
            movej_p_target5.pose.orientation.x = 1.0
            movej_p_target5.pose.orientation.y = 0.0
            movej_p_target5.pose.orientation.z = 0.0
            movej_p_target5.pose.orientation.w = 0.0
            movej_p_target5.speed = 10
            movej_p_target5.block = True
            self.movej_p_publisher_.publish(movej_p_target5)
            self.close_jaw_state = False
            time.sleep(1)

        if self.move_away_state1: # Step9: 移动到目标物体放置位置
            self.get_logger().info("Step9: 移动到目标物体位置")
            self.movejp6 = True
            movej_p_target6 = Movejp()
            movej_p_target6.pose.position.x = -0.270
            movej_p_target6.pose.position.y = -0.315
            movej_p_target6.pose.position.z = 0.510
            movej_p_target6.pose.orientation.x = 0.730
            movej_p_target6.pose.orientation.y = 0.370
            movej_p_target6.pose.orientation.z = -0.498
            movej_p_target6.pose.orientation.w = 0.285
            movej_p_target6.speed = 10
            movej_p_target6.block = True
            self.movej_p_publisher_.publish(movej_p_target6)
            self.move_away_state1 = False
            time.sleep(1)
        
        if self.move_away_state2:  #Step10: 松开夹爪
            self.get_logger().info("Step10: 松开夹爪")
            self.open_jaw_msg = Gripperset()
            self.open_jaw_msg.position = 1000
            self.open_jaw_msg.block = True
            self.open_jaw_publisher.publish(self.open_jaw_msg)
            self.move_away_state2 = False
            time.sleep(1)
            self.open_jaw_state = True

        if self.open_jaw_state: #Step11: 完成
            self.get_logger().info("Step11: 完成")
            self.open_jaw_state = False
            self.num +=1
            time.sleep(1)
            if self.num != 4:
                self.first_run = True

def main(args=None):
    rclpy.init(args=args)

    # 创建节点
    node_pub = GraspDemoPub()
    try:
        # 运行节点
        rclpy.spin(node_pub)
    except KeyboardInterrupt:
        # 捕获 Ctrl+C 信号
        pass
    finally:
        # 销毁节点
        node_pub.destroy_node()
        # 关闭 ROS 2 上下文
        rclpy.shutdown()


if __name__ == "__main__":
    main()    