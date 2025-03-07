import cv2
import utils


image,mask = utils.image_processor("/home/hhh/learning_space/ros2_workspaces/ros2_ws/src/bottle_grasp/bottle_grasp/images/image.jpg")
# for i in range(480):
#     for j in range(640):
#         if mask[i][j] == 255:
#             image[i][j] = 0
# img = cv2.imread("/home/hhh/learning_space/ros2_workspaces/ros2_ws/src/bottle_grasp/bottle_grasp/realsense_depth_image.png",cv2.IMREAD_UNCHANGED)
# print(img)
# for i in range(480):
#     for j in range(640):
#         if img[i][j] <= 400:
#             print(img[i][j])

cv2.imshow("image",image)
cv2.waitKey(0)  # 等待用户按键
cv2.destroyAllWindows()  # 关闭所有OpenCV窗口