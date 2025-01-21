
"""
This file starts a ROS node to run DOPE, 
listening to an image topic and publishing poses.
"""

from __future__ import print_function

import cv2
import message_filters
import numpy as np
import resource_retriever
import rclpy
from rclpy.node import Node
import tf2_py
import json
from PIL import Image
from PIL import ImageDraw
from cv_bridge import CvBridge
from bottle_grasp.inference.cuboid import Cuboid3d
from bottle_grasp.inference.cuboid_pnp_solver import CuboidPNPSolver
from bottle_grasp.inference.detector import ModelData, ObjectDetector
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image as ImageSensor_msg
from std_msgs.msg import String
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray


class Draw(object):
    """Drawing helper class to visualize the neural network output"""

    def __init__(self, im):
        """
        :param im: The image to draw in.
        """
        self.draw = ImageDraw.Draw(im)

    def draw_line(self, point1, point2, line_color, line_width=2):
        """Draws line on image"""
        if point1 is not None and point2 is not None:
            self.draw.line([point1, point2], fill=line_color, width=line_width)

    def draw_dot(self, point, point_color, point_radius):
        """Draws dot (filled circle) on image"""
        if point is not None:
            xy = [
                point[0] - point_radius,
                point[1] - point_radius,
                point[0] + point_radius,
                point[1] + point_radius
            ]
            self.draw.ellipse(xy,
                              fill=point_color,
                              outline=point_color
                              )

    def draw_cube(self, points, color=(255, 0, 0)):
        """
        Draws cube with a thick solid line across
        the front top edge and an X on the top face.
        """

        # draw front
        self.draw_line(points[0], points[1], color)
        self.draw_line(points[1], points[2], color)
        self.draw_line(points[3], points[2], color)
        self.draw_line(points[3], points[0], color)

        # draw back
        self.draw_line(points[4], points[5], color)
        self.draw_line(points[6], points[5], color)
        self.draw_line(points[6], points[7], color)
        self.draw_line(points[4], points[7], color)

        # draw sides
        self.draw_line(points[0], points[4], color)
        self.draw_line(points[7], points[3], color)
        self.draw_line(points[5], points[1], color)
        self.draw_line(points[2], points[6], color)

        # draw dots
        self.draw_dot(points[0], point_color=color, point_radius=4)
        self.draw_dot(points[1], point_color=color, point_radius=4)

        # draw x on the top
        self.draw_line(points[0], points[5], color)
        self.draw_line(points[1], points[4], color)


class DopeNode(Node):
    """ROS2 node that listens to image topic, runs DOPE, and publishes DOPE results"""
    def __init__(self):
        super().__init__('dope_node')
        self.pubs = {}
        self.models = {}
        self.pnp_solvers = {}
        self.pub_dimension = {}
        self.draw_colors = {}
        self.dimensions = {}
        self.class_ids = {}
        self.model_transforms = {}
        self.meshes = {}
        self.mesh_scales = {}
        self.cv_bridge = CvBridge()

        # Declare parameters
        self.declare_parameter('input_is_rectified', True)
        self.declare_parameter('downscale_height', 500)
        self.declare_parameter('thresh_angle', 0.5)
        self.declare_parameter('thresh_map', 0.01)
        self.declare_parameter('sigma', 3)
        self.declare_parameter('thresh_points', 0.1)
        self.declare_parameter('weights', '{}')
        self.declare_parameter('model_transforms', '{}') ###
        self.declare_parameter('meshes', '{}')   ###
        self.declare_parameter('mesh_scales', '{}')   ###
        self.declare_parameter('draw_colors', '{}')
        self.declare_parameter('dimensions', '{}')
        self.declare_parameter('class_ids', '{}')   ###
        self.declare_parameter('topic_publishing', 'dope')
        self.declare_parameter('topic_camera', 'camera/image_raw')
        self.declare_parameter('topic_camera_info', 'camera/camera_info')

        self.input_is_rectified = self.get_parameter('input_is_rectified').value
        self.downscale_height = self.get_parameter('downscale_height').value

        self.config_detect = lambda: None
        self.config_detect.mask_edges = 1
        self.config_detect.mask_faces = 1
        self.config_detect.vertex = 1
        self.config_detect.threshold = 0.5
        self.config_detect.softmax = 1000
        self.config_detect.thresh_angle = self.get_parameter('thresh_angle').value
        self.config_detect.thresh_map = self.get_parameter('thresh_map').value
        self.config_detect.sigma = self.get_parameter('sigma').value
        self.config_detect.thresh_points = self.get_parameter("thresh_points").value

        # For each object to detect, load network model, create PNP solver, and start ROS publisher
        weights = json.loads(self.get_parameter("weights").value)
        for model, weights_url in weights.items():
            self.models[model] = \
                ModelData(
                    model,
                    resource_retriever.get_filename(weights_url, use_protocol=False)
                )
            self.models[model].load_net_model()

            try:
                M = np.array(json.loads(self.get_parameter('model_transforms').value)[model], dtype='float64')
                self.model_transforms[model] = tf2_py.transformations.quaternion_from_matrix(M)
            except KeyError:
                self.model_transforms[model] = np.array([0.0, 0.0, 0.0, 1.0], dtype='float64')

            try:
                self.meshes[model] = json.loads(self.get_parameter('meshes').value)[model]
            except KeyError:
                pass

            try:
                self.mesh_scales[model] =json.loads(self.get_parameter('mesh_scales').value)[model]
            except KeyError:
                self.mesh_scales[model] = 1.0

            self.draw_colors[model] = tuple(json.loads(self.get_parameter('draw_colors').value)[model])
            self.dimensions[model] = tuple(json.loads(self.get_parameter('dimensions').value)[model])
            self.class_ids[model] = json.loads(self.get_parameter('class_ids').value)[model]

            self.pnp_solvers[model] = \
                CuboidPNPSolver(
                    model,
                    cuboid3d=Cuboid3d(self.dimensions[model])
                )
            self.pubs[model] = self.create_publisher(
                PoseStamped,
                f'{self.get_parameter("topic_publishing").value}/pose_{model}',
                10
            )
            self.pub_dimension[model] = \
                self.create_publisher(
                    String,
                    f'{self.get_parameter("topic_publishing").value}/dimension_{model}',
                    10
                )

        # Start ROS publishers
        self.pub_rgb_dope_points = self.create_publisher(
            ImageSensor_msg,
            f'{self.get_parameter("topic_publishing").value}/rgb_points',
            10
        )
        self.pub_detections = self.create_publisher(
            Detection3DArray,
            'detected_objects',
            10
        )
        self.pub_markers = self.create_publisher(
            MarkerArray,
            'markers',
            10
        )

        # Create subscribers for image and camera info
        image_sub = message_filters.Subscriber(
            self,
            ImageSensor_msg,
            self.get_parameter('topic_camera').value
        )
        info_sub = message_filters.Subscriber(
            self,
            CameraInfo,
            self.get_parameter('topic_camera_info').value
        )

        # Synchronize image and camera info topics
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [image_sub, info_sub],
            10,
            0.1
        )
        self.ts.registerCallback(self.image_callback)

        self.get_logger().info(f"Running DOPE...  (Listening to camera topic: {self.get_parameter('topic_camera').value})")
        self.get_logger().info("Ctrl-C to stop")

    def image_callback(self, image_msg, camera_info):
        """Image callback"""

        img = self.cv_bridge.imgmsg_to_cv2(image_msg, "rgb8")
        # cv2.imwrite('img.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # for debugging

        # Update camera matrix and distortion coefficients
        if self.input_is_rectified:
            P = np.matrix(camera_info.p, dtype='float64')
            P.resize((3, 4))
            camera_matrix = P[:, :3]
            dist_coeffs = np.zeros((4, 1))
        else:
            camera_matrix = np.matrix(camera_info.k, dtype='float64')
            camera_matrix.resize((3, 3))
            dist_coeffs = np.matrix(camera_info.d, dtype='float64')
            dist_coeffs.resize((len(camera_info.d), 1))

        # Downscale image if necessary
        height, width, _ = img.shape
        scaling_factor = float(self.downscale_height) / height
        if scaling_factor < 1.0:
            camera_matrix[:2] *= scaling_factor
            img = cv2.resize(img, (int(scaling_factor * width), int(scaling_factor * height)))

        for m in self.models:
            self.pnp_solvers[m].set_camera_intrinsic_matrix(camera_matrix)
            self.pnp_solvers[m].set_dist_coeffs(dist_coeffs)

        # Copy and draw image
        img_copy = img.copy()
        im = Image.fromarray(img_copy)
        draw = Draw(im)

        detection_array = Detection3DArray()
        detection_array.header = image_msg.header

        for m in self.models:
            # Detect object
            results = ObjectDetector.detect_object_in_image(
                self.models[m].net,
                self.pnp_solvers[m],
                img,
                self.config_detect
            )

            # Publish pose and overlay cube on image
            for _, result in enumerate(results):
                if result["location"] is None:
                    continue
                loc = result["location"]
                ori = result["quaternion"]

                # transform orientation
                transformed_ori = tf2_py.transformations.quaternion_multiply(ori, self.model_transforms[m])

                # rotate bbox dimensions if necessary
                # (this only works properly if model_transform is in 90 degree angles)
                dims = rotate_vector(vector=self.dimensions[m], quaternion=self.model_transforms[m])
                dims = np.absolute(dims)
                dims = tuple(dims)

                pose_msg = PoseStamped()
                pose_msg.header = image_msg.header
                CONVERT_SCALE_CM_TO_METERS = 100
                pose_msg.pose.position.x = loc[0] / CONVERT_SCALE_CM_TO_METERS
                pose_msg.pose.position.y = loc[1] / CONVERT_SCALE_CM_TO_METERS
                pose_msg.pose.position.z = loc[2] / CONVERT_SCALE_CM_TO_METERS
                pose_msg.pose.orientation.x = transformed_ori[0]
                pose_msg.pose.orientation.y = transformed_ori[1]
                pose_msg.pose.orientation.z = transformed_ori[2]
                pose_msg.pose.orientation.w = transformed_ori[3]

                # Publish
                self.pubs[m].publish(pose_msg)
                self.pub_dimension[m].publish(str(dims))

                # Add to Detection3DArray
                detection = Detection3D()
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.id = self.class_ids[result["name"]]
                hypothesis.score = result["score"]
                hypothesis.pose.pose = pose_msg.pose
                detection.results.append(hypothesis)
                detection.bbox.center = pose_msg.pose
                detection.bbox.size.x = dims[0] / CONVERT_SCALE_CM_TO_METERS
                detection.bbox.size.y = dims[1] / CONVERT_SCALE_CM_TO_METERS
                detection.bbox.size.z = dims[2] / CONVERT_SCALE_CM_TO_METERS
                detection_array.detections.append(detection)

                # Draw the cube
                if None not in result['projected_points']:
                    points2d = []
                    for pair in result['projected_points']:
                        points2d.append(tuple(pair))
                    draw.draw_cube(points2d, self.draw_colors[m])

        # Publish the image with results overlaid
        self.pub_rgb_dope_points.publish(
            CvBridge().cv2_to_imgmsg(
                np.array(im)[..., ::-1],
                "bgr8"
            )
        )
        self.pub_detections.publish(detection_array)
        self.publish_markers(detection_array)

    def publish_markers(self, detection_array):
        # Delete all existing markers
        markers = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        markers.markers.append(marker)
        self.pub_markers.publish(markers)

        # Object markers
        class_id_to_name = {class_id: name for name, class_id in self.class_ids.items()}
        markers = MarkerArray()
        for i, det in enumerate(detection_array.detections):
            name = class_id_to_name[det.results[0].id]
            color = self.draw_colors[name]

            # cube marker
            marker = Marker()
            marker.header = detection_array.header
            marker.action = Marker.ADD
            marker.pose = det.bbox.center
            marker.color.r = color[0] / 255.0
            marker.color.g = color[1] / 255.0
            marker.color.b = color[2] / 255.0
            marker.color.a = 0.4
            marker.ns = "bboxes"
            marker.id = i
            marker.type = Marker.CUBE
            marker.scale = det.bbox.size
            markers.markers.append(marker)

            # text marker
            marker = Marker()
            marker.header = detection_array.header
            marker.action = Marker.ADD
            marker.pose = det.bbox.center
            marker.color.r = color[0] / 255.0
            marker.color.g = color[1] / 255.0
            marker.color.b = color[2] / 255.0
            marker.color.a = 1.0
            marker.id = i
            marker.ns = "texts"
            marker.type = Marker.TEXT_VIEW_FACING
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.text = '{} ({:.2f})'.format(name, det.results[0].score)
            markers.markers.append(marker)

            # mesh marker
            try:
                marker = Marker()
                marker.header = detection_array.header
                marker.action = Marker.ADD
                marker.pose = det.bbox.center
                marker.color.r = color[0] / 255.0
                marker.color.g = color[1] / 255.0
                marker.color.b = color[2] / 255.0
                marker.color.a = 0.7
                marker.ns = "meshes"
                marker.id = i
                marker.type = Marker.MESH_RESOURCE
                marker.scale.x = self.mesh_scales[name]
                marker.scale.y = self.mesh_scales[name]
                marker.scale.z = self.mesh_scales[name]
                marker.mesh_resource = self.meshes[name]
                markers.markers.append(marker)
            except KeyError:
                # user didn't specify self.meshes[name], so don't publish marker
                pass

        self.pub_markers.publish(markers)



def rotate_vector(vector, quaternion):
    # 使用 tf2 的四元数运算
    q_conj = tf2_py.transformations.quaternion_conjugate(quaternion)
    vector = np.array(vector, dtype='float64')
    vector = np.append(vector, [0.0])
    vector = tf2_py.transformations.quaternion_multiply(q_conj, vector)
    vector = tf2_py.transformations.quaternion_multiply(vector, quaternion)
    return vector[:3]

def main():
    rclpy.init()
    node = DopeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()