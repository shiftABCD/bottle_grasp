from __future__ import print_function

import cv2
import message_filters
import numpy as np
import rclpy
import json
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from PIL import Image, ImageDraw
from cv_bridge import CvBridge
from inference.cuboid import Cuboid3d
from inference.cuboid_pnp_solver import CuboidPNPSolver
from inference.detector import ModelData, ObjectDetector
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image as ImageSensor_msg
from std_msgs.msg import String
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray


class Draw:
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
            self.draw.ellipse(xy, fill=point_color, outline=point_color)

    def draw_cube(self, points, color=(255, 0, 0)):
        """Draws cube with a thick solid line across the front top edge and an X on the top face."""
        # Draw front
        self.draw_line(points[0], points[1], color)
        self.draw_line(points[1], points[2], color)
        self.draw_line(points[3], points[2], color)
        self.draw_line(points[3], points[0], color)

        # Draw back
        self.draw_line(points[4], points[5], color)
        self.draw_line(points[6], points[5], color)
        self.draw_line(points[6], points[7], color)
        self.draw_line(points[4], points[7], color)

        # Draw sides
        self.draw_line(points[0], points[4], color)
        self.draw_line(points[7], points[3], color)
        self.draw_line(points[5], points[1], color)
        self.draw_line(points[2], points[6], color)

        # Draw dots
        self.draw_dot(points[0], point_color=color, point_radius=4)
        self.draw_dot(points[1], point_color=color, point_radius=4)

        # Draw x on the top
        self.draw_line(points[0], points[5], color)
        self.draw_line(points[1], points[4], color)


class DopeNode(Node):
    """ROS 2 node that listens to image topic, runs DOPE, and publishes DOPE results"""

    def __init__(self):
        super().__init__('dope_node')

        self.pubs = {}
        self.models = {}
        self.pnp_solvers = {}
        self.draw_colors = {}
        self.dimensions = {}
        self.class_ids = {} ###
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

        # Get parameters
        self.input_is_rectified = self.get_parameter('input_is_rectified').value
        self.downscale_height = self.get_parameter('downscale_height').value
        self.config_detect = lambda: None
        self.config_detect.thresh_angle = self.get_parameter('thresh_angle').value
        self.config_detect.thresh_map = self.get_parameter('thresh_map').value
        self.config_detect.sigma = self.get_parameter('sigma').value
        self.config_detect.thresh_points = self.get_parameter('thresh_points').value

        # Load models and initialize PnP solvers
        weights = self.get_parameter('weights').value
        weights = json.loads(weights)
        for model, weights_url in weights.items():
            self.models[model] = ModelData(model, weights_url)
            self.models[model].load_net_model()

            # Initialize PnP solver
            dimensions = self.get_parameter('dimensions').value[model]
            dimensions = json.loads(dimensions)
            self.pnp_solvers[model] = CuboidPNPSolver(model, cuboid3d=Cuboid3d(dimensions))

            # Initialize publishers
            self.pubs[model] = self.create_publisher(
                PoseStamped,
                f'{self.get_parameter("topic_publishing").value}/pose_{model}',
                10
            )

        # Create publishers for visualization
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

        self.get_logger().info("Running DOPE...")

    def image_callback(self, image_msg, camera_info):
        """Image callback"""
        img = self.cv_bridge.imgmsg_to_cv2(image_msg, "rgb8")

        # Update camera matrix and distortion coefficients
        if self.input_is_rectified:
            P = np.array(camera_info.p).reshape((3, 4))
            camera_matrix = P[:3, :3]
            dist_coeffs = np.zeros((4, 1))
        else:
            camera_matrix = np.array(camera_info.k).reshape((3, 3))
            dist_coeffs = np.array(camera_info.d)

        # Downscale image if necessary
        height, width, _ = img.shape
        scaling_factor = float(self.downscale_height) / height
        if scaling_factor < 1.0:
            camera_matrix[:2] *= scaling_factor
            img = cv2.resize(img, (int(scaling_factor * width), int(scaling_factor * height)))

        # Update PnP solvers with camera parameters
        for model in self.models:
            self.pnp_solvers[model].set_camera_intrinsic_matrix(camera_matrix)
            self.pnp_solvers[model].set_dist_coeffs(dist_coeffs)

        # Copy and draw image
        img_copy = img.copy()
        im = Image.fromarray(img_copy)
        draw = Draw(im)

        detection_array = Detection3DArray()
        detection_array.header = image_msg.header

        # Detect objects and publish results
        for model in self.models:
            results = ObjectDetector.detect_object_in_image(
                self.models[model].net,
                self.pnp_solvers[model],
                img,
                self.config_detect
            )

            for result in results:
                if result["location"] is None:
                    continue

                loc = result["location"]
                ori = result["quaternion"]

                # Publish pose
                pose_msg = PoseStamped()
                pose_msg.header = image_msg.header
                pose_msg.pose.position.x = loc[0] / 100.0  # Convert cm to meters
                pose_msg.pose.position.y = loc[1] / 100.0
                pose_msg.pose.position.z = loc[2] / 100.0
                pose_msg.pose.orientation.x = ori[0]
                pose_msg.pose.orientation.y = ori[1]
                pose_msg.pose.orientation.z = ori[2]
                pose_msg.pose.orientation.w = ori[3]
                self.pubs[model].publish(pose_msg)

                # Draw cube on image
                if None not in result['projected_points']:
                    points2d = [tuple(pair) for pair in result['projected_points']]
                    draw.draw_cube(points2d, self.draw_colors[model])

        # Publish the image with results overlaid
        self.pub_rgb_dope_points.publish(
            self.cv_bridge.cv2_to_imgmsg(np.array(im)[..., ::-1], "bgr8")
        )

        # Publish detection results
        self.pub_detections.publish(detection_array)

        # Publish markers for visualization
        self.publish_markers(detection_array)

    def publish_markers(self, detection_array):
        """Publish markers for visualization in RViz"""
        markers = MarkerArray()
        for i, det in enumerate(detection_array.detections):
            marker = Marker()
            marker.header = detection_array.header
            marker.id = i
            marker.type = Marker.CUBE
            marker.pose = det.bbox.center
            marker.scale = det.bbox.size
            marker.color.r = 1.0
            marker.color.a = 0.5
            markers.markers.append(marker)
        self.pub_markers.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = DopeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()