from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 获取 dope 包的路径
    dope_pkg_dir = get_package_share_directory('bottle_grasp')

    # 定义参数文件的路径
    config_file = os.path.join(dope_pkg_dir, 'config', 'config_pose.yaml')

    # 创建 DOPE 节点
    dope_node = Node(
        package='bottle_grasp',
        executable='dope_node',
        name='dope_node',
        output='screen',
        parameters=[config_file]
    )

    # 返回 LaunchDescription
    return LaunchDescription([
        dope_node
    ])