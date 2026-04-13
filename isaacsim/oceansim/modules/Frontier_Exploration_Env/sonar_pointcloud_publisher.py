"""
Simple ROS2 Point Cloud Publisher for OceanSim Imaging Sonar

Just copy this file next to your scenario.py and add these lines:

In setup_scenario():
    from .sonar_pointcloud_publisher import SonarPointCloudPublisher
    self._ros_pub = SonarPointCloudPublisher(self._sonar)

In update_scenario():
    if self._ros_pub:
        self._ros_pub.publish()

In teardown_scenario():
    if self._ros_pub:
        self._ros_pub.shutdown()
"""

import numpy as np
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


class SonarPointCloudPublisher:
    def __init__(self, sonar_sensor, topic='/sonar/point_cloud', frame_id='sonar_link'):
        """
        Simple point cloud publisher for OceanSim sonar.
        
        Args:
            sonar_sensor: Your ImagingSonarSensor instance
            topic: ROS2 topic name (default: /sonar/point_cloud)
            frame_id: TF frame (default: sonar_link)
        """
        self._sonar = sonar_sensor
        self._topic = topic
        self._frame_id = frame_id
        
        # Initialize ROS2
        if not rclpy.ok():
            rclpy.init()
        
        self._node = rclpy.create_node('sonar_publisher')
        self._pub = self._node.create_publisher(PointCloud2, topic, 10)
        
        # Spin in background thread
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        
        print(f"[SonarPointCloudPublisher] Publishing on {topic}")
    
    def _spin(self):
        while rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.01)
    
    def publish(self):
        """Call this in your update_scenario() after make_sonar_data()"""
        if self._sonar is None or self._sonar.sonar_map is None:
            return
        
        # Get sonar data
        sonar_map = self._sonar.sonar_map.numpy()  # Shape: (range_bins, azimuth_bins, 3)
        
        # Extract points with intensity > 0.1
        points = []
        for i in range(sonar_map.shape[0]):
            for j in range(sonar_map.shape[1]):
                x = sonar_map[i, j, 0]
                y = sonar_map[i, j, 1]
                intensity = sonar_map[i, j, 2]
                if intensity > 0.1:
                    points.append([x, y, 0.0, intensity])
        
        if not points:
            return
        
        # Create PointCloud2 message
        points_array = np.array(points, dtype=np.float32)
        
        msg = PointCloud2()
        msg.header = Header()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        msg.height = 1
        msg.width = len(points)
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.is_bigendian = False
        msg.data = points_array.tobytes()
        
        self._pub.publish(msg)
    
    def shutdown(self):
        """Call this in teardown_scenario()"""
        if self._node:
            self._node.destroy_node()
        print("[SonarPointCloudPublisher] Shutdown")