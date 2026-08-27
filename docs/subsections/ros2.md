# ROS 2

OceanSim publishes sensor data and accepts velocity commands through ROS when enabled. Use **Enable ROS** in the included scenario to turn publishing on or off.
Follow the example scenario to see how to utlilize ROS in your scenario.

**Important**: Ensure you have cloned [Oceansim_msgs](https://github.com/umfieldrobotics/oceansim_ros_msgs) into your ROS workspace. Build and source this package in the terminal from which you are starting Isaac Sim prior to launching. This is required as certain sensors depend on custom defined message types (this currently applies to the DVL only).

## Published topics

| Topic | Message | Sensor |
|-------|---------|--------|
| `/rgb` | `sensor_msgs/Image` | Underwater camera |
| `/depth` | `sensor_msgs/Image` (`32FC1`) | Underwater camera |
| `/pointcloud` | `sensor_msgs/PointCloud2` | Underwater camera |
| `/sonar_image` | `sensor_msgs/Image` | Imaging sonar |
| `/imu` | `sensor_msgs/Imu` | IMU |
| `/dvl` | `msgs/Dvl` | DVL |
| `/baro` | `sensor_msgs/FluidPressure` | Barometer |
| `/tf` | `tf2_msgs/TFMessage` | Camera / sonar frames |

Also published when camera or sonar is enabled: `{sensor_name}_camera_info` (`sensor_msgs/CameraInfo`). Camera depth and point cloud may also appear as `{sensor_name}_depth` and `{sensor_name}_pointcloud`.

## Subscribed topics

| Topic | Message | Notes |
|-------|---------|-------|
| `/cmd_vel` | `geometry_msgs/Twist` | Active when control mode is not Manual |

Use this topic to send control commands to your vehicle.
