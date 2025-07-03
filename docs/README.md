# OceanSim: ROS2 Bridge #

## Instructions: 
This is the ROS2 version of OceanSim which requires user to set up their ROS2 workspace with Isaac Sim following their official [tutorial](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_ros.html).

Note --[PR14](https://github.com/umfieldrobotics/OceanSim/pull/14#issue-3190565204): 

Before OceanSim extension being activated, the extension isaacsim.ros2.bridge should be activated, otherwise rclpy will fail to be loaded.

We suggest that make sure the extension isaacsim.ros2.bridge is being setup to "AUTOLOADED" in Window->Extension. 


## Usage:
We provided an exmaple util located at `isaacsim/oceansim/utils/ros2_control.py` for user to consult and develop on.

This util extends the control mode to ros control in the **sensor_example** extension. 

## Acknowledgement:
Great appreciation to [Tang-JingWei](https://github.com/Tang-JingWei) for contributng the ros bridge example for OceanSim.

