# OceanSim: ROS2 Bridge #

## Instructions: 
This is the ROS2 version of OceanSim which requires user to set up their ROS2 workspace with Isaac Sim following their official [tutorial](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_ros.html).

Note --[PR14](https://github.com/umfieldrobotics/OceanSim/pull/14#issue-3190565204): 

Before OceanSim extension being activated, the extension isaacsim.ros2.bridge should be activated, otherwise rclpy will fail to be loaded.

We suggest that make sure the extension isaacsim.ros2.bridge is being setup to "AUTOLOADED" in Window->Extension. 


## Usage:
### ros2 control:
We provided an exmaple util located at `isaacsim/oceansim/utils/ros2_control.py` for user to consult and develop on.

This util extends the control mode to ros control in the **sensor_example** extension. 

### ros2 publish uw image:
We add ros2 publish uw image function in the UW_Camera class, located at `isaacsim/oceansim/sensors/UW_Camera.py`.

For testing, we provide a ros2 subscriber example located at `isaacsim/oceansim/utils/ros2_image_subscriber.py`.

Test steps:
1. check the Underwater Camera checkbox in the **sensor_example** extension.
2. run the simulation.
3. open a terminal and run the ros2_image_subscriber.py.
```
cd /path/to/oceansim/utils
python3 ros2_image_subscriber.py
```

## Acknowledgement:
Great appreciation to [Tang-JingWei](https://github.com/Tang-JingWei) for contributng the ros bridge example for OceanSim.

