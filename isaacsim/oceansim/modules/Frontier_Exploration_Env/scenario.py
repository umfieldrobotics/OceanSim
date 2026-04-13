# Omniverse import
import numpy as np
from pxr import Gf, PhysxSchema

# Isaac sim import
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_path

# ROS2 integ
from isaacsim.oceansim.sensors import ros2_helpers # move to utils

class MHL_Sensor_Example_Scenario():
    def __init__(self):
        self._rob = None
        self._imu = None
        self._sonar = None
        self._cam = None
        self._DVL = None
        self._baro = None
        #self._zed = None

        self._ctrl_mode = None
        self._cmd_vel_controller = None

        self._running_scenario = False
        self._time = 0.0
    #def setup_scenario(self, rob, sonar, cam, DVL, baro, zed,  ctrl_mode):
    def setup_scenario(self, imu, rob, sonar, cam, DVL, baro,  ctrl_mode):
        #Initialize omni graph before dependant sensors
        self.omni_ros = ros2_helpers.OmniHandler(name="FrontierExploration")
        self._imu = imu
        self._rob = rob
        self._sonar = sonar
        self._cam = cam
        self._DVL = DVL
        self._baro = baro
        #self._zed = zed
        self._ctrl_mode = ctrl_mode

        # NOTE: cannot use manual control along with IMU
        if self._imu is not None:
            self._imu.initialize(og_node=self.omni_ros._imu_node)

        if self._sonar is not None:
            approx_freq = 30
            self._sonar.sonar_initialize(include_unlabelled=True,og_node=self.omni_ros._sonar_node)
            ros2_helpers.publish_camera_info( self._sonar, approx_freq)
            # ros2_helpers.publish_rgb( self._cam, approx_freq)
            ros2_helpers.publish_depth( self._sonar, approx_freq)
            ros2_helpers.publish_pointcloud_from_depth( self._sonar, approx_freq)
            ros2_helpers.publish_camera_tf( self._sonar)
        if self._cam is not None:
            self._cam.initialize(
                og_node=self.omni_ros._rgb_node,
                depth_og_node=self.omni_ros._depth_node,
                pointcloud_og_node=self.omni_ros._pointcloud_node,
            )
            approx_freq = 30
            # #info has type mismatch when calling read_camera_info Stage.GetPrimAtPath(Stage, NoneType) did not match C++ signature:
            ros2_helpers.publish_camera_info( self._cam, approx_freq)
            ros2_helpers.publish_rgb( self._cam, approx_freq)
            ros2_helpers.publish_camera_tf( self._cam)
        if self._DVL is not None:
            self._DVL.initialize(og_node=self.omni_ros._dvl_node)
            self._DVL_reading = [0.0, 0.0, 0.0]
        if self._baro is not None:
            self._baro.initialize(og_node=self.omni_ros._baro_node)

            self._baro_reading = 101325.0 # atmospheric pressure (Pa)
        # if self._zed is not None:
        #     self._zed.initalize()
        
        # Setup cmd_vel ROS2 subscriber (works with any control mode except Manual)
        if ctrl_mode != "Manual control":
            from ...utils.cmd_vel_subscriber import CmdVelController
            robot_path = get_prim_path(self._rob)
            self._cmd_vel_controller = CmdVelController(robot_prim_path=robot_path)

        # Apply the physx force schema if manual control
        if ctrl_mode == "Manual control":
            from ...utils.keyboard_cmd import keyboard_cmd

            self._rob_forceAPI = PhysxSchema.PhysxForceAPI.Apply(self._rob)
            self._force_cmd = keyboard_cmd(base_command=np.array([0.0, 0.0, 0.0]),
                                      input_keyboard_mapping={
                                        # forward command
                                        "W": [10.0, 0.0, 0.0],
                                        # backward command
                                        "S": [-10.0, 0.0, 0.0],
                                        # leftward command
                                        "A": [0.0, 10.0, 0.0],
                                        # rightward command
                                        "D": [0.0, -10.0, 0.0],
                                         # rise command
                                        "UP": [0.0, 0.0, 10.0],
                                        # sink command
                                        "DOWN": [0.0, 0.0, -10.0],
                                      })
            self._torque_cmd = keyboard_cmd(base_command=np.array([0.0, 0.0, 0.0]),
                                      input_keyboard_mapping={
                                        # yaw command (left)
                                        "J": [0.0, 0.0, 10.0],
                                        # yaw command (right)
                                        "L": [0.0, 0.0, -10.0],
                                        # pitch command (up)
                                        "I": [0.0, -10.0, 0.0],
                                        # pitch command (down)
                                        "K": [0.0, 10.0, 0.0],
                                        # row command (left)
                                        "LEFT": [-10.0, 0.0, 0.0],
                                        # row command (negative)
                                        "RIGHT": [10.0, 0.0, 0.0],
                                      })
            
        self._running_scenario = True
    # This function will only be called if ctrl_mode==waypoints and waypoints files are changed
    def setup_waypoints(self, waypoint_path, default_waypoint_path):
        def read_data_from_file(file_path):
            # Initialize an empty list to store the floats
            data = []
            
            # Open the file in read mode
            with open(file_path, 'r') as file:
                # Read each line in the file
                for line in file:
                    # Strip any leading/trailing whitespace and split the line by spaces
                    float_strings = line.strip().split()
                    
                    # Convert the list of strings to a list of floats
                    floats = [float(x) for x in float_strings]
                    
                    # Append the list of floats to the data list
                    data.append(floats)
            
            return data
        try:
            self.waypoints = read_data_from_file(waypoint_path)
            print('Waypoints loaded successfully.')
            print(f'Waypoint[0]: {self.waypoints[0]}')
        except:
            self.waypoints = read_data_from_file(default_waypoint_path)
            print('Fail to load this waypoints. Back to default waypoints.')

        
    def teardown_scenario(self):

        # Because these two sensors create annotator cache in GPU,
        # close() will detach annotator from render product and clear the cache.
        if self._imu is not None: # TODO do better than derefing
            self._imu = None
        if self._sonar is not None:
            self._sonar.close()
        if self._cam is not None:
            self._cam.close()

        # Clear cmd_vel controller
        if self._cmd_vel_controller is not None:
            self._cmd_vel_controller.cleanup()
            self._cmd_vel_controller = None

        # clear the keyboard subscription
        if self._ctrl_mode=="Manual control":
            self._force_cmd.cleanup()
            self._torque_cmd.cleanup()

        self._rob = None
        self._sonar = None
        self._cam = None
        self._DVL = None
        self._baro = None
       # self._zed = None
        self._running_scenario = False
        self._time = 0.0


    def update_scenario(self, step: float):


        if not self._running_scenario:
            return

        self._time += step
        if self._imu is not None:
            self._imu.read()
        if self._sonar is not None:
            self._sonar.make_sonar_data()
        if self._cam is not None:
            self._cam.render()
        if self._DVL is not None:
            self._DVL_reading = self._DVL.read()
        if self._baro is not None:
            self._baro_reading = float(self._baro.read()) # og was self._baro.get_pressure()
        # if self._zed is not None:
        #     self._zed.render()

        # Update cmd_vel controller if active
        if self._cmd_vel_controller is not None:
            self._cmd_vel_controller.update(self._rob)

        if self._ctrl_mode=="Manual control":
            force_cmd = Gf.Vec3f(*self._force_cmd._base_command)
            torque_cmd = Gf.Vec3f(*self._torque_cmd._base_command)
            self._rob_forceAPI.CreateForceAttr().Set(force_cmd)
            self._rob_forceAPI.CreateTorqueAttr().Set(torque_cmd)
        elif self._ctrl_mode=="Waypoints":
            if len(self.waypoints) > 0:
                waypoints = self.waypoints[0]
                self._rob.GetAttribute('xformOp:translate').Set(Gf.Vec3f(waypoints[0], waypoints[1], waypoints[2]))
                self._rob.GetAttribute('xformOp:orient').Set(Gf.Quatd(waypoints[3], waypoints[4], waypoints[5], waypoints[6]))
                self.waypoints.pop(0)
            else:
                print('Waypoints finished')                
        elif self._ctrl_mode=="Straight line":
            SingleRigidPrim(prim_path=get_prim_path(self._rob)).set_linear_velocity(np.array([0.5,0,0])) 




        

        
