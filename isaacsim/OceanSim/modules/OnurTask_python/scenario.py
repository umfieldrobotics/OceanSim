# Omniverse import
import numpy as np
from pxr import Gf, PhysxSchema

# Isaac sim import
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_path


class OnurLog_Scenario():
    def __init__(self):
        self._rob = None
        self._sonar = None
        self._cam = None
        self._DVL = None
        self._baro = None
        self._IMU = None

        self._ctrl_mode = None

        self._running_scenario = False
        self._time = 0.0

    def setup_scenario(self, rob, sonar, cam, DVL, baro, IMU, ctrl_mode):
        self._rob = rob
        self._sonar = sonar
        self._cam = cam
        self._DVL = DVL
        self._baro = baro
        self._IMU = IMU

        self._sonar_dt = 1/30
        self._cam_dt = 1/30
        self._DVL_dt = 1/10
        self._sonar_time = 0.0
        self._DVL_time = 0.0
        self._cam_time = 0.0


        self._ctrl_mode = ctrl_mode
        self._sonar.sonar_initialize()
        self._cam.initialize()
        self._DVL_reading = np.array([0.0, 0.0, 0.0], dtype=float)
        self._IMU_reading = {'time': 0.0, 
                             'physics_step': 0.0, 
                             'lin_acc': np.array([ 0.0, 0.0, -9.81], dtype=float), 
                             'ang_vel': np.array([0., 0., 0.], dtype=float), 
                             'orientation': np.array([1.0, 0.0, 0.0, 0.0],dtype=float)}

        self._baro_reading = 101325.0 # atmospheric pressure (Pa)

        # Apply the physx force schema if manual control
        if ctrl_mode == "Manual control":
            from isaacsim.OceanSim.utils.keyboard_cmd import keyboard_cmd

            self._force_cmd = keyboard_cmd(base_command=np.array([0.0, 0.0, 0.0]),
                                      input_keyboard_mapping={
                                        # forward command
                                        "W": [10.0, 0.0, 0.0],
                                        # backward command
                                        "S": [-10.0, 0.0, 0.0],
                                        # leftward command
                                        "A": [0.0, -10.0, 0.0],
                                        # rightward command
                                        "D": [0.0, 10.0, 0.0],
                                         # rise command
                                        "UP": [0.0, 0.0, -10.0],
                                        # sink command
                                        "DOWN": [0.0, 0.0, 10.0],
                                      })
            self._torque_cmd = keyboard_cmd(base_command=np.array([0.0, 0.0, 0.0]),
                                      input_keyboard_mapping={
                                        # yaw command (left)
                                        "J": [0.0, 0.0, -10.0],
                                        # yaw command (right)
                                        "L": [0.0, 0.0, 10.0],
                                        # pitch command (up)
                                        "I": [0.0, 10.0, 0.0],
                                        # pitch command (down)
                                        "K": [0.0, -10.0, 0.0],
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
        if self._sonar is not None:
            self._sonar.close()
        if self._cam is not None:
            self._cam.close()

        # clear the keyboard subscription
        if self._ctrl_mode=="Manual control":
            self._force_cmd.cleanup()
            self._torque_cmd.cleanup()

        self._rob = None
        self._sonar = None
        self._cam = None
        self._DVL = None
        self._baro = None
        self._IMU = None
        self._running_scenario = False
        self._time = 0.0


    def update_scenario(self, step: float):

        
        if not self._running_scenario:
            return
        
        self._time += step
        self._sonar_time += step
        self._cam_time += step
        self._DVL_time += step
        if self._sonar_time >= self._sonar_dt:
            self._sonar.make_sonar_data()
            self._sonar_time = 0.0
        if self._cam_time >= self._cam_dt:
            self._cam.render()
            self._cam_time = 0.0
        if self._DVL_time >= self._DVL_dt:
            self._DVL_reading = self._DVL.get_linear_vel()
            self._DVL_time = 0.0
        
        self._baro_reading = self._baro.get_pressure()
        self._IMU_reading = self._IMU.get_current_frame()
        
        if self._ctrl_mode=="Manual control":
            force_cmd = Gf.Vec3f(*self._force_cmd._base_command)
            torque_cmd = Gf.Vec3f(*self._torque_cmd._base_command)
            self._rob.GetAttribute("physxForce:force").Set(force_cmd)
            self._rob.GetAttribute("physxForce:torque").Set(torque_cmd)
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




        

        


