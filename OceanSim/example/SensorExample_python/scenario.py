# Omniverse import
import numpy as np
from pxr import Gf

# Isaac sim import
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_path
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.viewports import set_camera_view

# custom import
from ...utils.keyboard_cmd import keyboard_cmd
class MHL_Sensor_Example_Scenario():
    def __init__(self):
        self._rob = None
        self._sonar = None
        self._cam = None
        self._DVL = None
        self._baro = None

        self._rob_forceAPI = None

        self._running_scenario = False
        self._time = 0.0

        # self._output_dir = '/home/haoyu/Desktop/MHL_replica'




    def setup_scenario(self, rob, sonar, cam, DVL, baro, rob_forceAPI):
        self._rob = rob
        self._sonar = sonar
        self._cam = cam
        self._DVL = DVL
        self._baro = baro
        self._rob_forceAPI = rob_forceAPI
        
        if self._sonar is not None:
            self._sonar.sonar_initialize()
        if self._cam is not None:
            self._cam.initialize()
        if self._DVL is not None:
            self._DVL_reading = [0.0, 0.0, 0.0]
        if self._baro is not None:
            self._baro_reading = 101325.0 # atmospheric pressure (Pa)

        if self._rob_forceAPI is not None:
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
        
   
    def teardown_scenario(self):

        # Because these two sensors create annotator cache in GPU,
        # close() will detach annotator from render product and clear the cache.
        if self._sonar is not None:
            self._sonar.close()
        if self._cam is not None:
            self._cam.close()

        # clear the keyboard subscription
        if self._rob_forceAPI is not None:
            self._force_cmd.cleanup()
            self._torque_cmd.cleanup()

        self._rob = None
        self._sonar = None
        self._cam = None
        self._DVL = None
        self._baro = None
        self._running_scenario = False
        self._time = 0.0


    def update_scenario(self, step: float):

        
        if not self._running_scenario:
            return
        
        self._time += step
        
        if self._sonar is not None:
            self._sonar.make_sonar_data()
        if self._cam is not None:
            self._cam.render()
        if self._DVL is not None:
            self._DVL_reading = self._DVL.get_linear_vel()
        if self._baro is not None:
            self._baro_reading = self._baro.get_pressure()

        if self._rob_forceAPI is not None:
            force_cmd = Gf.Vec3f(*self._force_cmd._base_command)
            torque_cmd = Gf.Vec3f(*self._torque_cmd._base_command)
            self._rob_forceAPI.CreateForceAttr().Set(force_cmd)
            self._rob_forceAPI.CreateTorqueAttr().Set(torque_cmd)
        else:
            SingleRigidPrim(prim_path=get_prim_path(self._rob)).set_linear_velocity(np.array([0.5,0,0]))  

        


        

        


