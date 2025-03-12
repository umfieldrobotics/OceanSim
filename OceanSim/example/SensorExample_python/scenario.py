# Omniverse import
import numpy as np
import omni.ui as ui

# Isaac sim import
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_path
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.viewports import set_camera_view

class MHL_Sensor_Example_Scenario():
    def __init__(self):
        self._rob = None
        self._sonar = None
        self._cam = None
        self._DVL = None
        self._baro = None

        self._running_scenario = False
        self._time = 0.0

        # self._output_dir = '/home/haoyu/Desktop/MHL_replica'




    def setup_scenario(self, rob, sonar, cam, DVL, baro):
        self._rob = rob
        self._sonar = sonar
        self._cam = cam
        self._DVL = DVL
        self._baro = baro

        # rob initial pose
        # self._rob.set_local_pose(translation=(-0.5, 0.0, -0.95),
        #                         orientation=euler_angles_to_quat(np.array([0, 0, 0]), 
        #                                                             degrees=True,
        #                                                             extrinsic=False))
        
        if self._sonar is not None:
            self._sonar.sonar_initialize()
        if self._cam is not None:
            self._cam.initialize()
        if self._DVL is not None:
            self._DVL_reading = [0.0, 0.0, 0.0]
        
            
        

        self._running_scenario = True
        
   
    def teardown_scenario(self):


        if self._sonar is not None:
            self._sonar.close()
        if self._cam is not None:
            self._cam.close()


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
        
        SingleRigidPrim(prim_path=get_prim_path(self._rob)).set_linear_velocity(np.array([0.5,0,0]))  
              
        self._DVL_reading = self._DVL.get_linear_vel()




