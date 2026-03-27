# Omniverse import
import numpy as np
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf

class SDGplayground_Scenario():
    def __init__(self):

        self._running_scenario = False
        self._time = 0.0
        self._id = 0


    def setup_scenario(self):

        self._running_scenario = True


    def teardown_scenario(self):
        self._running_scenario = False
        self._time = 0.0
        self._id = 0



    def update_scenario(self, step: float):

        
        if not self._running_scenario:
            return
        
        
        self._time += step
        
        self._id += 1

       
   