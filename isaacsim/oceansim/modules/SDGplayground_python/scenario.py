# Omniverse import
import numpy as np
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf
from isaacsim.oceansim.sensors.UW_Camera import UW_Camera
class SDGplayground_Scenario():
    def __init__(self):

        self._running_scenario = False
        self._time = 0.0
        self._id = 0
        self._cam = None


    def setup_scenario(self, camera: UW_Camera):

        self._running_scenario = True
        self._cam = camera
        if self._cam is not None:
            self._cam.initialize()

    def teardown_scenario(self):
        self._running_scenario = False
        self._cam = None
        self._time = 0.0
        self._id = 0



    def update_scenario(self, step: float):

        
        if not self._running_scenario:
            return
        
        self._cam.render()
        
        self._time += step
        
        self._id += 1

       
   