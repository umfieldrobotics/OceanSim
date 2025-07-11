# Omniverse import
import numpy as np
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf

class TerrainInstancer_Scenario():
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

       
   
   
   
   
   
   
   
    # from omni.kit.viewport.utility import get_active_viewport
    # self.viewport_api = get_active_viewport()
    # capture = self.viewport_api.schedule_capture(ByteCapture(self.on_capture_completed, aov_name='LdrColor'))
    # def on_capture_completed(self, buffer, buffer_size, width, height, format):
    #     '''
    #     Example
    #     buffer: <capsule object NULL at 0x70805cd0c660>
    #     buffer_size: 3686400
    #     width: 1280
    #     height: 720
    #     format: TextureFormat.RGBA8_UNORM
    #     '''
    #     self.image_provider.set_raw_bytes_data(raw_bytes=buffer,
    #                                             sizes=[width, height],
    #                                             format=format)
