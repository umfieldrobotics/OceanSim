import numpy as np
from pxr import Gf, PhysxSchema, Vt
from isaacsim.oceansim.utils.ocean_deform import ocean_deform_launch_kernel
import warp as wp

class OceanScenario():
    def __init__(self):


        self._running_scenario = False
        self._time = 0.0
        self._output_dir = '/home/haoyu/Desktop/viz'
        self._id = 0
        
    def setup_scenario(self, ocean_surface, grid):
        self.ocean_surface = ocean_surface
        self.grid = grid
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
        deformed_points = ocean_deform_launch_kernel(self.grid, self._time)


        self.ocean_surface.GetAttribute("points").Set(deformed_points.numpy())
        


