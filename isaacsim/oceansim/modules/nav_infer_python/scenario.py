from pxr import PhysxSchema, Gf
import numpy as np
from omni.replicator.core import BackendDispatch
import omni.replicator.core.scripts.functional as F
from isaacsim.sensors.camera import Camera
from collections import defaultdict
import isaacsim.core.utils.rotations as rotations_utils
import isaacsim.core.utils.numpy as numpy_utils
from .nomad.explore import NoMadModel
from typing import Tuple
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.util.debug_draw import _debug_draw
import isaacsim.core.utils.transformations as transformations_utils
import isaacsim.core.utils.prims as prims_utils

EPS = 1e-8
DT = 1/60
MAX_V = 0.2
MAX_W = 0.4
class NavScenario:
    def __init__(self):
        
        self._running_scenario = False
        self._time = 0.0

        self._rob = None
        self._cam = None
        
        self._model = None
        self._infer_frequency = 1 / 30
        self._prev_infer_times = 0
        
        self._draw =  _debug_draw.acquire_debug_draw_interface()
    def setup_scenario(self, rob: SingleRigidPrim, cam: Camera):
        self._rob = rob
        self._cam = cam
        self._cam.initialize()

        self._model = NoMadModel()
        self._world_prim = prims_utils.get_prim_at_path("/Env")
    
        self.linear_vel = Gf.Vec3f(0.0)
        self.angular_vel = Gf.Vec3f(0.0)
        self._running_scenario = True
        
    def teardown_scenario(self):
        

        self._model = None
        self._running_scenario = False
        self._time = 0.0
        



    def update_scenario(self, step: float):
        if (not self._running_scenario) or (self._cam.get_rgb().size == 0):
            return
        
        self._model.callback_obs(self._cam.get_rgb())
        
        
        infer_times, _ = divmod(self._time, self._infer_frequency)
        if infer_times > self._prev_infer_times:
            self._draw.clear_lines()
            actions, waypoint = self._model.infer()
            dx ,dy = waypoint
            print("waypoints:", dx, dy)
            actions = actions[1:].reshape(-1, 8,2)
            num_traj = actions.shape[0]
            trajs = [[] for _ in range(num_traj)]
            for i in range(num_traj):
                for j in range(8):
                    pt = transformations_utils.get_translation_from_target(
                                        translation_from_source=np.array([actions[i,j,0], actions[i,j,1], 0.25]), 
                                        source_prim=self._rob.prim, 
                                        target_prim=self._world_prim)
                    trajs[i].append(pt)
                    
                self._draw.draw_lines_spline(trajs[i], (1/num_traj * i, 1, 1/num_traj * i, 1) , 7, False )

            self._prev_infer_times = infer_times


        #     self.linear_vel = Gf.Vec3f(v, 0.0, 0.0)
        #     self.angular_vel = Gf.Vec3f(0.0, 0.0, np.rad2deg(w))
        #     print(self.linear_vel)

        # self._rob.prim.GetAttribute("physics:velocity").Set(self.linear_vel)
        # self._rob.prim.GetAttribute("physics:angularVelocity").Set(self.angular_vel)



        self._time += step


