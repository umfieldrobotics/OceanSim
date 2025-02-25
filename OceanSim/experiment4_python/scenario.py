# Omniverse import
import numpy as np
import omni.replicator.core as rep
from omni.replicator.core.scripts.functional import write_image, write_json, write_np


# Isaac sim import
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_path, get_prim_at_path
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.viewports import set_camera_view

class figure_Scenario():
    def __init__(self):
        self._rob = None

        self._running_scenario = False
        self._time = 0.0




    def setup_scenario(self, rob):
        self._rob = rob

        self._rob_rigid_prim = SingleRigidPrim(prim_path=get_prim_path(self._rob),
                                                translation=(0.0, 0.0, 0.0),
                                                orientation=euler_angles_to_quat(np.array([-11, 0.0, 90]), 
                                                                                 degrees=True,
                                                                                 extrinsic=False))




        set_camera_view(eye=np.array([1,1,1]), target=self._rob_rigid_prim.get_world_pose()[0])
        
            
        self._running_scenario = True



    
    
    def teardown_scenario(self):

        self._running_scenario = False


        
        self._rob = None

        
        self._time = 0.0


    def update_scenario(self, step: float):

        
        if not self._running_scenario:
            return
        
        self._time += step




        # self._rob_rigid_prim.set_linear_velocity(np.array([1, 0, 0]))
        