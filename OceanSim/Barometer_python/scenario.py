# Omniverse import
import numpy as np
import omni.replicator.core as rep
from omni.replicator.core.scripts.functional import write_np, write_image
import omni.ui as ui

# Isaac sim import
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_path
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.viewports import set_camera_view



class Barometer_Scenario():
    def __init__(self):
        self._rob = None
        self._baro = None

        self._running_scenario = False
        self._time = 0.0



    def setup_scenario(self, rob, baro):
        self._rob = rob
        self._baro = baro
        self._running_scenario = True

        SingleRigidPrim(prim_path=get_prim_path(self._rob),
                        translation=np.array([0.0, 0.0, 0.7]),
                        orientation=euler_angles_to_quat(np.array([0.0, 0.0, 0.0]), degrees=True))
        

        set_camera_view(eye=[-2.0, 0.0, 3.0], target=np.array([0.0, 1.25, 0.15]), camera_prim_path="/OmniverseKit_Persp")
        
    
    def teardown_scenario(self):
        self._rob = None
        self._running_scenario = False
        self._time = 0.0
        self._id = 0


    def update_scenario(self, step: float):

        
        if not self._running_scenario:
            return
        self._time += step
        print(self._baro.get_pressure())
        # SingleRigidPrim(prim_path=get_prim_path(self._rob)).set_linear_velocity(np.array([0.5,0,0]))
        
        self._id += 1

       
            
    