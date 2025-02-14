# Omniverse import
import numpy as np
import omni.replicator.core as rep
from omni.replicator.core.scripts.functional import write_np, write_image

# Isaac sim import
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_path
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.viewports import set_camera_view

class MHL_sonar_test_Scenario():
    def __init__(self):
        self._rob = None
        self._sonar = None
        self._cam = None
        self._running_scenario = False
        self._time = 0.0

        self._output_dir = '/home/haoyu-ma/Desktop/MHL_replica'




    def setup_scenario(self, rob, sonar, cam):
        self._rob = rob
        
        self._rob_rigid_prim = SingleRigidPrim(prim_path=get_prim_path(self._rob),
                                                translation=(0.1, 2.5, 1.1),
                                                orientation=euler_angles_to_quat(np.array([-38, 0, 90]), 
                                                                                 degrees=True,
                                                                                 extrinsic=False))
        self._sonar = sonar
        self._sonar.initialize(self._output_dir)
        self._sonar.scan()
        self._sonar.make_sonar_data()

        self._cam = cam

        self._running_scenario = True
        
    

    def teardown_scenario(self):
        self._rob = None
        self._sonar = None
        self._cam = None
        self._running_scenario = False
        if self._sonar is not None:
            self._sonar.close()
        self._time = 0.0


    def update_scenario(self, step: float):

        
        if not self._running_scenario:
            return
        
        self._time += step
        self._rob_rigid_prim.set_linear_velocity(np.array([-0.1, 0, 0]))
        self._sonar.scan()
        self._sonar.make_sonar_data()



    def save(self):
        pass