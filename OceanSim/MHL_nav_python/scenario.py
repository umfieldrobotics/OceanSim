# Omniverse import
import numpy as np

# Isaac sim import
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_path
from isaacsim.core.utils.rotations import euler_angles_to_quat
class MHL_straighline_navigation_Scenario():
    def __init__(self):
        self._rob = None
        self._DVL = None
        self._cam = None
        self._running_scenario = False
        self._time = 0.0
        self._id = 0


        self._depth_buffer = []
        self._vel_buffer = []
        self._singleBeam_buffer = []

        self._output_dir = '/home/haoyu-ma/Desktop/MHL_replica'


    def setup_scenario(self, rob, DVL, cam):
        self._rob = rob
        self._DVL = DVL
        self._cam = cam
        self._cam.initialize(writing_dir = self._output_dir + '/cam')
        self._running_scenario = True

        rob_rigid_prim = SingleRigidPrim(prim_path=get_prim_path(self._rob),
                                        translation=np.array([0.0, 0.0, 0.0]),
                                        orientation=euler_angles_to_quat(np.array([0.0, 0.0, 0.0]), degrees=True))
        
    def teardown_scenario(self):

        self._running_scenario = False
        self._time = 0.0
        self._id = 0
        if self._cam is not None:
            self._cam.close()
        self._cam = None
        self._rob = None
        self._DVL = None


    def update_scenario(self, step: float):

        
        if not self._running_scenario:
            return
        self._time += step

        self._cam.render()
        SingleRigidPrim(prim_path=get_prim_path(self._rob)).set_linear_velocity(np.array([0.5,0,0]))
        self._vel_buffer.append(self._DVL.get_linear_vel())
        self._depth_buffer.append(self._DVL.get_depth())
        self._singleBeam_buffer.append(self._DVL.get_single_beam_range())

        if self._id == 1200:
            self.save()
        
        self._id += 1


    def save(self):
        np.save(file=self._output_dir+"/vel.npy", arr=self._vel_buffer)
        np.save(file=self._output_dir+"/fourBeam.npy", arr=self._depth_buffer)
        np.save(file=self._output_dir+'/singleBeam.npy', arr=self._singleBeam_buffer)   
        print(f'data written to {self._output_dir}')

