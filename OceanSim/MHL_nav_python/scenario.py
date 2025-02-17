# Omniverse import
import numpy as np
import omni.replicator.core as rep
from omni.replicator.core.scripts.functional import write_np, write_image

# Isaac sim import
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_path
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.viewports import set_camera_view
class MHL_straighline_navigation_Scenario():
    def __init__(self):
        self._rob = None
        self._DVL = None
        self._cam = None
        self._running_scenario = False
        self._time = 0.0

        self._output_dir = '/home/haoyu-ma/Desktop/MHL_replica'

        self._fourBeam_buffer = []
        self._vel_buffer = []


    def setup_scenario(self, rob, DVL, cam):
        self._rob = rob
        self._DVL = DVL
        self._cam = cam

        self._running_scenario = True

        rob_rigid_prim = SingleRigidPrim(prim_path=get_prim_path(self._rob),
                                        translation=np.array([0.0, 0.0, 0.0]),
                                        orientation=euler_angles_to_quat(np.array([0.0, 0.0, 0.0]), degrees=True))
        

        self._backend = rep.BackendDispatch({"paths": {"out_dir": self._output_dir}})
        rp = rep.create.render_product(
            camera='/MHL/rob/Camera',
            resolution=(1920,1080),
            )
        
        self._ldr = rep.AnnotatorRegistry.get_annotator("LdrColor")
        self._depth = rep.AnnotatorRegistry.get_annotator('distance_to_camera')
        self._ldr.attach(rp)
        self._depth.attach(rp)

        set_camera_view(eye=[-1.0, 0.0, -1.0], target=rob_rigid_prim.get_world_pose()[0], camera_prim_path="/OmniverseKit_Persp")

    def teardown_scenario(self):
        self._rob = None
        self._running_scenario = False
        self._time = 0.0
        self._id = 0


    def update_scenario(self, step: float):

        
        if not self._running_scenario:
            return
        self._time += step
        if self._ldr.get_data().size == 0:
            return
        

        SingleRigidPrim(prim_path=get_prim_path(self._rob)).set_linear_velocity(np.array([5,0,0]))

        self._fourBeam_buffer.append(self._DVL.get_depth())
        self._vel_buffer.append(self._DVL.get_linear_vel())
        self._backend.schedule(write_image, path=f'cam/rgb_{self._id}.png', data=self._ldr.get_data())
        self._backend.schedule(write_np, path=f'depth/depth_{self._id}.npy', data=self._depth.get_data())
        print(f'writing [{self._id}]')
        self._id += 1


    def save(self):
        np.save(file=self._output_dir+"/vel.npy", arr=self._vel_buffer)
        np.save(file=self._output_dir+"/fourBeam.npy", arr=self._fourBeam_buffer)   
        print(f'data written to {self._output_dir}')

