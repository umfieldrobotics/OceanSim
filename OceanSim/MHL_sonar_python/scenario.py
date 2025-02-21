# Omniverse import
import numpy as np
import omni.replicator.core as rep
from omni.replicator.core.scripts.functional import write_np, write_image
import omni.ui as ui
import warp as wp
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

        self._output_dir = '/home/haoyu/Desktop/MHL_replica'




    def setup_scenario(self, rob, sonar, cam):
        self._rob = rob
        self._cam = cam

        self._rob_rigid_prim = SingleRigidPrim(prim_path=get_prim_path(self._rob),
                                                translation=(-0.5, 0.0, -0.95),
                                                orientation=euler_angles_to_quat(np.array([0, 0, 0]), 
                                                                                 degrees=True,
                                                                                 extrinsic=False))
        self._sonar = sonar
        self._sonar.initialize()
        set_camera_view(eye=np.array([-1,-1,1]), target=self._rob_rigid_prim.get_world_pose()[0])

        self._rgb_annot = rep.AnnotatorRegistry.get_annotator(name='LdrColor', device= str(wp.get_preferred_device()))
        self._rgb_annot.attach(self._sonar.rp)
        # self.sonar_rgba_provider = ui.ByteImageProvider()
        self.sonar_provider = ui.ByteImageProvider()
        self.window = ui.Window("Sonar", width=1280, height=2* 720 + 40, visible=True)
        with self.window.frame:
            with ui.ZStack(height=2* 720 + 40 ):
                ui.Rectangle(style={"background_color": 0xFF000000})
                with ui.VStack(height=2* 720 + 40):
                    ui.ImageWithProvider(self.sonar_provider, 
                                         style={"width": 720, 
                                                "height": 720, 
                                                "fill_policy" : ui.FillPolicy.STRETCH,
                                                'alignment': ui.Alignment.CENTER})
                    # ui.ImageWithProvider(self.sonar_rgba_provider, width=1280, height=720)
        
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
        # self._rob_rigid_prim.set_linear_velocity(np.array([1, 0, 0]))
        if self._rgb_annot.get_data().size != 0:        
            self._sonar.scan()
            self._sonar.make_sonar_data(normalizing_method = "range")
            # self._sonar.make_sonar_data()
            # self.sonar_rgba_provider.set_bytes_data_from_gpu(self._rgb_annot.get_data().ptr, 
            #                                                 [self._rgb_annot.get_data().shape[1], self._rgb_annot.get_data().shape[0]])
            self.sonar_provider.set_bytes_data_from_gpu(self._sonar.make_sonar_image().ptr, 
                                                        [self._sonar.make_sonar_image().shape[1], self._sonar.make_sonar_image().shape[0]])

    def save(self):
        pass