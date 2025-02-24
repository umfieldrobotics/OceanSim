# Omniverse import
import numpy as np
import omni.replicator.core as rep
from omni.replicator.core.scripts.functional import write_image, write_json, write_np
import omni.ui as ui
import warp as wp

# Isaac sim import
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_path, get_prim_at_path
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.viewports import set_camera_view

class MHL_test2_Scenario():
    def __init__(self):
        self._rob = None
        self._sonar = None
        self._cam = None
        self._running_scenario = False
        self._time = 0.0

        # self._output_dir = '/home/haoyu-ma/Desktop/MHL_replica'
        self._output_dir = '/home/haoyu/Desktop/MHL_replica'



    def setup_scenario(self, rob, sonar, cam):
        self._rob = rob
        self._cam = cam

        self._rob_rigid_prim = SingleRigidPrim(prim_path=get_prim_path(self._rob),
                                                translation=(-0.05, 1.6, -1.0),
                                                orientation=euler_angles_to_quat(np.array([11.0, 0.0, -90.0]), 
                                                                                 degrees=True,
                                                                                 extrinsic=False))
        self._sonar = sonar
        self._sonar.sonar_initialize()
        set_camera_view(eye=np.array([-1,-1,1]), target=self._rob_rigid_prim.get_world_pose()[0])
        self._backend = rep.BackendDispatch({"paths": {"out_dir": self._output_dir}})
        self._rgba_annot = rep.AnnotatorRegistry.get_annotator(name='LdrColor', device=str(wp.get_preferred_device()))
        self._depth_annot = rep.AnnotatorRegistry.get_annotator(name="distance_to_camera", device=str(wp.get_preferred_device()))
        self._cam_param_annot = rep.AnnotatorRegistry.get_annotator(name="camera_params")

        # cam = rep.create.camera(get_prim_path(self._cam))
        rp = rep.create.render_product(camera='/World/rob/Camera', resolution=(1920,1080))
        self._rgba_annot.attach(rp)
        self._depth_annot.attach(rp)
        self._cam_param_annot.attach(rp)
        self._running_scenario = True



        
    
    
    def teardown_scenario(self):

        self._running_scenario = False
        if self._sonar is not None:
            self._sonar.close()

        self._rob = None
        self._sonar = None
        self._cam = None
        self._time = 0.0


    def update_scenario(self, step: float):

        
        if not self._running_scenario:
            return
        
        self._time += step
        # self._rob_rigid_prim.set_linear_velocity(np.array([1, 0, 0]))
        if self._rgba_annot.get_data().size !=0:
            self._sonar.make_sonar_data(normalizing_method = "range")
           

            # self._backend.schedule(write_json, path='cam_param.json', data=self._process_camera_parameters(self._cam_param_annot.get_data()))
            # self._backend.schedule(write_image, path='rgba.png', data=self._rgba_annot.get_data())
            # self._backend.schedule(write_np, path='depth.npy', data=self._depth_annot.get_data())
            # self._backend.schedule(write_image, path='sonar.png', data=self._sonar.make_sonar_image())



    def _process_camera_parameters(self, camera_params) -> dict:
        camera_data = {}
        camera_data["aperture"] = camera_params["cameraAperture"].tolist()
        camera_data["aperture_offset"] = camera_params["cameraApertureOffset"].tolist()
        camera_data["focal_length"] = float(camera_params["cameraFocalLength"])
        camera_data["resolution"] = camera_params["renderProductResolution"].tolist()
        camera_data["meters_per_scene_unit"] = float(camera_params["metersPerSceneUnit"])

        # OV only supports square pixels, so the pixel size is the same in both x and y directions
        # https://docs.omniverse.nvidia.com/materials-and-rendering/latest/cameras.html#cameras
        pixel_size = camera_params["cameraAperture"][0] / camera_params["renderProductResolution"][0]
        camera_data["intrinsics"] = {
            "fx": camera_params["cameraFocalLength"] / pixel_size,
            "fy": camera_params["cameraFocalLength"] / pixel_size,
            "cx": camera_params["renderProductResolution"][0] / 2.0 + camera_params["cameraApertureOffset"][0],
            "cy": camera_params["renderProductResolution"][1] / 2.0 + camera_params["cameraApertureOffset"][1],
        }
        camera_data["camera_view_matrix"] = np.round(camera_params["cameraViewTransform"], 5).reshape(4, 4).tolist()
        camera_data["camera_projection_matrix"] = np.round(camera_params["cameraProjection"], 5).reshape(4, 4).tolist()

        return camera_data
