# Omniverse import
import numpy as np
from pxr import Gf, PhysxSchema

# Isaac sim import
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_path
import omni.replicator.core as rep
from omni.replicator.core import AnnotatorRegistry, WriterRegistry, BackendDispatch, Writer
from omni.replicator.core.scripts.functional import write_image, write_json, write_np
from isaacsim.sensors.physics import IMUSensor
from isaacsim.core.utils.prims import get_prim_at_path


MAX_FREQ : int = 200
DVL_FREQ : int = 2
CAMERA_FREQ : int = 30


class RobDriver_Scenario():
    def __init__(self):
        self._rob = None
        self._imu = None
        self._id = 0
        self._ctrl_mode = None
        self._running_scenario = False
        self._time = 0.0
        self._output_directory = "/mnt/frog-users/projects/OceanSim/foundnationStero/sdg_data/"
        self._record_data = []

    def setup_scenario(self, rob, ctrl_mode):
        self._rob = rob
        self._ctrl_mode = ctrl_mode
        self._id = 0
        self._imu = IMUSensor(prim_path="/World/rob/base_link/imu_link/Imu_Sensor", frequency=MAX_FREQ)

        rp_L = rep.create.render_product("/World/rob/base_link/left_camera/cam_L", (960, 544))
        rp_R = rep.create.render_product("/World/rob/base_link/right_camera/cam_R", (960, 544))

        self._rgbAnnot_L = AnnotatorRegistry.get_annotator("rgb", device="cuda:0")
        self._rgbAnnot_R = AnnotatorRegistry.get_annotator("rgb", device="cuda:0")
        self._distance_to_image_planeAnnot_L = AnnotatorRegistry.get_annotator("distance_to_image_plane", device="cuda:0")
        self._distance_to_image_planeAnnot_R = AnnotatorRegistry.get_annotator("distance_to_image_plane", device="cuda:0")
        # self._camera_paramsAnnot_L = AnnotatorRegistry.get_annotator("camera_params")
        # self._camera_paramsAnnot_R = AnnotatorRegistry.get_annotator("camera_params")

        self._dvl = SingleRigidPrim("/World/rob")


        self._rgbAnnot_L.attach(rp_L)
        self._rgbAnnot_R.attach(rp_R)
        self._distance_to_image_planeAnnot_L.attach(rp_L)
        self._distance_to_image_planeAnnot_R.attach(rp_R)
        # self._camera_paramsAnnot_L.attach(rp_L)
        # self._camera_paramsAnnot_R.attach(rp_R)

        self._backend = BackendDispatch({"paths": {"out_dir": self._output_directory}})
        
        # Apply the physx force schema if manual control
        from ...utils.keyboard_cmd import keyboard_cmd

        self._rob_forceAPI = PhysxSchema.PhysxForceAPI.Apply(self._rob)
        self._force_cmd = keyboard_cmd(base_command=np.array([0.0, 0.0, 0.0]),
                                    input_keyboard_mapping={
                                    # forward command
                                    "W": [10.0, 0.0, 0.0],
                                    # backward command
                                    "S": [-10.0, 0.0, 0.0],
                                    # leftward command
                                    "A": [0.0, 10.0, 0.0],
                                    # rightward command
                                    "D": [0.0, -10.0, 0.0],
                                        # rise command
                                    "UP": [0.0, 0.0, 10.0],
                                    # sink command
                                    "DOWN": [0.0, 0.0, -10.0],
                                    })
        self._torque_cmd = keyboard_cmd(base_command=np.array([0.0, 0.0, 0.0]),
                                    input_keyboard_mapping={
                                    # yaw command (left)
                                    "J": [0.0, 0.0, 10.0],
                                    # yaw command (right)
                                    "L": [0.0, 0.0, -10.0],
                                    # pitch command (up)
                                    "I": [0.0, -10.0, 0.0],
                                    # pitch command (down)
                                    "K": [0.0, 10.0, 0.0],
                                    # row command (left)
                                    "LEFT": [-10.0, 0.0, 0.0],
                                    # row command (negative)
                                    "RIGHT": [10.0, 0.0, 0.0],
                                    })

        self._running_scenario = True
        self._record_data = []
    # This function will only be called if ctrl_mode==waypoints and waypoints files are changed
    

        
    def teardown_scenario(self):


        # clear the keyboard subscription
        if self._ctrl_mode:
            self._force_cmd.cleanup()
            self._torque_cmd.cleanup()

        self._rob = None
        self._imu = None
        self._dvl = None
        self._running_scenario = False
        self._time = 0.0
        self._record_data = []



    def update_scenario(self, step: float):

        
        if not self._running_scenario:
            return
        
        self._time += step
        

        self.data_frame = self._imu.get_current_frame()
        self.collect_data(int(self.data_frame['physics_step']))

        
        force_cmd = Gf.Vec3f(*self._force_cmd._base_command)
        torque_cmd = Gf.Vec3f(*self._torque_cmd._base_command)
        self._rob_forceAPI.CreateForceAttr().Set(force_cmd)
        self._rob_forceAPI.CreateTorqueAttr().Set(torque_cmd)

        
        self._id += 1


    def collect_data(self, index: int):
        print(f"Collecting f={self.data_frame['time']} data at index {index}")
        gt_position = self._dvl.get_current_dynamic_state().position
        gt_orientation = self._dvl.get_current_dynamic_state().orientation
        self.data_frame.update({"gt_position" : gt_position.tolist(), "gt_orientation" : gt_orientation.tolist()})
        for key, value in self.data_frame.items():
            if isinstance(value, np.ndarray):
                self.data_frame[key] = value.tolist()

        
        if index % CAMERA_FREQ == 0:
            print(f"[{index}] Writing camera data")
            self._backend.schedule(write_image, data=self._rgbAnnot_L.get_data(), path="cam_L/rgb" + f"/rgb_{index}.png")
            self._backend.schedule(write_image, data=self._rgbAnnot_R.get_data(), path="cam_R/rgb" + f"/rgb_{index}.png")
            self._backend.schedule(write_np, data=self._distance_to_image_planeAnnot_L.get_data(), path="cam_L/depth" + f"/distance_to_image_plane_{index}.npy")
            self._backend.schedule(write_np, data=self._distance_to_image_planeAnnot_R.get_data(), path="cam_R/depth" + f"/distance_to_image_plane_{index}.npy")

        if index % DVL_FREQ == 0:
            self.data_frame.update({"dvl" : self._dvl.get_linear_velocity().tolist()})
        
        self._record_data.append(self.data_frame)
        

    def save_data_frame(self):
        if self._record_data:
            self._backend.schedule(write_json, data=self._record_data, path="data.json")
            print(f"Saved {len(self._record_data)} data frames to {self._output_directory}/data.json")
        self._record_data = []
        
    @staticmethod
    def _process_camera_parameters(camera_params) -> dict:
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
