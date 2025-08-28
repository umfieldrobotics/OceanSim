# Omniverse import
import numpy as np
from pxr import Gf, PhysxSchema

# Isaac sim import
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_path
import omni.replicator.core as rep
from omni.replicator.core import AnnotatorRegistry, WriterRegistry, BackendDispatch, Writer
from omni.replicator.core.scripts.functional import write_image, write_json, write_np

class RobDriver_Scenario():
    def __init__(self):
        self._rob = None
        self._sensor = None
        self._id = 0
        self._ctrl_mode = None
        self._running_scenario = False
        self._time = 0.0
        self._output_directory = "C:/Users/mahaoyu/Desktop/SDG/"

    def setup_scenario(self, rob, ctrl_mode):
        self._rob = rob
        self._ctrl_mode = ctrl_mode
        self._id = 0
        rp_L = rep.create.render_product("/World/rob/StereoCam/StereoCam_L_Xform/StereoCam_L", (960, 544))
        rp_R = rep.create.render_product("/World/rob/StereoCam/StereoCam_R_Xform/StereoCam_R", (960, 544))


        self._rgbAnnot_L = AnnotatorRegistry.get_annotator("rgb", device="cuda:0")
        self._rgbAnnot_R = AnnotatorRegistry.get_annotator("rgb", device="cuda:0")
        self._distance_to_image_planeAnnot_L = AnnotatorRegistry.get_annotator("distance_to_image_plane", device="cuda:0")
        self._distance_to_image_planeAnnot_R = AnnotatorRegistry.get_annotator("distance_to_image_plane", device="cuda:0")
        self._camera_paramsAnnot_L = AnnotatorRegistry.get_annotator("camera_params")
        self._camera_paramsAnnot_R = AnnotatorRegistry.get_annotator("camera_params")



        self._rgbAnnot_L.attach(rp_L)
        self._rgbAnnot_R.attach(rp_R)
        self._distance_to_image_planeAnnot_L.attach(rp_L)
        self._distance_to_image_planeAnnot_R.attach(rp_R)
        self._camera_paramsAnnot_L.attach(rp_L)
        self._camera_paramsAnnot_R.attach(rp_R)

        self._backend = BackendDispatch({"paths": {"out_dir": self._output_directory}})
        
        # Apply the physx force schema if manual control
        if ctrl_mode == "Manual control":
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
    # This function will only be called if ctrl_mode==waypoints and waypoints files are changed
    def setup_waypoints(self, waypoint_path, default_waypoint_path):
        def read_data_from_file(file_path):
            # Initialize an empty list to store the floats
            data = []
            
            # Open the file in read mode
            with open(file_path, 'r') as file:
                # Read each line in the file
                for line in file:
                    # Strip any leading/trailing whitespace and split the line by spaces
                    float_strings = line.strip().split()
                    
                    # Convert the list of strings to a list of floats
                    floats = [float(x) for x in float_strings]
                    
                    # Append the list of floats to the data list
                    data.append(floats)
            
            return data
        try:
            self.waypoints = read_data_from_file(waypoint_path)
            print('Waypoints loaded successfully.')
            print(f'Waypoint[0]: {self.waypoints[0]}')
        except:
            self.waypoints = read_data_from_file(default_waypoint_path)
            print('Fail to load this waypoints. Back to default waypoints.')

        
    def teardown_scenario(self):



        # clear the keyboard subscription
        if self._ctrl_mode=="Manual control":
            self._force_cmd.cleanup()
            self._torque_cmd.cleanup()

        self._rob = None
        self._sensor = None
        self._running_scenario = False
        self._time = 0.0



    def update_scenario(self, step: float):

        
        if not self._running_scenario:
            return
        
        self._time += step
        if self._rgbAnnot_L.get_data().shape[0] > 0:
            self._backend.schedule(write_image, data=self._rgbAnnot_L.get_data(), path="cam_L/rgb" + f"/rgb_{self._id}.png")
            self._backend.schedule(write_image, data=self._rgbAnnot_R.get_data(), path="cam_R/rgb" + f"/rgb_{self._id}.png")
            self._backend.schedule(write_np, data=self._distance_to_image_planeAnnot_L.get_data(), path="cam_L/depth" + f"/distance_to_image_plane_{self._id}.npy")
            self._backend.schedule(write_np, data=self._distance_to_image_planeAnnot_R.get_data(), path="cam_R/depth" + f"/distance_to_image_plane_{self._id}.npy")
            self._backend.schedule(write_json, data=self._process_camera_parameters(self._camera_paramsAnnot_L.get_data()), path="cam_L/cameraParams" + f"/camera_params_{self._id}.json")
            self._backend.schedule(write_json, data=self._process_camera_parameters(self._camera_paramsAnnot_R.get_data()), path="cam_R/cameraParams" + f"/camera_params_{self._id}.json")
            print(f"writing {self._id} data to disk")
        
        
        
        
        
        
        if self._ctrl_mode=="Manual control":
            force_cmd = Gf.Vec3f(*self._force_cmd._base_command)
            torque_cmd = Gf.Vec3f(*self._torque_cmd._base_command)
            self._rob_forceAPI.CreateForceAttr().Set(force_cmd)
            self._rob_forceAPI.CreateTorqueAttr().Set(torque_cmd)
        elif self._ctrl_mode=="Waypoints":
            if len(self.waypoints) > 0:
                waypoints = self.waypoints[0]
                self._rob.GetAttribute('xformOp:translate').Set(Gf.Vec3f(waypoints[0], waypoints[1], waypoints[2]))
                self._rob.GetAttribute('xformOp:orient').Set(Gf.Quatd(waypoints[3], waypoints[4], waypoints[5], waypoints[6]))
                self.waypoints.pop(0)
            else:
                print('Waypoints finished')                
        elif self._ctrl_mode=="Straight line":
            SingleRigidPrim(prim_path=get_prim_path(self._rob)).set_linear_velocity(np.array([0.5,0,0])) 
        
        
        self._id += 1




        

        
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
