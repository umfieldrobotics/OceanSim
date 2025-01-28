import numpy as np
from omni.replicator.core import AnnotatorRegistry, BackendDispatch, Writer, WriterRegistry
from omni.replicator.core.scripts.functional import write_image, write_json
import time
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

class underwaterImageWriter(Writer):
    """Imaging Sonar Writer
    Args:
        output_dir:
            Output directory string that indicates the directory to save the results.
        use_subfolders:
            If True, the writer will create subfolders for each render product, otherwise all data is saved in the same folder.
    """
    RGB_ANNOT_NAME = "rgb"
    POINTCLOUD_ANNOT_NAME = "pointcloud"
    CAM_PARAMS_ANNOT_NAME = "camera_params"
    
    def __init__(
        self,
        output_dir,
        use_subfolders=True,
    ):
        

        self.version = "0.0.1"
        self.backend = BackendDispatch({"paths": {"out_dir": output_dir}})



        self.max_range = 4
        self.base_intensity = 255
        self.reflectivity = 1
        self.attenuation = 0.01

        self._use_subfolders = use_subfolders

        # Handle multiple render products scenario (e.g. single render product:'rgb', multiple render products: 'rgb-{rp_name}')
        self._render_product_names = []
        self._multiple_render_products = False

        # Store processed data to be written every frame in the selected format
        self._frame_data = {}

        # Setting the annotators
        self.annotators = []
        # Annotate with an rgb reading for ground truth
        self.annotators.append(AnnotatorRegistry.get_annotator(self.RGB_ANNOT_NAME)) 
        # Annotate with a point cloud reading for sonar computation
        self.annotators.append(AnnotatorRegistry.get_annotator(self.POINTCLOUD_ANNOT_NAME))
        # Annotate with a camera info for sonar computation
        self.annotators.append(AnnotatorRegistry.get_annotator(self.CAM_PARAMS_ANNOT_NAME))
        
        self._frame_id = 0
        self._start_time = time.time()



    def write(self, data: dict):

        # In case of multiple render products annotator names are suffixed with the render product name:
        # (e.g. 'rgb' -> 'rgb-{rp_name}')
        for rp_name in self._render_product_names:
            # Process the frame data of the current render product
            self._process_frame_data(data, rp_name)


            # Create subfolder name if data should be separated for each render product
            rp_subfolder = f"{rp_name}/" if self._multiple_render_products and self._use_subfolders else ""

            # Write frame data to disk
            self._write_frame_data(data, rp_name, rp_subfolder)
            print(f'[Frame:{self._frame_id}][t:{self._get_time_past()}] Writing {rp_name} data to {self.backend.output_dir}/{rp_subfolder}')

            # If render products are NOT separated into subfolders increment the frame id after processing each render product
            if not self._use_subfolders:
                self._frame_id += 1

        # If render products are separated into subfolders increment the frame id after processing all render products
        if self._use_subfolders:
            self._frame_id += 1

    
    # Override to cache the render product names
    def attach(self, render_products, trigger="omni.replicator.core.OgnOnFrame"):
        super().attach(render_products, trigger)
        self._cache_render_product_names(render_products)

    # Override to clear the writer state
    def detach(self):
        super().detach()
        self._reset_writer_state()

    # Save the render product names for easier data access in the write function
    def _cache_render_product_names(self, render_products):
        if not isinstance(render_products, list):
            render_products = [render_products]
        for rp in render_products:
            rp_name = rp.hydra_texture.get_name()
            self._render_product_names.append(rp_name)
        # Check if there are multiple render products, this is used to suffix the annotator names for data access
        self._multiple_render_products = len(self._render_product_names) > 1

    # Reset the writer state
    def _reset_writer_state(self):
        self._render_product_names = []
        self._frame_id = 0
        self._multiple_render_products = False
        self._start_time = time.time()

    def _get_time_past(self):
        return format(time.time() - self._start_time, ".2f")


    def _process_frame_data(self, data: dict, render_product_name: str):
        # Store the frame data for writing to disk
        self._frame_data = {}

        # Get and process the camera parameters annotator data in the selected format
        camera_params_annot_name = (
            f"{self.CAM_PARAMS_ANNOT_NAME}-{render_product_name}"
            if self._multiple_render_products else self.CAM_PARAMS_ANNOT_NAME
        )

        # Get RGB data from annotator
        rgb_annot_name = (
            f"{self.RGB_ANNOT_NAME}-{render_product_name}" 
            if self._multiple_render_products else self.RGB_ANNOT_NAME
        )
        
        
        self._frame_data['rgb_data'] = data[rgb_annot_name]
        self._frame_data["camera_data"] = self._process_camera_parameters(data[camera_params_annot_name])

        # For now, the camera only takes raw rgb image and camera's z cooridnate as inputs
        camera_z_pos = -(np.transpose(data[camera_params_annot_name]['cameraViewTransform'])[2,3])
        self._frame_data["processed_rgb_data"] = self._process_rgb(raw_img=data[rgb_annot_name], 
                                                                   depth=camera_z_pos)
    

    def _write_frame_data(self, data: dict, render_product_name: str, render_product_subfolder: str = ""):

        # Write rgb to png image
        rgb_file_path = f"{render_product_subfolder}{render_product_name}_rgb_{self._frame_id}.png"
        self.backend.schedule(write_image, path=rgb_file_path, data=self._frame_data['rgb_data'])
        # Write camera parameters to JSON 
        file_path_cameraParams = f"{render_product_subfolder}{render_product_name}_cameraParams_{self._frame_id}.json"
        self.backend.schedule(write_json, path=file_path_cameraParams, data=self._frame_data['camera_data'])

    
    
    
    # Get the camera parameters from the annotator data
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
    
    def _process_rgb(self, raw_img, depth):
        pass


    
    


WriterRegistry.register(underwaterImageWriter)
