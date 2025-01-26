import numpy as np
from omni.replicator.core import AnnotatorRegistry, BackendDispatch, Writer, WriterRegistry
from omni.replicator.core.scripts.functional import write_image, write_json

import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
import time

class imagingSonarWriter(Writer):
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
            print(f'[Frame {self._frame_id}] Writing {rp_name} data to {self.backend.output_dir}/{rp_subfolder}')

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
        
        pcl_annot_name = (
            f'{self.POINTCLOUD_ANNOT_NAME}-{render_product_name}' 
            if self._multiple_render_products else self.POINTCLOUD_ANNOT_NAME
        )

        self._frame_data['sonar_data'] = self.make_sonar_data(pcl=data[pcl_annot_name]['data'],
                                                              normals=data[pcl_annot_name]['pointNormals'],
                                                              viewTransform=data[camera_params_annot_name]['cameraViewTransform'])
        self._frame_data['rgb_data'] = data[rgb_annot_name]
        self._frame_data["camera_data"] = self._process_camera_parameters(data[camera_params_annot_name])

    

    def _write_frame_data(self, data: dict, render_product_name: str, render_product_subfolder: str = ""):

        # Write rgb to png image
        rgb_file_path = f"{render_product_subfolder}{render_product_name}_rgb_{self._frame_id}.png"
        self.backend.schedule(write_image, path=rgb_file_path, data=self._frame_data['rgb_data'])
        # Write camera parameters to JSON 
        file_path_cameraParams = f"{render_product_subfolder}{render_product_name}_cameraParams_{self._frame_id}.json"
        self.backend.schedule(write_json, path=file_path_cameraParams, data=self._frame_data['camera_data'])

        # Convert sonar data to rgba and save as png
        file_path_sonar = f'{render_product_subfolder}{render_product_name}_sonarImage_{self._frame_id}.png'
        sonar_image = self.make_sonar_map(self._frame_data["sonar_data"])
        self.backend.schedule(write_image, path=file_path_sonar, data=sonar_image)
    
    
    
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
    
    def make_sonar_map(self, sonar_data:np.ndarray) -> np.ndarray:
        
        fig = plt.figure(dpi=600)
        ax1 = fig.add_subplot(1,1,1)
        sonar_plot = ax1.scatter(sonar_data[:,0], sonar_data[:,1], c=sonar_data[:,2], cmap='jet', s=0.5, marker='.')
        fig.colorbar(mappable=sonar_plot, ax=ax1)
        fig.canvas.draw()
        image_array = np.array(fig.canvas.renderer.buffer_rgba())
        fig.clear() 

        return image_array

    def make_sonar_data(self, pcl:np.ndarray, normals:np.ndarray, viewTransform:np.ndarray) -> np.ndarray:
        
        
        def arctan_with_quadrants(y, x):
            # Compute arctan for the ratio y/x
            angle = np.arctan(np.divide(y, x, where=x != 0))  # Avoid division by zero with `where`
            
            # Adjust angles based on the quadrant
            angle = np.where((x > 0), angle, angle + np.pi)  # Quadrants II and III
            angle = np.where((x < 0) & (y < 0), angle - 2 * np.pi, angle)  # Quadrant III correction
            angle = np.where((x == 0) & (y > 0), np.pi / 2, angle)  # Positive y-axis
            angle = np.where((x == 0) & (y < 0), -np.pi / 2, angle)  # Negative y-axis

            return angle
        

        def cartesian_to_spherical(cart_coords):
            x, y, z = cart_coords[:, 0], cart_coords[:, 1], cart_coords[:, 2]
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = arctan_with_quadrants(y, x)
            phi = np.arccos(z / r)
            return np.vstack((r, theta, phi)).T


        def bin_intensity(num_r_bins, num_azi_bins, pcl, intensity):
            min_r = pcl[:,0].min()
            max_r = pcl[:,0].max()
            min_azi = pcl[:,1].min()
            max_azi = pcl[:,1].max()
            r_bins = np.linspace(min_r, max_r, num_r_bins, endpoint=True)
            azi_bins = np.linspace(min_azi, max_azi, num_azi_bins, endpoint=True)

            intensity_binned, r_edges, azi_edges, _ = binned_statistic_2d(pcl[:,0], pcl[:,1], intensity, statistic='mean', bins=[r_bins, azi_bins])
            r_mid = (r_edges[:-1] + r_edges[1:]) / 2  
            azi_mid = (azi_edges[:-1] + azi_edges[1:]) / 2
            r, azi = np.meshgrid(r_mid, azi_mid, indexing='ij')
            
            return np.stack((r, azi, intensity_binned), axis=-1).reshape(-1,3)

        
        
        



        normals = np.delete(arr=normals, obj=3, axis=1)
        viewTransform = viewTransform.reshape(4,4).T
        render_trans = -(np.transpose(viewTransform)[:3,3])
        render_rot = np.transpose(viewTransform)[:3,:3]
        dist = np.linalg.norm(pcl-render_trans, axis=1)
        directs = pcl - render_trans
        unit_directs = directs/np.linalg.norm(directs)

        theta = np.arccos(np.sum(unit_directs * normals, axis=1))
        intensity = self.base_intensity * self.reflectivity * np.abs(np.cos(theta)) * (1/self.max_range)**2 * np.exp(-self.attenuation * 2 * dist)

        # Pre-multiplication to produce transform with respect to world frame
        pcl_local = (viewTransform @ np.hstack((pcl, np.ones([pcl.shape[0], 1]))).T).T 
        # Change the axis location to make z pointing upwards  and x pointing forwards for spherical coordinate transformation
        pcl_local = np.delete(pcl_local, obj=3, axis=1) @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])


        pcl_spher_local = cartesian_to_spherical(pcl_local)
        sonar_map = bin_intensity(1024, 1024, pcl_spher_local, intensity)
        sonar_map = np.array([sonar_map[:,0] * np.cos(sonar_map[:,1]), 
                            sonar_map[:,0] * np.sin(sonar_map[:,1]),
                            sonar_map[:,2]]).T
        
        return sonar_map
    


WriterRegistry.register(imagingSonarWriter)