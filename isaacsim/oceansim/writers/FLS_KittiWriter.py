import csv
import io
import json
import os
from typing import List

import carb
import numpy as np
from omni.syntheticdata.scripts.SyntheticData import SyntheticData
import omni.replicator.core.scripts.functional as F
from omni.replicator.core import AnnotatorRegistry, BackendDispatch
from omni.replicator.core.scripts.writers import Writer
from isaacsim.replicator.writers.scripts.utils import calculate_truncation_ratio_simple

# Import sonar rendering kernel
from isaacsim.oceansim.utils.ImagingSonar_kernels import *
from pxr import Gf
import isaacsim.core.utils.rotations as rotations_utils

from PIL import Image, ImageDraw
from functools import partial
import random


__version__ = "0.0.1"


class FLS_KittiWriter(Writer):
    """
    .. my note::
        - occlusion in sonar image can not be achieved now 
        (future: process occlusion can only be done without parallelism ie. can not be on cuda, 
        otherwise to do in cuda, we have to initialize a really big memory by extending one dimention equal to the total number of semantics )
        - dimensions and locations of the object should be possible. 
        by Transferring 3D bbox information to local frame 
    """

    def __init__(
        self,
        output_dir: str,
        # configuration of sonar renderings
        sonar_param: dict = {
                    "max_range": 3, 
                    "min_range": 0.2,
                    "hori_fov": 130, # Notice: on camera end, hori_fov and vert_fov is required to 
                    "vert_fov": 20,  # compute camera AR and vert_res given arbitrary hori_res
                    "range_res": 0.005, 
                    "angular_res": 0.25,
                    "normalizing_method": "range",
                    "query_prop": "reflectivity", 
                    "attenuation": 0.1,
                    "gau_noise_param": 0.2,
                    "ray_noise_param": 0.05,
                    "intensity_offset": 0.0,
                    "intensity_gain": 1.0,
                    "central_peak": 2,
                    "central_std": 0.001,
                    "hori_res": 5000
                    },
        # extra config for data writing
        s3_bucket: str = None,
        s3_region: str = None,
        s3_endpoint: str = None,
        semantic_types: List[str] = None,
        omit_semantic_type: bool = False,
        mapping_path: str = None,
        mapping_dict: dict = None,
        colorize_instance_segmentation: bool = True,
        include_unlabelled: bool = True,
        semantic_filter_predicate: str = None,
        use_kitti_dir_names: bool = False,
        cuboid_keypoints_order: list = ["Center", "LDB", "LDF", "LUB", "LUF", "RDB", "RDF", "RUB", "RUF"],
        debug_mode: bool = False,
    ):
        self.version = __version__
        self._frame_id = 0
        if s3_bucket:
            self._backend = BackendDispatch(
                output_dir=output_dir,  # Kept to maintain previous behaviour
                key_prefix=output_dir,
                bucket=s3_bucket,
                region=s3_region,
                endpoint_url=s3_endpoint,
            )
        else:
            self._backend = BackendDispatch(output_dir=output_dir)
        self.backend = self._backend
        self._omit_semantic_type = omit_semantic_type
        self._use_kitti_dir_names = use_kitti_dir_names
        self.colorize_instance_segmentation = colorize_instance_segmentation
        self.include_unlabelled = include_unlabelled
        self._device = str(wp.get_preferred_device())
        self._debug_mode = debug_mode
        self._cuboid_keypoints_order = cuboid_keypoints_order

        if debug_mode:
            self._CUBOID_KEYPOINT_COLORS = ["white", "red", "green", "blue", "yellow", "cyan", "magenta", "orange", "purple"]
            self._CUBOID_EDGE_COLORS = {"front": "red", "back": "blue", "connecting": "green"}
            self._debug_data = {}

        if mapping_path and mapping_dict:
            raise ValueError("Cannot have both mapping_path and mapping_dict specified")
        elif mapping_path:
            self.mapping_dict = self._procure_labels_from_json(mapping_path)
            carb.log_info(f"Using label mapping from {mapping_path}")
        elif mapping_dict and isinstance(mapping_dict, dict):
            self.mapping_dict = mapping_dict
            carb.log_info("Using label mapping from provided dictionary")
        else:
            self.mapping_dict = {
                    "UNLABELLED": 0,
                    "BACKGROUND": 1,
            }
            carb.log_info("Using default KITTI label mapping")

        # Specify the semantic types that will be included in output
        if semantic_types is not None:
            if semantic_filter_predicate is None:
                semantic_filter_predicate = ":*; ".join(semantic_types) + ":*"
            else:
                raise ValueError(
                    "`semantic_types` and `semantic_filter_predicate` are mutually exclusive. Please choose only one."
                )
        elif semantic_filter_predicate is None:
            semantic_filter_predicate = "class:*"

        if semantic_filter_predicate is not None:
            SyntheticData.Get().set_instance_mapping_semantic_filter(semantic_filter_predicate)

        self.annotators = [
            # We don't need these three annotators for they are for camera rendering 
            "rgb",
            # We need pointcloud data as the result of rayquest
            AnnotatorRegistry.get_annotator(
                "pointcloud", init_params={"includeUnlabelled": include_unlabelled}, device=self._device
            ),
            AnnotatorRegistry.get_annotator(
                "semantic_segmentation", init_params={"mapping": self._get_anno_semantic_mapping()}
            ),
            AnnotatorRegistry.get_annotator(
                "instance_segmentation_fast", init_params={"colorize": colorize_instance_segmentation}
            ),
            "camera_params",
            "bounding_box_3d_fast"
        ]

        self._initialize_sonar_renderer(sonar_param)


    def _initialize_sonar_renderer(self, sonar_param:dict):
        '''Takes in sonar parameters to allocate memory on cuda and load up kernels.
        Must call this function before rendering sonar images.
        Args:
            sonar_param (dict) : Sonar parameters
        '''
        # Boilerplate to intake params
        # Below params are used to define the sonar grid
        self.max_range = sonar_param['max_range']
        self.min_range = sonar_param['min_range']
        self.range_res = sonar_param['range_res']
        self.hori_fov = sonar_param['hori_fov']
        self.vert_fov = sonar_param['vert_fov']
        self.angular_res = np.deg2rad(sonar_param['angular_res'])
        # Below params are used to define sonar noise and rendering method
        self.query_prop = sonar_param['query_prop']
        self.attenuation = sonar_param['attenuation']
        self.gau_noise_param = sonar_param['gau_noise_param']
        self.ray_noise_param = sonar_param['ray_noise_param']
        self.intensity_offset = sonar_param['intensity_offset']
        self.intensity_gain = sonar_param['intensity_gain']
        self.central_peak = sonar_param['central_peak']
        self.central_std = sonar_param['central_std']

        # Allocate memeory on cuda
        
        # Generate sonar grid  r and azi meshgrid
        self.min_azi = np.deg2rad(90-self.hori_fov/2)
        r, azi = np.meshgrid(np.arange(self.min_range,self.max_range,self.range_res),
                                       np.arange(np.deg2rad(90-self.hori_fov/2), np.deg2rad(90+self.hori_fov/2), self.angular_res),
                                       indexing='ij')
        self.r = wp.array(r, shape=r.shape, dtype=wp.float32)
        self.azi = wp.array(azi, shape=r.shape, dtype=wp.float32)
       
        # Accumulated intensity per bin
        self.bin_sum = wp.empty(shape=self.r.shape, dtype=wp.float32)
        # Accumulated ray count per bin
        self.bin_count = wp.empty(shape=self.r.shape, dtype=wp.int32)
        # Minimum zenith bookkeeper to only keep the highest ray semantics
        self.bin_min_zenith = wp.full(shape=self.r.shape, value=wp.PI, dtype=wp.float32)
        # semantics per bin (only support 1 semantics per bin, no occlusion)
        self.bin_semantics = wp.empty(shape=self.r.shape, dtype=wp.uint8)
        # Instance per bin
        self.bin_instances = wp.empty(shape=self.r.shape, dtype=wp.uint8)
        # Resulted sonar data per bin (cartesian_x of bin in local frame, cartesian_y, normalized intensity value)
        self.sonar_data = wp.empty(shape=self.r.shape, dtype=wp.vec3)
        # Rendered sonar image
        self.sonar_image = wp.empty(shape=(self.r.shape[0], self.r.shape[1], 4), dtype=wp.uint8)
        # Rendered sonar semantics
        self.sonar_semantics_image = wp.empty(shape=(self.r.shape[0], self.r.shape[1], 4), dtype=wp.uint8)
        # Corresponding kernel for different method to nomalize 
        if sonar_param['normalizing_method'] == "all":
            self._max_intensity = wp.zeros(shape=(1,), dtype=wp.float32)
            self._compute_max_intensity = compute_max_intensity_all
            self._make_sonar_map = make_sonar_map_all
        elif sonar_param['normalizing_method'] == "range":
            self._max_intensity = wp.zeros(shape=(self.r.shape[0],), dtype=wp.float32)
            self._compute_max_intensity = compute_max_intensity_range
            self._make_sonar_map = make_sonar_map_range      
        
        # Sonar noise
        self.gau_noise = wp.empty(shape=self.r.shape, dtype=wp.float32)
        self.range_dependent_ray_noise = wp.empty(shape=self.r.shape, dtype=wp.float32)
        
        # Sonar grid information
        self.sonar_grid = sonarGrid()
        self.sonar_grid.x_offset = self.min_range
        self.sonar_grid.y_offset = self.min_azi
        self.sonar_grid.x_res = self.range_res
        self.sonar_grid.y_res = self.angular_res
        self.sonar_grid.x_num = self.r.shape[0]
        self.sonar_grid.y_num = self.r.shape[1]
        

    
    def _render_sonar(self, 
                      data, 
                      pointcloud_annot: str, 
                      cameraParams_annot: str, 
                      semantic_seg_annot: str):

        pcl = data[pointcloud_annot]["data"][0]  # shape :(1,N,3) <class 'warp.types.array'>
        normals = data[pointcloud_annot]['info']['pointNormals'][0] # shape :(1,N,4) <class 'warp.types.array'>
        semantics = data[pointcloud_annot]['info']['pointSemantic'][0] # shape: (1, N) <class 'warp.types.array'>
        instances = data[pointcloud_annot]['info']['pointInstance'][0] # shape: (1, N) <class 'warp.types.array'>
        viewTransform = data[cameraParams_annot]['cameraViewTransform'].reshape(4,4).T # 4 by 4 np.ndarray extrinsic matrix
        idToLabels = data[semantic_seg_annot]["info"]["idToLabels"] # dict 
        def make_indexToProp_array(idToLabels: dict, query_property: str) -> np.ndarray:
            """ A utility function helps to convert idToLabels into indexToProp array
            This manipulation facilitates warp computation framework
            indexToProp is an 1-dim array where the values associated with the query property 
            are placed at the index corresponding to the key
            First two entry are always zero because {'0': {'class': 'BACKGROUND'}, '1': {'class': 'UNLABELLED'}}
            eg: indexToProp = [0, 0, 0.1, 1 .....] 
            """
            max_id = max(idToLabels.keys(), default=-1)
            # TODO we can initilize a big chunk of memory (>number of objects) for indexToPro_array 
            # in the beginning and overwrite during looping 
            indexToProp_array = np.ones((int(max_id)+1,))
            for id in idToLabels.keys():
                for property in idToLabels.get(id):
                    if property == query_property:
                        indexToProp_array[int(id)] = idToLabels.get(id).get(property)
            return indexToProp_array

        num_points = pcl.shape[0]
        # Load these small numpy arrays to cuda
        indexToRefl_np = make_indexToProp_array(idToLabels=idToLabels,
                                                query_property=self.query_prop)
        indexToRefl = wp.array(data=indexToRefl_np, dtype=wp.float32)
        viewTransform = wp.mat44(viewTransform)
        
        
        # Compute intensity for each ray query     
        intensity = wp.empty(shape=(num_points,), dtype=wp.float32)
        wp.launch(kernel=compute_intensity,
                dim=num_points,
                inputs=[
                    pcl,
                    normals,
                    viewTransform,
                    semantics,
                    indexToRefl,
                    self.attenuation,
                ],
                outputs=[
                    intensity
                ]
                )
                
        # Transform pointcloud from world cooridates to sonar local and convert to spherical coord
        pcl_bin_idx = wp.empty(shape=(num_points, ), dtype=wp.vec2ui)
        pcl_local_spher = wp.empty(shape=(num_points,), dtype=wp.vec3) 
        wp.launch(kernel=world2local,
                dim=num_points,
                inputs=[
                    viewTransform,
                    pcl
                ],
                    outputs=[
                    pcl_local_spher
                    ]
                )
        # Collapse three dimensional intensity data to 2D
        # Simply sum intensity return and compute number of return that falls into the same bin
        # Zero out intensity in each bin (do not omit this, this is necessary)
        self.bin_sum.zero_()
        self.bin_count.zero_()

        self.bin_min_zenith.fill_(wp.PI)
        self.bin_semantics.zero_()
        self.bin_instances.zero_()
        wp.launch(kernel=bin_process,
                dim=num_points,
                inputs=[
                    pcl_local_spher,
                    intensity,
                    semantics,
                    self.sonar_grid
                ],
                outputs=[
                    self.bin_sum,
                    self.bin_count,
                    pcl_bin_idx,
                    self.bin_min_zenith
                ]
                )
        
        wp.launch(kernel=bin_segmentation_process,
                dim=num_points,
                inputs=[
                    pcl_local_spher,
                    semantics,
                    instances,
                    pcl_bin_idx,
                    self.bin_min_zenith
                ],
                outputs=[
                    self.bin_semantics,
                    self.bin_instances
                ]
                )
        
        # Calculate multiplicative gaussian noise
        
        wp.launch(
            kernel=normal_2d,
            dim=self.bin_sum.shape,
            inputs=[
                self._frame_id,   # use frame id for RNG seed increment
                0.0,
                self.gau_noise_param
            ],
            outputs=[
                self.gau_noise
            ]
        )

        # Calculate additive rayleigh noise (range dependent and mimic central beam)

        wp.launch(
            kernel=range_dependent_rayleigh_2d,
            dim=self.bin_sum.shape,
            inputs=[
                self._frame_id,   # use frame num for RNG seed increment
                self.r,
                self.azi,
                self.max_range,
                self.ray_noise_param,
                self.central_peak,
                self.central_std,
            ],
            outputs=[
                self.range_dependent_ray_noise

            ]
        )

        
        self._max_intensity.fill_(-wp.inf)
        # Normalizing intensity at each bin either by global maximum or rangewise maximum
        wp.launch(
            dim=self.bin_sum.shape,
            kernel=self._compute_max_intensity,
            inputs=[
                self.bin_sum,
            ],
            outputs=[
                self._max_intensity 
            ]
        )

        # Apply noise, normalize, and convert (r, azi) to (x,y) for plotting
        wp.launch(
            kernel=self._make_sonar_map,
            dim=self.sonar_data.shape,
            inputs=[
                self.r,
                self.azi,
                self.bin_sum,
                self._max_intensity,
                self.gau_noise,
                self.range_dependent_ray_noise,
                self.intensity_offset,
                self.intensity_gain
            ],
            outputs=[
                self.sonar_data
            ]
            )
    
        wp.launch(
            dim=self.sonar_data.shape,
            kernel=make_sonar_image,
            inputs=[
                self.sonar_data
            ],
            outputs=[
                self.sonar_image
            ]
        )
        wp.synchronize()

    def _write_sonar_image(self, sub_dir: str):
        sonar_dir_name = "sonar_image_02" if self._use_kitti_dir_names else "sonar_image"
        sonar_file_path = os.path.join(sub_dir, sonar_dir_name, f"{self._frame_id}.png")
        self._backend.schedule(F.write_image, data=self.sonar_image, path=sonar_file_path)



    def _get_anno_semantic_mapping(self):
        anno_semantic_mapping = {}
        for k, v in self.mapping_dict.items():
            is_valid_id = isinstance(v, int)
            is_valid_colour = isinstance(v, (list, tuple)) and len(v) == 4 and all(isinstance(e, int) for e in v)
            if not is_valid_id and not is_valid_colour:
                raise ValueError(
                    f"Provided mapping maps to invalid values. All target values must be an integer ID or integer RGBA values"
                )
            if ":" in k:
                anno_semantic_mapping[k] = v
            else:
                # fallback on `class` semantic type
                anno_semantic_mapping[f"class:{k}"] = v
        return json.dumps(anno_semantic_mapping)


    def _write_object_pose(self, data, sub_dir: str, bbox_3d_annotator: str, camera_param_annotator: str):
        objs_data = self._process_bounding_boxes_3d(data[bbox_3d_annotator], data[camera_param_annotator])
        pose_dir_name = "pose_02" if self._use_kitti_dir_names else "pose"
        pose_file_path = os.path.join(sub_dir, pose_dir_name, f"{self._frame_id}.json")
        self._backend.schedule(F.write_json, path=pose_file_path, data=objs_data, indent=2)

    
    
    def _process_bounding_box_3d_single(self, bounding_box_3d_data: dict, camera_params: dict) ->dict :
        bbox = bounding_box_3d_data
        obj = {}
        # `occlusionRatio` represents (visible pixels / total pixels) where `0.0` is fully visible and `1.0` is fully occluded
        # NOTE: `obj_visibility` is inverted to match the format where `0.0` is fully occluded and `1.0`` is fully visible
        obj_visibility = 1.0 - abs(float(bbox["occlusionRatio"]))


        obj["visibility"] = round(obj_visibility, 3)

        # Local space to to world transform (row-major)
        local_to_world_tf = bbox["transform"]

        obj["local_to_world_transform"] = local_to_world_tf.tolist()
        # Extract world frame location (last row) and rotation matrix (3x3) from the row-major transform matrix
        location_world_frame = local_to_world_tf[3, :3]
        obj["location_world_frame"] = location_world_frame.tolist()
        rotation_matrix_world_frame = local_to_world_tf[:3, :3]
        obj["rotation_matrix_world_frame"] = rotation_matrix_world_frame.tolist()

        # Get the world frame quaternion using Gf.Transform (row-major)
        local_to_world_tf_gf = Gf.Transform()
        local_to_world_tf_gf.SetMatrix(Gf.Matrix4d(local_to_world_tf.tolist()))
        quat_world_frame_gf = local_to_world_tf_gf.GetRotation().GetQuat()
        obj["quat_wxyz_world_frame"] = [quat_world_frame_gf.GetReal()] + list(
            quat_world_frame_gf.GetImaginary()
        )

        # World to camera transform (row-major) (transform a point from world coordinate to camera coordinate)
        world_to_camera_tf = camera_params["cameraViewTransform"].reshape(4, 4)
        # Object world space to camera frame transform (row-major matrix multiplication)
        obj_to_camera_tf = local_to_world_tf @ world_to_camera_tf
        # Extract camera frame location (last row) and rotation matrix (3x3) from the row-major transform matrix
        location_camera_frame = obj_to_camera_tf[3, :3]
        obj["location_camera_frame"] = location_camera_frame.tolist()
        rotation_matrix_camera_frame = obj_to_camera_tf[:3, :3]
        obj["rotation_matrix_camera_frame"] = rotation_matrix_camera_frame.tolist()
        # Get the camera frame quaternion using Gf.Transform (row-major)
        obj_to_camera_tf_gf = Gf.Transform()
        obj_to_camera_tf_gf.SetMatrix(Gf.Matrix4d(obj_to_camera_tf.tolist()))
        quat_camera_frame_gf = obj_to_camera_tf_gf.GetRotation().GetQuat()
        obj["quat_wxyz_camera_frame"] = [quat_camera_frame_gf.GetReal()] + list(
            quat_camera_frame_gf.GetImaginary()
        )
        # yaw is the angle between the object local forward X direction and the camera rightward X direction [-pi, pi]
        row, pitch, yaw = rotations_utils.quat_to_euler_angles(np.array(obj["quat_wxyz_camera_frame"]), extrinsic=False)
        # which is rotation_y in Kitti format (NOTE: this is the rotation with respect to the camera frame, Y up)
        rotation_y = yaw
        # α measures the object’s orientation relative to camera’s observation angle towards the center of the object
        # NOTE: we flip the z-axis because camera's observation angle is -z axis, for x we dont need to do anything
        alpha = rotation_y - np.arctan2(location_camera_frame[0], -location_camera_frame[2])
        # Normalize the alpha to [-pi, pi]
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        obj["rotation_y"] = float(rotation_y)
        obj["alpha"] = float(alpha)

        # Size of the object before scale (NOTE: scale is not applied yet to objects in local frame)
        min_local = np.array([bbox["x_min"], bbox["y_min"], bbox["z_min"], 1])
        max_local = np.array([bbox["x_max"], bbox["y_max"], bbox["z_max"], 1])
        size_local = np.abs(max_local - min_local)[:3].tolist()
        center_local = min_local + (max_local - min_local) / 2

        # Cuboid keypoints in local frame
        keypoints_local = {
            "Center": center_local,
            "LDB": np.array([bbox["x_min"], bbox["y_min"], bbox["z_min"], 1]),  # Left-Down-Back
            "LDF": np.array([bbox["x_min"], bbox["y_min"], bbox["z_max"], 1]),  # Left-Down-Front
            "LUB": np.array([bbox["x_min"], bbox["y_max"], bbox["z_min"], 1]),  # Left-Up-Back
            "LUF": np.array([bbox["x_min"], bbox["y_max"], bbox["z_max"], 1]),  # Left-Up-Front
            "RDB": np.array([bbox["x_max"], bbox["y_min"], bbox["z_min"], 1]),  # Right-Down-Back
            "RDF": np.array([bbox["x_max"], bbox["y_min"], bbox["z_max"], 1]),  # Right-Down-Front
            "RUB": np.array([bbox["x_max"], bbox["y_max"], bbox["z_min"], 1]),  # Right-Up-Back
            "RUF": np.array([bbox["x_max"], bbox["y_max"], bbox["z_max"], 1]),  # Right-Up-Front
        }

        # Transform the cuboid keypoints from local to world frame in the given order
        keypoints_world_ordered = [keypoints_local[k] @ local_to_world_tf for k in self._cuboid_keypoints_order]
        obj["cuboid_keypoints_world_frame"] = [point[:3].tolist() for point in keypoints_world_ordered]
        # Transform the cuboid keypoints from world to camera frame
        keypoints_camera_ordered = [point @ world_to_camera_tf for point in keypoints_world_ordered]
        obj["cuboid_keypoints_camera_frame"] = [point[:3].tolist() for point in keypoints_camera_ordered]
        
        obj["center_camera_frame"] = keypoints_camera_ordered[0][:3].tolist()

        # Calculate the (scaled) size of the object from its world bounds (NOTE: scale is applied through the transform)
        all_world_keypoints = np.vstack(keypoints_world_ordered)
        min_world = np.min(all_world_keypoints[:, :3], axis=0)
        max_world = np.max(all_world_keypoints[:, :3], axis=0)
        size_world = np.abs(max_world - min_world).tolist()

        obj["size_world"] = size_world
        obj["size_local"] = size_local


        # Get the camera projection matrix and screen size to project the cuboid keypoints to screen space
        cam_projection_tf = camera_params["cameraProjection"].reshape((4, 4))
        screen_size = camera_params["renderProductResolution"]
        keypoints_projected_ordered = [
            self._project_camera_point_to_screen(point, cam_projection_tf, screen_size)
            for point in keypoints_camera_ordered
        ]
        obj["cuboid_keypoints_projected"] = keypoints_projected_ordered

        obj["truncation_ratio"] = calculate_truncation_ratio_simple(
            keypoints_projected_ordered, screen_size[0], screen_size[1]
        )

        return obj

    
    def _process_bounding_boxes_3d(self, bounding_box_3d: dict, camera_params: dict) -> list:
        # Map the ids to class names from the bbox annotator "idToLabels" data
        # ('idToLabels': {0: {'class': 'cube'}, 1: {'class': 'sphere'}} -> {0: 'cube', 1: 'sphere'})
        id_to_labels = {k: v["class"] for k, v in bounding_box_3d["info"]["idToLabels"].items()}

        if self._debug_mode:
            self._debug_data["world_frame_transforms"] = []
            self._debug_data["projected_keypoints"] = []
            self._debug_data["size_local"] = []
            self._debug_data["center_local"] = []
        # Iterate the bounding box data and extract the object informations
        objs = []
        for i, bbox in enumerate(bounding_box_3d["data"]):
            obj = {}
            # `occlusionRatio` represents (visible pixels / total pixels) where `0.0` is fully visible and `1.0` is fully occluded
            # NOTE: `obj_visibility` is inverted to match the format where `0.0` is fully occluded and `1.0`` is fully visible
            obj_visibility = 1.0 - abs(float(bbox["occlusionRatio"]))

            obj["label"] = id_to_labels[bbox["semanticId"]]
            # Skip ground objects
            if obj["label"] == "ground":
                continue
            obj["prim_path"] = bounding_box_3d["info"]["primPaths"][i]
            obj["visibility"] = round(obj_visibility, 3)

            # Local space to to world transform (row-major)
            local_to_world_tf = bbox["transform"]

            obj["local_to_world_transform"] = local_to_world_tf.tolist()
            # Extract world frame location (last row) and rotation matrix (3x3) from the row-major transform matrix
            location_world_frame = local_to_world_tf[3, :3]
            obj["location_world_frame"] = location_world_frame.tolist()
            rotation_matrix_world_frame = local_to_world_tf[:3, :3]
            obj["rotation_matrix_world_frame"] = rotation_matrix_world_frame.tolist()

            # Get the world frame quaternion using Gf.Transform (row-major)
            local_to_world_tf_gf = Gf.Transform()
            local_to_world_tf_gf.SetMatrix(Gf.Matrix4d(local_to_world_tf.tolist()))
            quat_world_frame_gf = local_to_world_tf_gf.GetRotation().GetQuat()
            obj["quat_wxyz_world_frame"] = [quat_world_frame_gf.GetReal()] + list(
                quat_world_frame_gf.GetImaginary()
            )
            if self._debug_mode:
                self._debug_data["world_frame_transforms"].append(local_to_world_tf)

            # World to camera transform (row-major) (transform a point from world coordinate to camera coordinate)
            world_to_camera_tf = camera_params["cameraViewTransform"].reshape(4, 4)
            # Object world space to camera frame transform (row-major matrix multiplication)
            obj_to_camera_tf = local_to_world_tf @ world_to_camera_tf
            # Extract camera frame location (last row) and rotation matrix (3x3) from the row-major transform matrix
            location_camera_frame = obj_to_camera_tf[3, :3]
            obj["location_camera_frame"] = location_camera_frame.tolist()

            rotation_matrix_camera_frame = obj_to_camera_tf[:3, :3]
            obj["rotation_matrix_camera_frame"] = rotation_matrix_camera_frame.tolist()
            # Get the camera frame quaternion using Gf.Transform (row-major)
            obj_to_camera_tf_gf = Gf.Transform()
            obj_to_camera_tf_gf.SetMatrix(Gf.Matrix4d(obj_to_camera_tf.tolist()))
            quat_camera_frame_gf = obj_to_camera_tf_gf.GetRotation().GetQuat()
            obj["quat_wxyz_camera_frame"] = [quat_camera_frame_gf.GetReal()] + list(
                quat_camera_frame_gf.GetImaginary()
            )

            # Size of the object before scale (NOTE: scale is not applied yet to objects in local frame)
            min_local = np.array([bbox["x_min"], bbox["y_min"], bbox["z_min"], 1])
            max_local = np.array([bbox["x_max"], bbox["y_max"], bbox["z_max"], 1])
            size_local = np.abs(max_local - min_local)[:3].tolist()
            center_local = min_local + (max_local - min_local) / 2
            if self._debug_mode:
                self._debug_data["size_local"].append(size_local)
                self._debug_data["center_local"].append(center_local[:3].tolist())

            # Cuboid keypoints in local frame
            keypoints_local = {
                "Center": center_local,
                "LDB": np.array([bbox["x_min"], bbox["y_min"], bbox["z_min"], 1]),  # Left-Down-Back
                "LDF": np.array([bbox["x_min"], bbox["y_min"], bbox["z_max"], 1]),  # Left-Down-Front
                "LUB": np.array([bbox["x_min"], bbox["y_max"], bbox["z_min"], 1]),  # Left-Up-Back
                "LUF": np.array([bbox["x_min"], bbox["y_max"], bbox["z_max"], 1]),  # Left-Up-Front
                "RDB": np.array([bbox["x_max"], bbox["y_min"], bbox["z_min"], 1]),  # Right-Down-Back
                "RDF": np.array([bbox["x_max"], bbox["y_min"], bbox["z_max"], 1]),  # Right-Down-Front
                "RUB": np.array([bbox["x_max"], bbox["y_max"], bbox["z_min"], 1]),  # Right-Up-Back
                "RUF": np.array([bbox["x_max"], bbox["y_max"], bbox["z_max"], 1]),  # Right-Up-Front
            }



            # Transform the cuboid keypoints from local to world frame in the given order
            keypoints_world_ordered = [keypoints_local[k] @ local_to_world_tf for k in self._cuboid_keypoints_order]
            obj["cuboid_keypoints_world_frame"] = [point[:3].tolist() for point in keypoints_world_ordered]
            # Transform the cuboid keypoints from world to camera frame
            keypoints_camera_ordered = [point @ world_to_camera_tf for point in keypoints_world_ordered]
            obj["cuboid_keypoints_camera_frame"] = [point[:3].tolist() for point in keypoints_camera_ordered]
            
            obj["center_camera_frame"] = keypoints_camera_ordered[0][:3].tolist()


            # Calculate the (scaled) size of the object from its world bounds (NOTE: scale is applied through the transform)
            all_world_keypoints = np.vstack(keypoints_world_ordered)
            min_world = np.min(all_world_keypoints[:, :3], axis=0)
            max_world = np.max(all_world_keypoints[:, :3], axis=0)
            size_world = np.abs(max_world - min_world).tolist()

            obj["size_world"] = size_world
            obj["size_local"] = size_local


            # Get the camera projection matrix and screen size to project the cuboid keypoints to screen space
            cam_projection_tf = camera_params["cameraProjection"].reshape((4, 4))
            screen_size = camera_params["renderProductResolution"]
            keypoints_projected_ordered = [
                self._project_camera_point_to_screen(point, cam_projection_tf, screen_size)
                for point in keypoints_camera_ordered
            ]
            obj["cuboid_keypoints_projected"] = keypoints_projected_ordered

            if self._debug_mode:
                self._debug_data["projected_keypoints"].append(keypoints_projected_ordered)

            obj["truncation_ratio"] = calculate_truncation_ratio_simple(
                keypoints_projected_ordered, screen_size[0], screen_size[1]
            )

            objs.append(obj)

        return objs


    # Project a 3D point from camera coordinates to 2D screen coordinates
    def _project_camera_point_to_screen(self, camera_point, projection_matrix, screen_size):
        # Apply the projection matrix to project to screen coordinates
        point_screen = camera_point @ projection_matrix

        # Normalize to NDC (Normalized Device Coordinates) by dividing x, y, z, by w: (x, y, z, w) -> (x/w, y/w, z/w, 1)
        point_screen_normalized = point_screen / point_screen[3]

        # Map NDC to screen coordinates. Adjust x and y for screen dimensions, flipping y to match screen's coordinate system.
        x = (point_screen_normalized[0] + 1) * screen_size[0] / 2
        y = (1 - point_screen_normalized[1]) * screen_size[1] / 2

        return round(x), round(y)
    
    
    # Procuring standard KITTI Labels for objects annotated in the KITTI-format
    # The dictionary is ordered where label idx corresponds to semantic ID
    # See https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    def _procure_labels_from_json(self, json_path):
        with open(json_path, "r") as f:
            labels_dict = json.load(f)
        return labels_dict

    def _write_sonar_segmentation(self, data, sub_dir: str, inst_annotator: str):
        """
        Instance segmentation follows the format specified here: https://www.vision.rwth-aachen.de/page/mots
        """
        inst_dir_name = "instance" if self._use_kitti_dir_names else "instance_segmentation"
        seg_filepath = os.path.join(sub_dir, "semantic_segmentation", f"{self._frame_id}.png")
        seg_mapping_filepath = os.path.join(sub_dir, "semantic_mapping.json")

        inst_filepath = os.path.join(sub_dir, inst_dir_name, f"{self._frame_id}.png")

        inst_id_to_labels = data[inst_annotator]["info"]["idToSemantics"]
        self._backend.schedule(F.write_json, data=self.mapping_dict, path=seg_mapping_filepath)

        inst_seg = self.bin_instances.numpy()
        height, width = inst_seg.shape[:2]

        if self.colorize_instance_segmentation:
            inst_col_filepath = os.path.join(sub_dir, "instance_rgb", f"{self._frame_id}.png")
            # inst_seg_img_colorized = inst_seg_img.view(np.uint8)
            # inst_seg_img_colorized = inst_seg_img_colorized.reshape(height, width, -1)
            self._backend.schedule(F.write_image, data=inst_seg, path=inst_col_filepath)

        # Re-label instances to be sequentially numbered
        # The instance segmentation is a 16bit png where the lower 8 bit contain the semantic ID and the higher 8 bits
        # contain the instance ID
        # Semantic segmentation is saved as a 3 channel image where each channel is the same 8 bit semantic ID
        # Instance IDs start from 1
        cur_idx = {}
        if self.colorize_instance_segmentation:
            # convert ids to uint32
            inst_id_to_labels = {
                (iid[0] | iid[1] << 8 | iid[2] << 16 | iid[3] << 24): v for iid, v in inst_id_to_labels.items()
            }

        instance_ids = list(inst_id_to_labels.keys())
        inst_seg_img_renumbered = np.zeros((height, width), dtype=np.uint16)
        sem_seg_img_renumbered = np.zeros((height, width), dtype=np.uint8)
        for i, iid in enumerate(instance_ids):
            semantic_class = inst_id_to_labels[iid].get("class", "unlabelled")
            is_unlabelled = semantic_class.lower() == "unlabelled"
            is_background = semantic_class.lower() == "background"
            is_in_mapping = semantic_class in self.mapping_dict
            if not is_in_mapping or is_unlabelled or is_background:
                inst_seg_img_renumbered[inst_seg == iid] = 0
            else:
                cur_semantics = str(inst_id_to_labels[iid])
                cur_idx.setdefault(cur_semantics, 0)
                cur_idx[cur_semantics] += 1
                semantics_renumbered = self.mapping_dict.get(semantic_class, 0)
                inst_seg_img_renumbered[inst_seg == iid] = cur_idx[cur_semantics] + semantics_renumbered * 256
                sem_seg_img_renumbered[inst_seg == iid] = semantics_renumbered

        self._backend.schedule(F.write_image, data=inst_seg_img_renumbered, path=inst_filepath)
        self._backend.schedule(F.write_image, data=sem_seg_img_renumbered, path=seg_filepath)
    
        
    
    def _write_camera_param(self, data, sub_dir: str, annotator:str):
        
        camera_params = data[annotator]
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

        camera_data["width"] = camera_params["renderProductResolution"].tolist()[0]
        camera_data["height"] = camera_params["renderProductResolution"].tolist()[1]
        # Debug data needed for the overlay projections
        if self._debug_mode:
            self._debug_data["camera_projection_matrix"] = camera_params["cameraProjection"].reshape(4, 4)
            self._debug_data["camera_view_matrix"] = camera_params["cameraViewTransform"].reshape(4, 4)
            self._debug_data["resolution"] = camera_params["renderProductResolution"]

        file_path = os.path.join(sub_dir, "camera_param", f"{self._frame_id}.json")
        self._backend.schedule(F.write_json, data=camera_data, path=file_path)

    def write(self, data):
        render_products = [k for k in data.keys() if k.startswith("rp_")]
        if len(render_products) == 1:
            sub_dir = data[render_products[0]]["camera"].split("/")[-1]
            self._render_sonar(data, "pointcloud", "camera_params", "semantic_segmentation")
            self._write_sonar_image(sub_dir)
            self._write_sonar_segmentation(data, sub_dir,  "instance_segmentation_fast")
            self._write_camera_param(data, sub_dir, "camera_params")

            if self._debug_mode:
                self._debug_data["raw_rgb"] = data["rgb"]
                self._write_object_pose(data, sub_dir, "bounding_box_3d_fast", "camera_params")
                self._write_debug_data(sub_dir)
        else:
            pass
            # for render_product in render_products:
            #     render_product_name = render_product[3:]
            #     sub_dir = os.path.join(render_product_name, data[render_product]["camera"].split("/")[-1])
                # self._write_rgb(data, sub_dir, f"rgb-{render_product_name}")
                # self._write_segmentation(
                #     data,
                #     sub_dir,
                #     f"semantic_segmentation-{render_product_name}",
                #     f"instance_segmentation_fast-{render_product_name}",
                # )
                # self._write_object_detection(
                #     data,
                #     sub_dir,
                #     render_product,
                #     f"bounding_box_2d_tight_fast-{render_product_name}",
                #     f"bounding_box_2d_loose_fast-{render_product_name}",
                # )
                # self._write_distance_to_camera(data, sub_dir, f"distance_to_camera-{render_product_name}")

        self._frame_id += 1

    #########################################################
    ########### Below are debug functions ###################
    #########################################################
    
    # Write overlay debug data to disk
    def _write_debug_data(self, sub_dir: str):
        # Create overlay image from the RGB data
        rgb_img = Image.fromarray(self._debug_data["raw_rgb"])
        draw = ImageDraw.Draw(rgb_img)
        debug_dir_name = "image_02" if self._use_kitti_dir_names else "debug"
        debug_file_path = os.path.join(sub_dir, debug_dir_name, f"{self._frame_id}.png")
        
        # Draw the projected cuboid and its edges
        for keypoints in self._debug_data["projected_keypoints"]:
            self._draw_projected_keypoints(draw, keypoints)

        # Get the stored camera parameters for debug purposes
        camera_projection_matrix = self._debug_data["camera_projection_matrix"]
        camera_view_matrix = self._debug_data["camera_view_matrix"]
        screen_size = self._debug_data["resolution"]

        # Draw objects local frame axes
        for i, tf in enumerate(self._debug_data["world_frame_transforms"]):
            size = self._debug_data["size_local"][i]
            center = self._debug_data["center_local"][i]
            self._draw_local_frame_axes(
                draw,
                tf,
                camera_view_matrix,
                camera_projection_matrix,
                screen_size,
                size_local=size,
                origin_local=center,
            )

        # Overlay the world frame axes on the bottom left part of the RGB image
        self._draw_world_frame_axes_bottom_left(draw, camera_view_matrix, camera_projection_matrix, screen_size)

        self._backend.schedule(F.write_image, path=debug_file_path, data=np.asarray(rgb_img))


    # Project a 3D point from world coordinates to 2D screen coordinates
    def _project_world_point_to_screen(self, world_point, view_matrix, projection_matrix, screen_size):
        point_camera = self._world_point_to_camera_point(world_point, view_matrix)
        return self._project_camera_point_to_screen(point_camera, projection_matrix, screen_size)
    
    # Transform a 3D point from world coordinates to camera coordinates
    def _world_point_to_camera_point(self, world_point, view_matrix):
        # Convert the 3D point to homogeneous coordinates (if not already in that form)
        point_homogeneous = np.array(world_point) if len(world_point) == 4 else np.array([*world_point, 1.0])

        # Transform to camera frame (row-major representation where the translation vector is on the left side of the multiplication)
        point_camera = point_homogeneous @ view_matrix

        return point_camera   
    
    # Draws the world frame axes at the bottom left corner of the image.
    def _draw_world_frame_axes_bottom_left(
        self, draw, camera_view_matrix, camera_projection_matrix, screen_size, axes_scale=0.03, margin_percentage=0.03
    ):
        # Set a world location for the axes origin (1 unit in front of the camera) where -Z is the camera's forward direction
        camera_to_world_matrix = np.linalg.inv(camera_view_matrix)
        point_in_camera_space = np.array([0, 0, -1, 1])

        # Create the axes in world (1 unit in front of the camera) with the given axes size
        origin_world = point_in_camera_space @ camera_to_world_matrix
        x_axis_end_point_world = np.array([axes_scale + origin_world[0], origin_world[1], origin_world[2], 1])
        y_axis_end_point_world = np.array([origin_world[0], axes_scale + origin_world[1], origin_world[2], 1])
        z_axis_end_point_world = np.array([origin_world[0], origin_world[1], axes_scale + origin_world[2], 1])

        # Create a partial function with fixed camera parameters
        project_to_screen = partial(
            self._project_world_point_to_screen,
            view_matrix=camera_view_matrix,
            projection_matrix=camera_projection_matrix,
            screen_size=screen_size,
        )

        # Project the origin and axes end points into 2D screen coordinates
        origin_2d = project_to_screen(origin_world)
        x_axis_end_2d = project_to_screen(x_axis_end_point_world)
        y_axis_end_2d = project_to_screen(y_axis_end_point_world)
        z_axis_end_2d = project_to_screen(z_axis_end_point_world)

        # Calculate offset margin (a percentage of the screen size) to ensure axes are not on the edge of the screen
        margin = int(margin_percentage * min(screen_size))
        offset_x = margin - min(origin_2d[0], x_axis_end_2d[0], y_axis_end_2d[0], z_axis_end_2d[0])
        offset_y = screen_size[1] - margin - max(origin_2d[1], x_axis_end_2d[1], y_axis_end_2d[1], z_axis_end_2d[1])

        # Apply the offset to the projected points
        origin_2d = (origin_2d[0] + offset_x, origin_2d[1] + offset_y)
        x_axis_end_2d = (x_axis_end_2d[0] + offset_x, x_axis_end_2d[1] + offset_y)
        y_axis_end_2d = (y_axis_end_2d[0] + offset_x, y_axis_end_2d[1] + offset_y)
        z_axis_end_2d = (z_axis_end_2d[0] + offset_x, z_axis_end_2d[1] + offset_y)

        # Draw the axes with the specified colors
        draw.line([origin_2d, x_axis_end_2d], fill="red", width=2)  # X-axis in red
        draw.line([origin_2d, y_axis_end_2d], fill="green", width=2)  # Y-axis in green
        draw.line([origin_2d, z_axis_end_2d], fill="blue", width=2)  # Z-axis in blue


    # Draw the projected cuboid and its edges
    def _draw_projected_keypoints(self, draw, keypoints, point_size=4, edge_size=2):
        # Draw the projected cuboid keypoint vertices in the specified colors
        for i, point in enumerate(keypoints):
            draw.ellipse(
                (point[0] - point_size, point[1] - point_size, point[0] + point_size, point[1] + point_size),
                fill=self._CUBOID_KEYPOINT_COLORS[i],
            )

        # Draw the edges of the projected cuboid with specified colors for each set
        edges = {
            "front": [(1, 2), (2, 4), (4, 3), (3, 1)],  # Front face
            "back": [(5, 6), (6, 8), (8, 7), (7, 5)],  # Back face
            "connecting": [(1, 5), (2, 6), (3, 7), (4, 8)],  # Connecting edges
        }
        for edge_type, edge_list in edges.items():
            for start, end in edge_list:
                draw.line(keypoints[start] + keypoints[end], fill=self._CUBOID_EDGE_COLORS[edge_type], width=edge_size)

    # Projects the local frame axes of the object to the screen
    def _draw_local_frame_axes(
        self,
        draw,
        local_to_world_transform,
        camera_view_matrix,
        camera_projection_matrix,
        screen_size,
        size_local=[1, 1, 1],
        origin_local=[0, 0, 0],
        axes_length_perc=0.25,
    ):
        # The length of the local axes is a percentage of the mean size of the object in local frame (before any scaling)
        local_axes_length = np.mean(size_local) * axes_length_perc

        # Define the end points of the local coordinate system axes include the local center of the object bounds
        origin_local = np.array([origin_local[0], origin_local[1], origin_local[2], 1])
        x_axis_end_point_local = np.array([local_axes_length + origin_local[0], origin_local[1], origin_local[2], 1])
        y_axis_end_point_local = np.array([origin_local[0], local_axes_length + origin_local[1], origin_local[2], 1])
        z_axis_end_point_local = np.array([origin_local[0], origin_local[1], local_axes_length + origin_local[2], 1])

        # Transform local end points to world frame using row-major matrix multiplication (translation on the left side)
        origin_world = origin_local @ local_to_world_transform
        x_axis_end_point_world = x_axis_end_point_local @ local_to_world_transform
        y_axis_end_point_world = y_axis_end_point_local @ local_to_world_transform
        z_axis_end_point_world = z_axis_end_point_local @ local_to_world_transform

        # Define a partial helper function to project 3D world points to 2D screen points
        project_to_screen = partial(
            self._project_world_point_to_screen,
            view_matrix=camera_view_matrix,
            projection_matrix=camera_projection_matrix,
            screen_size=screen_size,
        )

        # Project the origin and axes end points from 3D world coordinates to 2D screen coordinates
        origin_2d = project_to_screen(origin_world)
        x_axis_end_2d = project_to_screen(x_axis_end_point_world)
        y_axis_end_2d = project_to_screen(y_axis_end_point_world)
        z_axis_end_2d = project_to_screen(z_axis_end_point_world)

        # Draw the 3D axes on the 2D screen using lines with appropriate colors for each axis
        draw.line([origin_2d, x_axis_end_2d], fill="red", width=2)  # X-axis in red
        draw.line([origin_2d, y_axis_end_2d], fill="green", width=2)  # Y-axis in green
        draw.line([origin_2d, z_axis_end_2d], fill="blue", width=2)  # Z-axis in blue


    def _is_bbox_valid(self, bbox_tight: dict):
        if not bbox_tight:
            return False
        if not self._is_bbox_big_enough(bbox_tight, self._bbox_height_threshold):
            return False
        if not self._is_bbox_image_region_visible_by_veiling(self.uw_image_np, bbox_tight, self._veiling_visibility_threshold):
            return False
        if not self._is_bbox_in_scope(self.uw_image_np, bbox_tight):
            return False
        return True

    def _is_bbox_in_scope(self, image: np.ndarray, bbox_tight: dict):
        return bbox_tight["x_min"] >= 0 and bbox_tight["y_min"] >= 0 and bbox_tight["x_max"] < image.shape[1] and bbox_tight["y_max"] < image.shape[0]

    # NOTE: This test will fail if objects are dense therefore visible object will appear in the bbox of invisible object
    # TODO: Fix this in the future or use less dense objects SDG for now
    def _is_bbox_image_region_visible_by_veiling(self, image: np.ndarray, bbox_tight: dict, threshold: float):
        """
        Returns True if the maximum color distance between pixels in the bbox and self._veiling is above threshold.
        """
        bbox_region = image[bbox_tight["y_min"]:bbox_tight["y_max"], bbox_tight["x_min"]:bbox_tight["x_max"], :3]  # Only RGB, ignore alpha if present
        pixels = bbox_region.reshape(-1, 3)
        veiling_rgb = np.array(self._veiling) * 255  # If self._veiling is in [0,1]
        color_dists = np.linalg.norm(pixels - veiling_rgb, axis=1)
        max_dist = np.max(color_dists)
        return max_dist >= threshold

    def _is_bbox_big_enough(self, bbox_tight: dict, threshold: float):
        return (bbox_tight["x_max"] - bbox_tight["x_min"] >= threshold) and (bbox_tight["y_max"] - bbox_tight["y_min"] >= threshold)

    def _get_bbox_from_instance_id(self, inst_seg_uint32: np.ndarray, iid: int):
        """
        Returns the tight bounding box [x_min, y_min, x_max, y_max] for the given instance id.
        """
        ys, xs = np.where(inst_seg_uint32 == iid)
        if len(xs) == 0 or len(ys) == 0:
            return None  # No pixels found for this iid
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}

