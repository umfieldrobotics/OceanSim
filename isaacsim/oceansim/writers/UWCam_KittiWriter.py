import csv
import io
from typing import List, Union
import carb
import numpy as np
import warp as wp
from pxr import Gf
from omni.syntheticdata.scripts.SyntheticData import SyntheticData
import omni.replicator.core.scripts.functional as F
from omni.replicator.core import AnnotatorRegistry, BackendDispatch
from omni.replicator.core.scripts.writers import Writer
from isaacsim.oceansim.utils.UWrenderer_utils import *
from isaacsim.replicator.writers.scripts.utils import calculate_truncation_ratio_simple
import isaacsim.core.utils.rotations as rotations_utils
import yaml
import os
import json
from PIL import Image, ImageDraw
from functools import partial

# NOTE: This is an unintuitive import for a writer class since we should expect a deterministic output
# The good thing is we can use seed to make it deterministic
import random

__version__ = "0.1.0"

DULUTH_PARAM_DICT = {
    "scale_range": (0.5, 1.0),
    "veiling": {
        "duluth": (0.19, 0.30, 0.0)
        }, 
    "backscatter": {
        "duluth": (0.53, 0.68, 0.99)
        },
    "attenuation": {
        "duluth": (0.53, 0.68, 0.99)
        }
    }

SEACLEAR_PARAM_DICT = {
    "scale_range": (0.5, 1.0),
    "veiling": {
        "seaclear": (0.01, 0.66, 0.68)
        },
    "backscatter": {
        "seaclear": (0.01, 0.66, 0.68)
        },
    "attenuation": {
        "seaclear": (0.50, 0.90, 1.0)
        }
    }

    
UW_PARAM_DICT = {
    "scale_range": (1.0, 1.0),
    "veiling": {
            "deep_sea": (0.0, 0.0, 0.28),
            # "shallow_water": (0.05, 0.11, 0.7),
            "akdeniz": (0.14, 0.3, 0.5),
            "river": (0.294, 0.4, 0.263),
            "mud": (0.259, 0.259, 0.024),
            "mhl": (0.0, 0.3021, 0.239),
            "murky": (0.275, 0.212, 0.071),
            "seaclear_sea_urchin": (0.08, 0.42, 0.52),
        },
    "backscatter": {
            "Type I": (0.905, 0.961, 0.982),
            "Type IA": (0.804, 0.954, 0.975),
            "Type IB": (0.830, 0.940, 0.968),
            "Type II": (0.800, 0.925, 0.940),
            "Type III": (0.750, 0.885, 0.890),
            "Type 1": (0.750, 0.885, 0.875),
            "Type 3": (0.710, 0.820, 0.800),
            "Type 5": (0.670, 0.730, 0.670),
            "Type 7": (0.620, 0.610, 0.590),
            "Type 9": (0.550, 0.460, 0.290),
    },
    "attenuation": {
            "Type I": (0.905, 0.961, 0.982),
            "Type IA": (0.804, 0.954, 0.975),
            "Type IB": (0.830, 0.940, 0.968),
            "Type II": (0.800, 0.925, 0.940),
            "Type III": (0.750, 0.885, 0.890),
            "Type 1": (0.750, 0.885, 0.875),
            "Type 3": (0.710, 0.820, 0.800),
            "Type 5": (0.670, 0.730, 0.670),
            "Type 7": (0.620, 0.610, 0.590),
            "Type 9": (0.550, 0.460, 0.290),
    }
}


class UWCam_KittiWriter(Writer):
    """Writer outputting data in the ``KITTI`` annotation format:
    http://www.cvlibs.net/datasets/kitti/

    .. note::
        Development work to provide full-support is ongoing.

    Supported Annotations:
    - RGB
    - Object Detection (partial 2D support, see notes)
    - Depth
    - Semantic Segmentation
    - Instance Segmentation

    Args:
        output_dir: Output directory to which ``KITTI`` annotations will be saved.
        semantic_types: List of semantic types to consider. If ``None``, only consider semantic types ``"class"``.
        omit_semantic_type: If ``True``, only record the semantic data (ie. ``class: car`` becomes ``car``).
        bbox_height_threshold: The minimum valid bounding box height, in pixels. Value must be positive integers.
        partly_occluded_threshold: Minimum occlusion factor for bounding boxes to be considered partly occluded.
        fully_visible_threshold: Minimum occlusion factor for bounding boxes to be considered fully visible.
        mapping_path: File path to JSON to use as the label to color mapping for ``KITTI``. ex: ``{'car':(155, 255, 74, 255)}``
            If no ``mapping_path`` is supplied, the default semantics specified in the KITTI spec will be used. Note
            that semantics not specified in the mapping will be labelled as "unlabelled". The mapping may include both
            "unlabelled" and "background" labels to specify how each is colored when ``colorize_instance_segmentation``
            is ``True``
        mapping_dict: Dictionary of labels and their colors in (R,G,B,A). ex: ``{"my_semantic": (12, 07, 83, 255)}``
            mapping_dict and mapping_path cannot both be specified.
        colorize_instance_segmentation: If ``True``, save an additional colorized instance segmentation image to the
            ``instance_rgb`` directory
        use_kitti_dir_names: If ``True``, use standard ``KITTI`` directory names: ``rgb`` -> ``image_02``,
            ``semantic_segmentation`` -> ``semantic``, ``instance_segmentation`` -> ``instance``, ``object_detection`` -> ``label_02``

    .. note::
        - Object Detection
        - Bounding boxes with a height smaller than 25 pixels are discarded
        - **Supported:** bounding box extents, semantic labels
        - **Partial Support:** occluded (occlusion is estimated from the area ratio of tight / loose bounding boxes)
    """

    def __init__(
        self,
        output_dir: str,
        s3_bucket: str = None,
        s3_region: str = None,
        s3_endpoint: str = None,
        semantic_types: List[str] = None,
        omit_semantic_type: bool = True,
        bbox_height_threshold: int = 25,
        bbox2d_partly_occluded_threshold: float = 0.5,
        bbox2d_fully_visible_threshold: float = 0.95,
        veiling_visibility_threshold: float = None,
        use_tight_bbox: bool = False,
        mapping_path: str = None,
        mapping_dict: dict = None,
        colorize_instance_segmentation: bool = False,
        semantic_filter_predicate: str = None,
        use_kitti_dir_names: bool = False,
        UW_param: Union[str, dict] = UW_PARAM_DICT,
        cuboid_keypoints_order: list = ["Center", "LDB", "LDF", "LUB", "LUF", "RDB", "RDF", "RUB", "RUF"],
        debug_mode: bool = False,
        enable_caustics: bool = False,
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
        self.backend = self._backend # I know, boilerplate...
        self._omit_semantic_type = omit_semantic_type
        self._bbox_height_threshold = bbox_height_threshold
        self._use_tight_bbox = use_tight_bbox
        self._bbox2d_partly_occluded_threshold = bbox2d_partly_occluded_threshold
        self._bbox2d_fully_visible_threshold = bbox2d_fully_visible_threshold
        self._use_kitti_dir_names = use_kitti_dir_names
        self._cuboid_keypoints_order = cuboid_keypoints_order
        self._debug_mode = debug_mode
        self._veiling_visibility_threshold = veiling_visibility_threshold
        self._enable_caustics = enable_caustics
        if self._debug_mode:
            self._CUBOID_KEYPOINT_COLORS = ["white", "red", "green", "blue", "yellow", "cyan", "magenta", "orange", "purple"]
            self._CUBOID_EDGE_COLORS = {"front": "red", "back": "blue", "connecting": "green"}
            self._debug_data = {}

        if isinstance(UW_param, str):
            with open(UW_param, 'r') as file:
                try:
                    # Load the YAML content
                    self._UW_param = yaml.safe_load(file)
                    print(f"Loaded render parameters {self._UW_param} from {UW_param}")
                except yaml.YAMLError as exc:
                    carb.log_error(f"Error reading render parameter YAML from {UW_param} file: {exc}")
                    self._UW_param = UW_PARAM_DICT
                    carb.log_error(f"Fallback to default: {self._UW_param}")
        else:
            self._UW_param = UW_param
            print(f"Using render param {self._UW_param}")
        

        self.colorize_instance_segmentation = colorize_instance_segmentation

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
            AnnotatorRegistry.get_annotator(
                "rgb", device="cuda",
            ),
            AnnotatorRegistry.get_annotator(
                "normals", device="cuda",
            ),
            "bounding_box_2d_tight_fast",
            "bounding_box_2d_loose_fast",
            AnnotatorRegistry.get_annotator(
                "semantic_segmentation", init_params={"mapping": self._get_anno_semantic_mapping()}
            ),
            AnnotatorRegistry.get_annotator(
                "instance_segmentation_fast", init_params={"colorize": colorize_instance_segmentation}
            ),
            AnnotatorRegistry.get_annotator(
                "distance_to_camera", device="cuda"
            ),
            "bounding_box_3d_fast", 
            "camera_params",
        ]


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

    def _write_rgb(self, data, sub_dir: str, rgb_annotator: str, dist_to_cam_annotator:str, camera_param_annotator:str, normals_annotator:str):

        if self._debug_mode:
            self._debug_data["raw_rgb"] = data[rgb_annotator].numpy()
        width, height = data[rgb_annotator].shape[:2]
        uw_image = wp.empty(shape=data[rgb_annotator].shape, dtype=wp.uint8)
        uw_rgb_dir_name = "uw_image_02" if self._use_kitti_dir_names else "uw_rgb"
        uw_rgb_file_path = os.path.join(sub_dir, uw_rgb_dir_name, f"{self._frame_id}.png")

        self._scale = random.uniform(self._UW_param["scale_range"][0], self._UW_param["scale_range"][1])
        self._veiling = random.choice(list(self._UW_param["veiling"].values()))
        self._backscatter = random.choice(list(self._UW_param["backscatter"].values()))
        # self._attenuation = self._backscatter # Defaul the attentuation to be the same as the backscatter
        self._attenuation = random.choice(list(self._UW_param["attenuation"].values()))
        uw_image = data[rgb_annotator]
        if self._enable_caustics:
            _caustics_tex = wp.empty(shape=(height, width, 4), dtype=wp.uint8)
            _world_pos = wp.empty(shape=(height, width, 3), dtype=wp.float32)

            wp.launch(
                kernel=water_caustics,
                dim=(width, height),  # (x, y)
                inputs=[_caustics_tex, width, height, self._frame_id, 2.0],
            )


            # Launch depth to world kernel once (this doesn't change)
            wp.launch(
                kernel=depth_to_world_pos,
                dim=(width, height),  # Launch dimensions (width, height)
                inputs=[
                    data[dist_to_cam_annotator],
                    wp.mat44(data[camera_param_annotator]["cameraProjection"].reshape(4, 4)),
                    wp.mat44(data[camera_param_annotator]["cameraViewTransform"].reshape(4, 4)),
                    width,
                    height
                ],
                outputs=[_world_pos],
            )
            wp.launch(
                kernel=blend_caustics,
                dim=(width, height),
                inputs=[
                    data[rgb_annotator],
                    _world_pos,
                    data[normals_annotator],
                    _caustics_tex,
                    wp.vec3f(0.0, 0.0, 1.0),
                    1.0,       # blend_weight
                    random.uniform(0.5, 1.5),       # uv_scale_x (horizontal scaling)
                    random.uniform(0.5, 1.5),       # uv_scale_y (vertical scaling)
                    0.0,       # depth_min
                    100.0,   # depth_max
                    width,         # tex_w
                    height,         # tex_h
                ],
                outputs=[uw_image],
            )


        
            wp.launch(
                    dim=data[rgb_annotator].shape[:2],
                    kernel=UW_render_2,
                    inputs=[
                        uw_image,
                        data[dist_to_cam_annotator],
                        self._scale,
                        self._veiling,
                        self._backscatter,
                        self._attenuation,

                    ],
                    outputs=[
                        uw_image
                    ]
                )  
        else:
            wp.launch(
                    dim=data[rgb_annotator].shape[:2],
                    kernel=UW_render_2,
                    inputs=[
                        data[rgb_annotator],
                        data[dist_to_cam_annotator],
                        self._scale,
                        self._veiling,
                        self._backscatter,
                        self._attenuation,

                    ],
                    outputs=[
                        uw_image
                    ]
                )  


        self.uw_image_np = uw_image
        self._backend.schedule(F.write_image, data=uw_image, path=uw_rgb_file_path)

    def _write_object_pose(self, data, sub_dir: str, bbox_3d_annotator: str, camera_param_annotator: str):
        objs_data = self._process_bounding_boxes_3d(data[bbox_3d_annotator], data[camera_param_annotator])
        pose_dir_name = "pose_02" if self._use_kitti_dir_names else "pose"
        pose_file_path = os.path.join(sub_dir, pose_dir_name, f"{self._frame_id}.json")
        self._backend.schedule(F.write_json, path=pose_file_path, data=objs_data, indent=2)

    # Write a function to avoid creating bounding boxes or recording information for any objects located beyond a specified distance (e.g., 10 meters, set as a variable)
    # def _obj_beyond_max_distance(self, obj_location_world_frame, camera_params, max_distance=None, enable_filter=None):
    #     """
    #     Determines whether an object is beyond the specified max distance from the camera.
        
    #     Parameters:
    #     - obj_location_world_frame (list or np.array): [x, y, z] position of the object in world coordinates.
    #     - camera_params (dict): Camera parameters containing the camera view transform.
    #     - max_distance (float, optional): Maximum allowable distance in meters. Uses self._max_distance if None.
    #     - enable_filter (bool, optional): If False, the function always returns False (i.e., no filtering).
    #                                     Uses self._enable_distance_filter if None.
        
    #     Returns:
    #     - bool: True if object is beyond max_distance and filtering is enabled, False otherwise.
    #     """
    #     # Determine whether distance filtering is enabled
    #     # if distance filtering is enabled and use default filter and distance
    #     if enable_filter is None:
    #         enable_filter = self._enable_distance_filter
    #     if max_distance is None:
    #         max_distance = self._max_distance

    #     # if disabled, keep objects
    #     if not enable_filter:
    #         return False
        
    #     # Reshape camera view transform matrix (4x4)
    #     world_to_camera_tf = camera_params["cameraViewTransform"].reshape(4, 4)
        
    #     # Invert the matrix to get the camera's position in world coordinates
    #     camera_to_world_tf = np.linalg.inv(world_to_camera_tf)
    #     # The translation vector (x,y,z) is stored in the last row (index 3) of the inverse matrix
    #     camera_position_world = camera_to_world_tf[3, :3]  # Extract translation component for camera location in world space
        
    #     # Convert object position to numpy array
    #     obj_position = np.array(obj_location_world_frame[:3])
        
    #     # Compute Euclidean distance
    #     distance = np.linalg.norm(obj_position - camera_position_world)

    #     # Return True only if the object is farther than the limit
    #     return distance > max_distance

    def _write_object_detection(
        self,
        data,
        sub_dir: str,
        render_product_annotator: str,
        bbox_2d_tight_annotator: str,
        bbox_2d_loose_annotator: str,
        bbox_3d_annotator: str,
        camera_param_annotator: str,
    ):
        r"""
        Saves the labels for the object detection data in Kitti format.

        Unsupported fields: alpha, rotation_y, truncated (all set to default values of 0.0)

        Notes on occlusion:
        # This estimation relies on the ratio between loose (unoccluded) and tight bounding boxes
        # and may produce unexpected results in certain cases:
        #
        #        //           XXXX                 //  XXXX
        #  _____//____/_______XXXX          ______//___XXXX______
        # )   __          __  XXXX         )   __      XXXX_     \
        # |__/  \________/  \_XXXX         |__/  \_____XXXX \____|
        # ___\__/________\__/_XXXX__      ____\_ /_____XXXX_/______
        # PARTLY OCCLUDED (OK!)           FULLY VISIBLE (INCORRECT)
        """
        label_set = []

        rp_width = data[render_product_annotator]["resolution"][0]
        rp_height = data[render_product_annotator]["resolution"][1]

        bbox_tight = data[bbox_2d_tight_annotator]["data"]
        bbox_loose = data[bbox_2d_loose_annotator]["data"]
        bbox_3d = data[bbox_3d_annotator]["data"]

        bbox_tight_bbox_ids = data[bbox_2d_tight_annotator]["info"]["bboxIds"]
        bbox_loose_bbox_ids = data[bbox_2d_loose_annotator]["info"]["bboxIds"]
        bbox_3d_bbox_ids = data[bbox_3d_annotator]["info"]["bboxIds"]
        
        tight_id_to_bbox = {bbox_tight_id: bbox_tight_data for bbox_tight_id, bbox_tight_data in zip(bbox_tight_bbox_ids, bbox_tight)}
        loose_id_to_bbox = {bbox_loose_id: bbox_loose_data for bbox_loose_id, bbox_loose_data in zip(bbox_loose_bbox_ids, bbox_loose)}
        bbox3d_id_to_bbox = {bbox_3d_id: bbox_3d_data for bbox_3d_id, bbox_3d_data in zip(bbox_3d_bbox_ids, bbox_3d)}


        # For box in tight and bbox_3d, find the corresponding index of box in loose
        shared_ids = np.intersect1d(
                bbox_tight_bbox_ids,
                np.intersect1d(bbox_loose_bbox_ids, bbox_3d_bbox_ids)
                )
        


        for id in shared_ids:
            box_tight = tight_id_to_bbox[id]
            box_loose = loose_id_to_bbox[id]
            
            label = []
            
            # Specify to use tight or loose bbox
            box = box_tight if self._use_tight_bbox else box_loose
            box_annotator = bbox_2d_tight_annotator if self._use_tight_bbox else bbox_2d_loose_annotator

            if not self._is_bbox_valid(box):
                continue

            area_tight = (box_tight["x_max"] - box_tight["x_min"]) * (box_tight["y_max"] - box_tight["y_min"])
            area_loose = (box_loose["x_max"] - box_loose["x_min"]) * (box_loose["y_max"] - box_loose["y_min"])
            area_ratio = area_tight / (area_loose + 1e-5)

            if area_ratio >= self._bbox2d_fully_visible_threshold:
                occlusion_estimation = 0
            elif area_ratio >= self._bbox2d_partly_occluded_threshold:
                occlusion_estimation = 1
            else:
                occlusion_estimation = 2
                continue  # Skip if heavily occluded



            # Only compute object's 3d information after the above test
            bbox3d_info = self._process_bounding_box_3d_single(bbox3d_id_to_bbox[id], data[camera_param_annotator])
            
            # Check if a majority part of the object is out of the frame
            if bbox3d_info['truncation_ratio'] > 0.6:
                continue
            
            
            # Check if object is beyond max distance, skip if it is
            # if self._obj_beyond_max_distance(bbox3d_info["location_world_frame"], data[camera_param_annotator]):
            #     continue

            
            semantic_label = data[box_annotator]["info"]["idToLabels"].get(box["semanticId"])

            if self._omit_semantic_type:
                # omit semantic type
                semantic_label = semantic_label.get("class", "Unlabelled")
            

            # Adding Kitti Data,  NOTE: Only class and 2d bbox coordinates are filled in
            label.append(semantic_label)  # semantic
            label.append(f"{bbox3d_info['truncation_ratio']:.2f}")  # truncated
            label.append(occlusion_estimation)  # occluded (estimation, NOT ACCURATE!)
            label.append(f"{bbox3d_info['alpha']:.2f}")  # alpha 
            label.append(box["x_min"])  # x min
            label.append(box["y_min"])  # y min
            label.append(box["x_max"])  # x max
            label.append(box["y_max"])  # y max
            #NOTE: size is in world frame (meters) and this represents the size of the 3D bbox that does not rotate with the object
            #NOTE: To get the local frame (cm), use bbox3d_info["size_local"] and this represents the size of the 3D bbox that rotates with the object
            label.append(f"{bbox3d_info['size_world'][2]:.2f}")  # z_size represents height
            label.append(f"{bbox3d_info['size_world'][1]:.2f}")  # y_size represents width
            label.append(f"{bbox3d_info['size_world'][0]:.2f}")  # x_size represents length
            # location of the xform origin in camera frame (NOTE: not the object centroid which is bbox3d_info["center_camera_frame"])
            label.extend([f"{v:.3f}" for v in bbox3d_info["location_camera_frame"]])
            label.append(f"{bbox3d_info['rotation_y']:.2f}")  # rotation_y

            label_set.append(label)

        det_dir_name = "label_02" if self._use_kitti_dir_names else "object_detection"
        kitti_filepath = os.path.join(sub_dir, det_dir_name, f"{self._frame_id}.txt")
        buf = io.StringIO()

        writer = csv.writer(buf, delimiter=" ")
        writer.writerows(label_set)

        self._backend.schedule(self._backend.write_blob, data=bytes(buf.getvalue(), "utf-8"), path=kitti_filepath)
    
    
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

    def _write_segmentation(self, data, sub_dir: str, sem_annotator: str, inst_annotator: str):
        """
        Instance segmentation follows the format specified here: https://www.vision.rwth-aachen.de/page/mots
        """
        # Setup file paths
        inst_dir_name = "instance" if self._use_kitti_dir_names else "instance_segmentation"
        seg_filepath = os.path.join(sub_dir, "semantic_segmentation", f"{self._frame_id}.png")
        seg_mapping_filepath = os.path.join(sub_dir, "semantic_mapping.json")

        inst_filepath = os.path.join(sub_dir, inst_dir_name, f"{self._frame_id}.png")

        inst_id_to_labels = data[inst_annotator]["info"]["idToSemantics"]
        self._backend.schedule(F.write_json, data=self.mapping_dict, path=seg_mapping_filepath)

        inst_seg_img = data[inst_annotator]["data"]
        height, width = inst_seg_img.shape[:2]

        if self.colorize_instance_segmentation:
            inst_col_filepath = os.path.join(sub_dir, "instance_rgb", f"{self._frame_id}.png")
            inst_seg_img_colorized = inst_seg_img.view(np.uint8)
            inst_seg_img_colorized = inst_seg_img_colorized.reshape(height, width, -1)
            self._backend.schedule(F.write_image, data=inst_seg_img_colorized, path=inst_col_filepath)

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
        inst_seg_uint32 = inst_seg_img.view(np.uint32).squeeze()
        inst_seg_img_renumbered = np.zeros((height, width), dtype=np.uint16)
        sem_seg_img_renumbered = np.zeros((height, width), dtype=np.uint8)
        for i, iid in enumerate(instance_ids):
            semantic_class = inst_id_to_labels[iid].get("class", "unlabelled")
            is_unlabelled = semantic_class.lower() == "unlabelled"
            is_background = semantic_class.lower() == "background"
            is_in_mapping = semantic_class in self.mapping_dict
            bbox_tight = self._get_bbox_from_instance_id(inst_seg_uint32, iid)
            is_valid = self._is_bbox_valid(bbox_tight)
            if not is_in_mapping or is_unlabelled or is_background or not is_valid:
                inst_seg_img_renumbered[inst_seg_uint32 == iid] = 0
            else:
                cur_semantics = str(inst_id_to_labels[iid])
                cur_idx.setdefault(cur_semantics, 0)
                cur_idx[cur_semantics] += 1
                semantics_renumbered = self.mapping_dict.get(semantic_class, 0)
                inst_seg_img_renumbered[inst_seg_uint32 == iid] = cur_idx[cur_semantics] + semantics_renumbered * 256
                sem_seg_img_renumbered[inst_seg_uint32 == iid] = semantics_renumbered

        self._backend.schedule(F.write_image, data=inst_seg_img_renumbered, path=inst_filepath)
        self._backend.schedule(F.write_image, data=sem_seg_img_renumbered, path=seg_filepath)
    
        
    
    def _write_distance_to_camera(self, data, sub_dir: str, annotator: str):
        distance_to_camera_metres = data[annotator]
        distance_to_camera_metres = np.nan_to_num(distance_to_camera_metres.numpy(), posinf=0.0)
        distance_to_camera_uint16 = (distance_to_camera_metres * 256).astype(np.uint16)
        file_path = os.path.join(sub_dir, "depth", f"{self._frame_id}.png")
        self._backend.schedule(F.write_image, data=distance_to_camera_uint16, path=file_path)
    
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
        # NOTE: the writing order matters, the rgb is always processed first because the latter processing may depend on the UW_rgb image
        # NOTE: for testing visibility
        render_products = [k for k in data.keys() if k.startswith("rp_")]
        if len(render_products) == 1:
            sub_dir = data[render_products[0]]["camera"].split("/")[-1]
            self._write_rgb(data, sub_dir, "rgb", "distance_to_camera", "camera_params", "normals")
            self._write_segmentation(data, sub_dir, "semantic_segmentation", "instance_segmentation_fast")
            self._write_object_detection(
                data, 
                sub_dir, 
                render_products[0], 
                "bounding_box_2d_tight_fast", 
                "bounding_box_2d_loose_fast", 
                "bounding_box_3d_fast",
                "camera_params"
            )
            self._write_distance_to_camera(data, sub_dir, "distance_to_camera")
            self._write_camera_param(data, sub_dir, "camera_params")
            if self._debug_mode:
                # NOTE: To use the debug mode, you have to write the object pose to get some debug info
                self._write_object_pose(data, sub_dir, "bounding_box_3d_fast", "camera_params")
                self._write_debug_data(sub_dir)

        else:
            for render_product in render_products:
                render_product_name = render_product[3:]
                sub_dir = os.path.join(render_product_name, data[render_product]["camera"].split("/")[-1])
                self._write_rgb(
                    data, 
                    sub_dir, 
                    f"rgb-{render_product_name}", 
                    f"distance_to_camera-{render_product_name}", 
                    f'camera_params-{render_product_name}',
                    f'normals-{render_product_name}'
                )
                self._write_segmentation(
                    data,
                    sub_dir,
                    f"semantic_segmentation-{render_product_name}",
                    f"instance_segmentation_fast-{render_product_name}",
                )
                self._write_object_detection(
                    data,
                    sub_dir,
                    render_product,
                    f"bounding_box_2d_tight_fast-{render_product_name}",
                    f"bounding_box_2d_loose_fast-{render_product_name}",
                    f"bounding_box_3d_fast-{render_product_name}",
                    f"camera_params-{render_product_name}"
                )
                self._write_distance_to_camera(data, sub_dir, f"distance_to_camera-{render_product_name}")
                self._write_camera_param(data, sub_dir, f"camera_params-{render_product_name}")
                if self._debug_mode:
                    self._write_object_pose(data, sub_dir, f"bounding_box_3d_fast-{render_product_name}", f"camera_params-{render_product_name}")
                    self._write_debug_data(sub_dir)

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

    ### might have to add the enable filter her to disable bbox validation ###
    def _is_bbox_valid(self, bbox_tight: dict):
        if not bbox_tight:
            return False
        if not self._is_bbox_big_enough(bbox_tight, self._bbox_height_threshold):
            return False
        # if not self._is_bbox_image_region_visible_by_veiling(self.uw_image_np, bbox_tight, self._veiling_visibility_threshold):
        #     return False
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