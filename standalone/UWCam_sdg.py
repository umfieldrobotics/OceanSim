###################################################
#### Here goes arg parser and Isaac Sim config ####
###################################################
import argparse
import json
import os
import sys
import carb.settings
import yaml
from isaacsim import SimulationApp
import carb
from PIL import Image, ImageDraw
from functools import partial

# Default config dict, can be updated/replaced using json/yaml config files ('--config' cli argument)
config = {
    "launch_config": {
        "renderer": "RaytracedLighting",
        "headless": False,
        "extra_args": [
            "--/persistent/renderer/rtpt/enabled=True",              # This enables RTX realtime preview renderer
            "--/log/level=error",                                    # These will shut up isaac sim as I could 
            "--/log/fileLogLevel=error", 
            "--/log/outputStreamLevel=error"
            ]
    },
    "num_cameras" : 1,
    # "camera_collider_radius": 0.2,
    "env_url": "/frog-drive/ocean-sim/sim2real/sceneAssets/duluth/Collected_pebble_floor/padded_pebble_floor_water.usd",
    "rt_subframes": 16,
    "num_frames": 10,
    "resolution": (1920, 1080),
    "camera_properties_kwargs": {
        "focal_length": 24.0,
        "focus_distance": 400,
        "f_stop": 0.0,
        "clipping_range": (0.01, 100),
    },
    "writer_type": "UWCam_KittiWriter",
    "writer_kwargs": {
        "output_dir": "/home/haoyu/Desktop/viz/",
        "colorize_instance_segmentation": True,
        "UW_param": "/frog-drive/ocean-sim/sim2real/sceneAssets/duluth/duluth.yaml",
        "debug_mode": False,

    },
    "obj_workspace": {
        "min" : (-1.5, -2.3, 1.25),
        "max" : (1.5, 3.7, 1.65)
    },
    "cam_workspace" : {
        "min" : (-1.5, -2.3, 1.25),
        "max" : (1.5, 3.7, 1.5)
    },
    "object_mask_rate": 0.075, # The percentage of objects to be shown among all objects
}


# Check if there are any config files (yaml or json) are passed as arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=False, help="Include specific config parameters (json or yaml))")
args, unknown = parser.parse_known_args()
args_config = {}
if args.config and os.path.isfile(args.config):
    with open(args.config, "r") as f:
        if args.config.endswith(".json"):
            args_config = json.load(f)
        elif args.config.endswith(".yaml"):
            args_config = yaml.safe_load(f)
        else:
            carb.log_warn(f"File {args.config} is not json or yaml, will use default config")
else:
    carb.log_warn(f"File {args.config} does not exist, will use default config")

# Update the default config dict with the external one
config.update(args_config)

print(f"[SDG] Using config:\n{config}")

launch_config = config.get("launch_config", {})
simulation_app = SimulationApp(launch_config=launch_config)
from isaacsim.simulation_app import SimulationApp

# load up OceanSim
import isaacsim.core.utils.extensions as extensions_utils
value = extensions_utils.enable_extension(extension_name='isaacsim.oceansim')
if value:
    print("[SDG] OceanSim loaded successfully")
else:
    simulation_app.update()
    simulation_app.close()
    sys.exit("[SDG] OceanSim loaded failed. SDG Stopped...")

# Load an environment extension that some usd scenes will rely on

extensions_utils.enable_extension(extension_name="omni.kit.actions.core")
extensions_utils.enable_extension(extension_name="omni.kit.window.preferences")
extensions_utils.enable_extension(extension_name="omni.kit.widget.sliderbar")
extensions_utils.enable_extension(extension_name="omni.kit.viewport.utility")
extensions_utils.enable_extension(extension_name="omni.kit.usd.layers")
extensions_utils.enable_extension(extension_name="omni.rtx.window.settings")
extensions_utils.enable_extension(extension_name="omni.kit.notification_manager")
extensions_utils.enable_extension(extension_name="omni.kit.window.filepicker")
extensions_utils.enable_extension(extension_name="omni.kit.environment.core")
extensions_utils.enable_extension(extension_name="omni.kit.property.environment")
extensions_utils.enable_extension(extension_name="omni.kit.window.environment")

############################################
#### Here goes implementation of writer #### 
############################################
# Notice writer should not accept any parameters definition from config file


import csv
import io
from typing import List, Union

import carb
import numpy as np
import warp as wp

from omni.syntheticdata.scripts.SyntheticData import SyntheticData
import omni.replicator.core.scripts.functional as F
from omni.replicator.core import AnnotatorRegistry, WriterRegistry, BackendDispatch
from omni.replicator.core.scripts.writers import Writer
from isaacsim.oceansim.utils.UWrenderer_utils import *
from isaacsim.replicator.writers.scripts.utils import calculate_truncation_ratio_simple
import isaacsim.core.utils.rotations as rotations_utils


__version__ = "0.1.0"


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
        bbox_height_threshold: int = 10,
        bbox2d_partly_occluded_threshold: float = 0.5,
        bbox2d_fully_visible_threshold: float = 0.95,
        mapping_path: str = None,
        mapping_dict: dict = None,
        colorize_instance_segmentation: bool = False,
        semantic_filter_predicate: str = None,
        use_kitti_dir_names: bool = False,
        UW_param:Union[list, str] = [0.0, 0.31, 0.24, 0.05, 0.05, 0.2, 0.05, 0.05, 0.05 ],
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
        self._bbox_height_threshold = bbox_height_threshold
        self._bbox2d_partly_occluded_threshold = bbox2d_partly_occluded_threshold
        self._bbox2d_fully_visible_threshold = bbox2d_fully_visible_threshold
        self._use_kitti_dir_names = use_kitti_dir_names
        self._cuboid_keypoints_order = cuboid_keypoints_order
        self._debug_mode = debug_mode

        if self._debug_mode:
            self._CUBOID_KEYPOINT_COLORS = ["white", "red", "green", "blue", "yellow", "cyan", "magenta", "orange", "purple"]
            self._CUBOID_EDGE_COLORS = {"front": "red", "back": "blue", "connecting": "green"}
            self._debug_data = {}

        if isinstance(UW_param, str):
            with open(UW_param, 'r') as file:
                try:
                    # Load the YAML content
                    yaml_content = yaml.safe_load(file)
                    self._backscatter_value = wp.vec3f(*yaml_content['backscatter_value'])
                    self._atten_coeff = wp.vec3f(*yaml_content['atten_coeff'])
                    self._backscatter_coeff = wp.vec3f(*yaml_content['backscatter_coeff'])
                    self._UW_param = [*yaml_content['backscatter_value'], *yaml_content['atten_coeff'], *yaml_content['backscatter_coeff']]
                    print(f"Loaded render parameters {self._UW_param} from {UW_param}")
                except yaml.YAMLError as exc:
                    carb.log_error(f"Error reading render parameter YAML from {UW_param} file: {exc}")
                    self._UW_param = [0.0, 0.31, 0.24, 0.05, 0.05, 0.2, 0.05, 0.05, 0.05 ]
                    carb.log_error(f"Fallback to default: {self._UW_param}")
        else:
            self._UW_param = UW_param
            print(f"Using render param {self._UW_param}")
        
        self._device = str(wp.get_preferred_device())
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
                    "UNLABELLED": (0, 0, 0, 255),
                    "BACKGROUND": (0, 0, 0, 0),
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
                "rgb", device=self._device
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
                "distance_to_camera", device=self._device
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

    def _write_rgb(self, data, sub_dir: str, rgb_annotator: str, dist_to_cam_annotator:str, UW_param: list):

        if self._debug_mode:
            self._debug_data["raw_rgb"] = data[rgb_annotator].numpy()

        uw_image = wp.empty(shape=data[rgb_annotator].shape, dtype=wp.uint8)
        uw_rgb_dir_name = "uw_image_02" if self._use_kitti_dir_names else "uw_rgb"
        uw_rgb_file_path = os.path.join(sub_dir, uw_rgb_dir_name, f"{self._frame_id}.png")
        wp.launch(
                dim=data[rgb_annotator].shape[:2],
                kernel=UW_render,
                inputs=[
                    data[rgb_annotator],
                    data[dist_to_cam_annotator],
                    wp.vec3f(*UW_param[0:3]),
                    wp.vec3f(*UW_param[3:6]),
                    wp.vec3f(*UW_param[6:9])
                ],
                outputs=[
                    uw_image
                ]
            )  
        self._backend.schedule(F.write_image, data=uw_image, path=uw_rgb_file_path)

    def _write_object_pose(self, data, sub_dir: str, bbox_3d_annotator: str, camera_param_annotator: str):
        objs_data = self._process_bounding_boxes_3d(data[bbox_3d_annotator], data[camera_param_annotator])
        pose_dir_name = "pose_02" if self._use_kitti_dir_names else "pose"
        pose_file_path = os.path.join(sub_dir, pose_dir_name, f"{self._frame_id}.json")
        self._backend.schedule(F.write_json, path=pose_file_path, data=objs_data, indent=2)


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

            # Skip boxes shorter than threshold pixels in height
            if box_tight["y_max"] - box_tight["y_min"] < self._bbox_height_threshold:
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


            # Check if bounding boxes are in the viewport
            if (
                box_tight["x_min"] < 0
                or box_tight["y_min"] < 0
                or box_tight["x_max"] > rp_width
                or box_tight["y_max"] > rp_height
                or box_tight["x_min"] > rp_width
                or box_tight["y_min"] > rp_height
                or box_tight["y_max"] < 0
                or box_tight["x_max"] < 0
            ):
                continue
            
            # Only compute object's 3d information after the above test
            bbox3d_info = self._process_bounding_box_3d_single(bbox3d_id_to_bbox[id], data[camera_param_annotator])
            

            
            semantic_label = data[bbox_2d_tight_annotator]["info"]["idToLabels"].get(box_tight["semanticId"])

            if self._omit_semantic_type:
                # omit semantic type
                semantic_label = semantic_label.get("class", "Unlabelled")
            

            # Adding Kitti Data,  NOTE: Only class and 2d bbox coordinates are filled in
            label.append(semantic_label)  # semantic
            label.append(f"{bbox3d_info['truncation_ratio']:.2f}")  # truncated
            label.append(occlusion_estimation)  # occluded (estimation, NOT ACCURATE!)
            label.append(f"{bbox3d_info['alpha']:.2f}")  # alpha 
            label.append(box_tight["x_min"])  # x min
            label.append(box_tight["y_min"])  # y min
            label.append(box_tight["x_max"])  # x max
            label.append(box_tight["y_max"])  # y max
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
        sem_rgb_dir_name = "semantic_rgb" if self._use_kitti_dir_names else "semantic_segmentation"
        inst_dir_name = "instance" if self._use_kitti_dir_names else "instance_segmentation"
        seg_filepath = os.path.join(sub_dir, "semantic", f"{self._frame_id}.png")
        seg_col_filepath = os.path.join(sub_dir, sem_rgb_dir_name, f"{self._frame_id}.png")
        seg_mapping_filepath = os.path.join(sub_dir, sem_rgb_dir_name, "semantic_mapping.json")

        inst_filepath = os.path.join(sub_dir, inst_dir_name, f"{self._frame_id}.png")

        inst_id_to_labels = data[inst_annotator]["info"]["idToSemantics"]
        self._backend.schedule(F.write_image, data=data[sem_annotator]["data"], path=seg_col_filepath)
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
        semantic_classes = list(self.mapping_dict.keys())
        inst_seg_uint32 = inst_seg_img.view(np.uint32).squeeze()
        inst_seg_img_renumbered = np.zeros((height, width), dtype=np.uint16)
        sem_seg_img_renumbered = np.zeros((height, width), dtype=np.uint8)
        for i, iid in enumerate(instance_ids):
            semantic_class = inst_id_to_labels[iid].get("class", "unlabelled")
            is_unlabelled = semantic_class.lower() == "unlabelled"
            is_background = semantic_class.lower() == "background"
            is_in_mapping = semantic_class in self.mapping_dict
            if not is_in_mapping or is_unlabelled or is_background:
                inst_seg_img_renumbered[inst_seg_uint32 == iid] = 0
            else:
                cur_semantics = str(inst_id_to_labels[iid])
                cur_idx.setdefault(cur_semantics, 0)
                cur_idx[cur_semantics] += 1
                semantics_renumbered = semantic_classes.index(semantic_class)
                inst_seg_img_renumbered[inst_seg_uint32 == iid] = cur_idx[cur_semantics] + semantics_renumbered * 256
                sem_seg_img_renumbered[inst_seg_uint32 == iid] = semantics_renumbered

        self._backend.schedule(F.write_image, data=inst_seg_img_renumbered, path=inst_filepath)
        self._backend.schedule(F.write_image, data=sem_seg_img_renumbered, path=seg_filepath)

    def _write_distance_to_camera(self, data, sub_dir: str, annotator: str):
        distance_to_camera_metres = data[annotator].numpy()
        distance_to_camera_metres = np.nan_to_num(distance_to_camera_metres, posinf=0.0)
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
        render_products = [k for k in data.keys() if k.startswith("rp_")]
        if len(render_products) == 1:
            sub_dir = data[render_products[0]]["camera"].split("/")[-1]
            self._write_rgb(data, sub_dir, "rgb", "distance_to_camera", self._UW_param)
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
                    self._UW_param
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


WriterRegistry.register(UWCam_KittiWriter)


#########################################################
#### Here goes scene construction and randomnization ####
#########################################################


import time
import omni.replicator.core as rep
from pxr import PhysxSchema, Sdf, UsdGeom, UsdPhysics, Gf
from isaacsim.oceansim.utils.UWCam_sdg_utils import *
import omni.usd

# Increase maximum assets loading time in case assets are too many
carb.settings.get_settings().set('/exts/omni.replicator.core/maxAssetLoadingTime', 1000)


# ENVIRONMENT
# Create an empty or load a custom stage (clearing any previous semantics)
env_url = config.get("env_url", "")
if env_url:
    omni.usd.get_context().open_stage(env_url)
    stage = omni.usd.get_context().get_stage()
else:
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    # Add a distant light to the empty stage
    distant_light = stage.DefinePrim("/World/Lights/DistantLight", "DistantLight")
    distant_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(400.0)
    if not distant_light.HasAttribute("xformOp:rotateXYZ"):
        UsdGeom.Xformable(distant_light).AddRotateXYZOp()
    distant_light.GetAttribute("xformOp:rotateXYZ").Set((0, 60, 0))

# Create a physics scene to modify custom physics settings
physics_scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/PhysicsScene"))
physx_scene.GetTimeStepsPerSecondAttr().Set(60)


num_frames = config.get("num_frames", 10)
rt_subframes = config.get("rt_subframes", -1) 
obj_ws = config.get("obj_workspace")
cam_ws = config.get("cam_workspace")
mask_object_rate = config.get("object_mask_rate", 1.0)

resolution = config.get("resolution", (640, 480))
# Create the writer and attach the render products
writer_type = config.get("writer_type", "BasicWriter")
writer_kwargs = config.get("writer_kwargs", {})

# If not an absolute path, set it relative to the current working directory
if out_dir := writer_kwargs.get("output_dir"):
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(os.getcwd(), out_dir)
        writer_kwargs["output_dir"] = out_dir
    print(f"[SDG] Writing data to: {out_dir}")
num_cameras = config.get("num_cameras", 1)
camera_properties_kwargs = config.get("camera_properties_kwargs", {})
# camera_collider_radius = config.get("camera_collider_radius", 0)



with rep.new_layer(name="SDG"):
    cameras = []
    render_products = []
    
    object_group, kitti_labels = add_COU_objects(physics=True)
    object_prims = object_group.get_output_prims()["prims"]
    num_objects = len(object_prims)
    print(f"[SDG] {num_objects} objects being added to the scene") 
    print(f"[SDG] KITTI labels: {kitti_labels}")

    for i in range(num_cameras):
        cam = rep.create.camera(**camera_properties_kwargs, name=f"Camera_{i}")
        rp = rep.create.render_product(cam, resolution)
        writer = rep.writers.get(writer_type)
        writer.initialize(**writer_kwargs, mapping_dict=kitti_labels)
        writer.attach(rp)
        cameras.append(cam)        
        render_products.append(rp)

        # This is buggy, adding a collider will mess up with the camera orientation in multi-camera SDG
        # Add collision spheres (disabled by default) to cameras to avoid objects overlaping with the camera view
        # if camera_collider_radius > 0:
            # cam_path = cam.get_output_prims()["prims"][0].GetPath().pathString
            # cam_collider = stage.DefinePrim(f"{cam_path}/CollisionSphere", "Sphere")
            # cam_collider.GetAttribute("radius").Set(camera_collider_radius)
            # add_colliders(cam_collider)
            # collision_api = UsdPhysics.CollisionAPI(cam_collider)
            # UsdGeom.Imageable(cam_collider).MakeInvisible()

    # Setup the camera and object groups
    camera_group = rep.create.group(cameras)

    print(f"[SDG] Camera group created with {num_cameras} cameras.")
    

    
    with rep.trigger.on_custom_event(event_name="randomize_object"):
        with object_group:
            rep.modify.pose(
                position=rep.distribution.uniform(obj_ws.get("min"), obj_ws.get("max")),
                rotation=rep.distribution.uniform((0, 0, 0), (360, 360, 360)),
                scale=rep.distribution.uniform((0.5, 0.5, 0.5), (1.5, 1.5, 1.5)),
            )
            rep.modify.visibility(False)

    
    with rep.trigger.on_custom_event(event_name="show_random_object_and_camera_look_at"):
        shown_objects = rep.distribution.choice(object_prims, num_samples=int(num_objects * mask_object_rate))

        
        with shown_objects:
            rep.modify.visibility(True)
        

        # TODO: This is not working, the camera is not looking at the object
        # Stupid rep.distribution.choice just output empty list
        with camera_group:
            rep.modify.pose(
                position=rep.distribution.uniform(cam_ws.get("min"), cam_ws.get("max")),
                look_at=rep.distribution.choice(shown_objects)
            )    



# Data will be captured manually using step
rep.orchestrator.set_capture_on_play(False)
# Set the timeline parameters (start, end, no looping) and start the timeline

print(f"[SDG] Call the randomizer once and update few frames to heat up the simulation")
rep.utils.send_og_event(event_name="randomize_object")
rep.utils.send_og_event(event_name="show_random_object_and_camera_look_at")

for _ in range(100):
    simulation_app.update()

print(f"[SDG] Timeline starts. Running the simulation for {num_frames} frames")
timeline = omni.timeline.get_timeline_interface()
timeline.set_start_time(0)
timeline.set_end_time(1e8)
timeline.set_looping(False)
timeline.play()
timeline.commit()

wall_time_start = time.perf_counter()

for i in range(num_frames):
    rep.utils.send_og_event(event_name="randomize_object")


    for _ in range(3):
        simulation_app.update()
    
    rep.utils.send_og_event(event_name="show_random_object_and_camera_look_at")

    # set_render_products_updates(render_products, True, include_viewport=True)
    print(f"[SDG] Capturing frame {i}/{num_frames}, at simulation time: {timeline.get_current_time():.2f}")
    # The step function provides new data to the annotators, triggers the randomizers and the writer
    rep.orchestrator.step(rt_subframes=rt_subframes)
    # set_render_products_updates(render_products, False, include_viewport=True)



set_render_products_updates(render_products, True, include_viewport=True)

rep.orchestrator.wait_until_complete()
# Get the stats
wall_duration = time.perf_counter() - wall_time_start
avg_frame_fps = num_frames / wall_duration

print(
    f"[SDG] Captured {num_frames} frames in {wall_duration:.2f} seconds.\n"
    f"\t Simulation duration: {timeline.get_current_time():.2f}\n"
    f"\t Average frame FPS: {avg_frame_fps:.2f}\n"
)

timeline.stop()
simulation_app.update()
simulation_app.close()




