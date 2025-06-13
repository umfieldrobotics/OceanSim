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


# Default config dict, can be updated/replaced using json/yaml config files ('--config' cli argument)
config = {
    "launch_config": {
        "renderer": "RaytracedLighting",
        "headless": True,
    },
    "env_url": "/Isaac/Environments/Simple_Warehouse/warehouse.usd",
    "rt_subframes": 4,
    "num_frames": 2,
    "num_cameras": 1, 
    "distractors": "warehouse",
    "simulation_duration_between_captures": 0.05,
    "resolution": (1920, 1080),
    "camera_properties_kwargs": {
        "focalLength": 24.0,
        "focusDistance": 400,
        "fStop": 0.0,
        "clippingRange": (0.01, 10000),
    },
    "writer_type": "UWCam_KittiWriter",
    "writer_kwargs": {
        "output_dir": "/home/haoyu/Desktop/viz/",
        "colorize_instance_segmentation": True
    },
    "UW_param": [0.0, 0.31, 0.24, 0.05, 0.05, 0.2, 0.05, 0.05, 0.05 ]
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

# load up OceanSim
import isaacsim.core.utils.extensions as extensions_utils
value = extensions_utils.enable_extension(extension_name='isaacsim.oceansim')
if value:
    print("OceanSim loaded successfully")
else:
    simulation_app.update()
    simulation_app.close()
    sys.exit("OceanSim loaded failed. SDG Stopped...")

############################################
#### Here goes implementation of writer #### 
############################################
# Notice writer should not accept any parameters definition from config file
# In future, move writier class into modules of OceanSim

import csv
import io
from typing import List

import carb
import numpy as np
import warp as wp

from omni.syntheticdata.scripts.SyntheticData import SyntheticData
import omni.replicator.core.scripts.functional as F
from omni.replicator.core import AnnotatorRegistry, WriterRegistry, BackendDispatch
from omni.replicator.core.scripts.writers import Writer
from isaacsim.oceansim.utils.UWrenderer_utils import *

EPS = 1e-5
# Procuring standard KITTI Labels for objects annotated in the KITTI-format
# The dictionary is ordered where label idx corresponds to semantic ID
# See https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
KITTI_LABELS = {
    "UNLABELLED": (0, 0, 0, 0),
    "BACKGROUND": (0, 0, 0, 0),
    "wall": (0, 0, 0, 0),
    "pillar": (0, 0, 0, 0),
    "floor": (0, 0, 0, 0),
    "floor_decal": (0, 0, 0, 0),
    "klt_bin": (111, 74, 0, 255),
    "palletjack": (81, 0, 81, 255),
    "cone": (128, 64, 128, 255),
    "sign": (244, 35, 232, 255),
    "barel": (250, 170, 160, 255),
    "bottle": (230, 150, 140, 255),
    "box": (70, 70, 70, 255),
    "crate": (102, 102, 156, 255),
    "cart": (190, 153, 153, 255),
    "rack": (180, 165, 180, 255),
    "sign": (150, 100, 100, 255),
    "bucket": (150, 120, 90, 255),
    "wire": (153, 153, 153, 255),
    "pallet": (153, 153, 153, 255),
    "fire_extinguisher": (250, 170, 30, 255),
    "fuse_box": (220, 220, 0, 255),

}


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
        - **Unsupported:** alpha, dimensions, location, rotation_y, truncated (all set to default values of ``0.0``)
    """

    def __init__(
        self,
        output_dir: str,
        s3_bucket: str = None,
        s3_region: str = None,
        s3_endpoint: str = None,
        semantic_types: List[str] = None,
        omit_semantic_type: bool = False,
        bbox_height_threshold: int = 25,
        partly_occluded_threshold: float = 0.5,
        fully_visible_threshold: float = 0.95,
        renderproduct_idxs: List[tuple] = None,
        mapping_path: str = None,
        mapping_dict: dict = None,
        colorize_instance_segmentation: bool = False,
        semantic_filter_predicate: str = None,
        use_kitti_dir_names: bool = False,
        UW_param:list = [0.0, 0.31, 0.24, 0.05, 0.05, 0.2, 0.05, 0.05, 0.05 ]
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
        self._partly_occluded_threshold = partly_occluded_threshold
        self._fully_visible_threshold = fully_visible_threshold
        self._render_product_idxs = renderproduct_idxs
        self._use_kitti_dir_names = use_kitti_dir_names
        self._UW_param = UW_param
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
            self.mapping_dict = KITTI_LABELS
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

    def _write_rgb(self, data, sub_dir: str, rgb_annotator: str, dist_to_cam_annotator:str, UW_param: list, write_raw:bool =False):
        if write_raw:
            rgb_dir_name = "image_02" if self._use_kitti_dir_names else "rgb"
            rgb_file_path = os.path.join(sub_dir, rgb_dir_name, f"{self._frame_id}.png")
            self._backend.schedule(F.write_image, data=data[rgb_annotator], path=rgb_file_path)
        
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
                    wp.vec3f(*UW_param[6:9]),
                    wp.vec3f(*UW_param[3:6])
                ],
                outputs=[
                    uw_image
                ]
            )  
        self._backend.schedule(F.write_image, data=uw_image, path=uw_rgb_file_path)
    
    def _write_object_detection(
        self,
        data,
        sub_dir: str,
        render_product_annotator: str,
        bbox_2d_tight_annotator: str,
        bbox_2d_loose_annotator: str,
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

        bbox_tight_bbox_ids = data[bbox_2d_tight_annotator]["info"]["bboxIds"]
        bbox_loose_bbox_ids = data[bbox_2d_loose_annotator]["info"]["bboxIds"]

        # For box in tight, find the corresponding index of box in loose
        bbox_loose_indices = np.where(np.isin(bbox_loose_bbox_ids, bbox_tight_bbox_ids))[0]
        selected_bbox_loose = bbox_loose[bbox_loose_indices]

        for box_tight, box_loose in zip(bbox_tight, selected_bbox_loose):

            label = []

            # Skip boxes shorter than threshold pixels in height
            if box_tight["y_max"] - box_tight["y_min"] < self._bbox_height_threshold:
                continue

            area_tight = (box_tight["x_max"] - box_tight["x_min"]) * (box_tight["y_max"] - box_tight["y_min"])
            area_loose = (box_loose["x_max"] - box_loose["x_min"]) * (box_loose["y_max"] - box_loose["y_min"])
            area_ratio = area_tight / (area_loose + EPS)

            if area_ratio >= self._fully_visible_threshold:
                occlusion_estimation = 0
            elif area_ratio >= self._partly_occluded_threshold:
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

            semantic_label = data[bbox_2d_tight_annotator]["info"]["idToLabels"].get(box_tight["semanticId"])

            if self._omit_semantic_type:
                # omit semantic type
                semantic_label = semantic_label.get("class", "Unlabelled")

            # Adding Kitti Data,  NOTE: Only class and 2d bbox coordinates are filled in
            label.append(semantic_label)  # semantic
            label.append(f"{0.00:.2f}")  # truncated (not supported)
            label.append(occlusion_estimation)  # occluded (estimation)
            label.append(f"{0.00:.2f}")  # alpha (not supported)
            label.append(box_tight["x_min"])  # x min
            label.append(box_tight["y_min"])  # y min
            label.append(box_tight["x_max"])  # x max
            label.append(box_tight["y_max"])  # y max
            for _ in range(7):
                label.append(f"{0.00:.2f}")  # dimensions, location, rotation_y, score

            label_set.append(label)

        det_dir_name = "label_02" if self._use_kitti_dir_names else "object_detection"
        kitti_filepath = os.path.join(sub_dir, det_dir_name, f"{self._frame_id}.txt")
        buf = io.StringIO()

        writer = csv.writer(buf, delimiter=" ")
        writer.writerows(label_set)

        self._backend.schedule(self._backend.write_blob, data=bytes(buf.getvalue(), "utf-8"), path=kitti_filepath)

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
        inst_filepath = os.path.join(sub_dir, inst_dir_name, f"{self._frame_id}.png")
        inst_col_filepath = os.path.join(sub_dir, "instance_rgb", f"{self._frame_id}.png")

        inst_id_to_labels = data[inst_annotator]["info"]["idToSemantics"]
        self._backend.schedule(F.write_image, data=data[sem_annotator]["data"], path=seg_col_filepath)

        inst_seg_img = data[inst_annotator]["data"]
        height, width = inst_seg_img.shape[:2]

        if self.colorize_instance_segmentation:
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
            is_in_mapping = semantic_class in self.mapping_dict
            if not is_in_mapping or is_unlabelled:
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


        file_path = os.path.join(sub_dir, "camera_param", f"{self._frame_id}.json")
        self._backend.schedule(F.write_json, data=camera_data, path=file_path)

    def write(self, data):
        render_products = [k for k in data.keys() if k.startswith("rp_")]
        if len(render_products) == 1:
            sub_dir = data[render_products[0]]["camera"].split("/")[-1]
            self._write_rgb(data, sub_dir, "rgb", "distance_to_camera", self._UW_param)
            self._write_segmentation(data, sub_dir, "semantic_segmentation", "instance_segmentation_fast")
            self._write_object_detection(
                data, sub_dir, render_products[0], "bounding_box_2d_tight_fast", "bounding_box_2d_loose_fast"
            )
            self._write_distance_to_camera(data, sub_dir, "distance_to_camera")
            self._write_camera_param(data, sub_dir, "camera_params")
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
                )
                self._write_distance_to_camera(data, sub_dir, f"distance_to_camera-{render_product_name}")
                self._write_camera_param(data, sub_dir, f"camera_params-{render_product_name}")

        self._frame_id += 1

WriterRegistry. register(UWCam_KittiWriter)


#########################################################
#### Here goes scene construction and randomnization ####
#########################################################


import time
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.storage.native import get_assets_root_path
import isaacsim.core.utils.prims as prims_utils
from pxr import PhysxSchema, Sdf, UsdGeom, UsdPhysics, Gf




# This is the location of the palletjacks in the simready asset library
PALLETJACKS = [
    "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/DigitalTwin/Assets/Warehouse/Equipment/Pallet_Trucks/Scale_A/PalletTruckScale_A01_PR_NVD_01.usd",
    "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/DigitalTwin/Assets/Warehouse/Equipment/Pallet_Trucks/Heavy_Duty_A/HeavyDutyPalletTruck_A01_PR_NVD_01.usd",
    "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/DigitalTwin/Assets/Warehouse/Equipment/Pallet_Trucks/Low_Profile_A/LowProfilePalletTruck_A01_PR_NVD_01.usd",
]


# The warehouse distractors which will be added to the scene and randomized
DISTRACTORS_WAREHOUSE = 2 * [
    "/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/S_WetFloorSign.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_03.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_B_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_B_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_B_03.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_C_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticA_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticB_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticA_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticA_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticD_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticE_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BucketPlastic_B.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_1262.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_1268.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_1482.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_1683.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_291.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_01_1454.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_01_1513.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_A_04.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_B_03.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_B_05.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_C_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_E_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_PushcartA_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_RackPile_04.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_RackPile_03.usd",
]


## Additional distractors which can be added to the scene
DISTRACTORS_ADDITIONAL = [
    "/Isaac/Environments/Hospital/Props/Pharmacy_Low.usd",
    "/Isaac/Environments/Hospital/Props/SM_BedSideTable_01b.usd",
    "/Isaac/Environments/Hospital/Props/SM_BooksSet_26.usd",
    "/Isaac/Environments/Hospital/Props/SM_BottleB.usd",
    "/Isaac/Environments/Hospital/Props/SM_BottleA.usd",
    "/Isaac/Environments/Hospital/Props/SM_BottleC.usd",
    "/Isaac/Environments/Hospital/Props/SM_Cart_01a.usd",
    "/Isaac/Environments/Hospital/Props/SM_Chair_02a.usd",
    "/Isaac/Environments/Hospital/Props/SM_Chair_01a.usd",
    "/Isaac/Environments/Hospital/Props/SM_Computer_02b.usd",
    "/Isaac/Environments/Hospital/Props/SM_Desk_04a.usd",
    "/Isaac/Environments/Hospital/Props/SM_DisposalStand_02.usd",
    "/Isaac/Environments/Hospital/Props/SM_FirstAidKit_01a.usd",
    "/Isaac/Environments/Hospital/Props/SM_GasCart_01c.usd",
    "/Isaac/Environments/Hospital/Props/SM_Gurney_01b.usd",
    "/Isaac/Environments/Hospital/Props/SM_HospitalBed_01b.usd",
    "/Isaac/Environments/Hospital/Props/SM_MedicalBag_01a.usd",
    "/Isaac/Environments/Hospital/Props/SM_Mirror.usd",
    "/Isaac/Environments/Hospital/Props/SM_MopSet_01b.usd",
    "/Isaac/Environments/Hospital/Props/SM_SideTable_02a.usd",
    "/Isaac/Environments/Hospital/Props/SM_SupplyCabinet_01c.usd",
    "/Isaac/Environments/Hospital/Props/SM_SupplyCart_01e.usd",
    "/Isaac/Environments/Hospital/Props/SM_TrashCan.usd",
    "/Isaac/Environments/Hospital/Props/SM_Washbasin.usd",
    "/Isaac/Environments/Hospital/Props/SM_WheelChair_01a.usd",
    "/Isaac/Environments/Office/Props/SM_WaterCooler.usd",
    "/Isaac/Environments/Office/Props/SM_TV.usd",
    "/Isaac/Environments/Office/Props/SM_TableC.usd",
    "/Isaac/Environments/Office/Props/SM_Recliner.usd",
    "/Isaac/Environments/Office/Props/SM_Personenleitsystem_Red1m.usd",
    "/Isaac/Environments/Office/Props/SM_Lamp02_162.usd",
    "/Isaac/Environments/Office/Props/SM_Lamp02.usd",
    "/Isaac/Environments/Office/Props/SM_HandDryer.usd",
    "/Isaac/Environments/Office/Props/SM_Extinguisher.usd",
]


# The textures which will be randomized for the wall and floor
TEXTURES = [
    "/Isaac/Materials/Textures/Patterns/nv_asphalt_yellow_weathered.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_tile_hexagonal_green_white.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_rubber_woven_charcoal.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_granite_tile.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_tile_square_green.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_marble.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_brick_reclaimed.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_concrete_aged_with_lines.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_wooden_wall.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_stone_painted_grey.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_wood_shingles_brown.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_tile_hexagonal_various.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_carpet_abstract_pattern.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_wood_siding_weathered_green.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_animalfur_pattern_greys.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_artificialgrass_green.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_bamboo_desktop.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_brick_reclaimed.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_brick_red_stacked.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_fireplace_wall.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_fabric_square_grid.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_granite_tile.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_marble.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_gravel_grey_leaves.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_plastic_blue.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_stone_red_hatch.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_stucco_red_painted.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_rubber_woven_charcoal.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_stucco_smooth_blue.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_wood_shingles_brown.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_wooden_wall.jpg",
]



# needed for loading textures correctly
def prefix_with_isaac_asset_server(relative_path):
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        raise Exception(
            "Nucleus server not found, could not access Isaac Sim assets folder"
        )
    return assets_root_path + relative_path


def full_distractors_list(distractor_type="warehouse"):
    """Distractor type allowed are warehouse, additional or None. They load corresponding objects and add
    them to the scene for DR"""
    full_dist_list = []

    if distractor_type == "warehouse":
        for distractor in DISTRACTORS_WAREHOUSE:
            full_dist_list.append(prefix_with_isaac_asset_server(distractor))
    elif distractor_type == "additional":
        for distractor in DISTRACTORS_ADDITIONAL:
            full_dist_list.append(prefix_with_isaac_asset_server(distractor))
    else:
        print("No Distractors being added to the current scene for SDG")

    return full_dist_list


def full_textures_list():
    full_tex_list = []
    for texture in TEXTURES:
        full_tex_list.append(prefix_with_isaac_asset_server(texture))

    return full_tex_list


def add_palletjacks():
    rep_obj_list = [
        rep.create.from_usd(
            palletjack_path, semantics=[("class", "palletjack")], count=2
        )
        for palletjack_path in PALLETJACKS
    ]
    rep_palletjack_group = rep.create.group(rep_obj_list)
    return rep_palletjack_group


def add_distractors(distractor_type="warehouse"):
    full_distractors = full_distractors_list(distractor_type)
    distractors = [
        rep.create.from_usd(distractor_path, count=1)
        for distractor_path in full_distractors
    ]
    distractor_group = rep.create.group(distractors)
    return distractor_group


def capture_pathtracing(duration=0.0, spp=128):

    # Set the render mode to PathTracing
    prev_render_mode = carb.settings.get_settings().get("/rtx/rendermode")
    carb.settings.get_settings().set("/rtx/pathtracing/clampSpp", 0)
    carb.settings.get_settings().set("/rtx/rendermode", "PathTracing")
    carb.settings.get_settings().set("/rtx/pathtracing/spp", spp)
    carb.settings.get_settings().set("/rtx/pathtracing/totalSpp", spp)
    carb.settings.get_settings().set("/rtx/pathtracing/optixDenoiser/enabled", 0)

    # Make sure the timeline is playing
    if not timeline.is_playing():
        timeline.play()

    # Capture the frame by advancing the simulation for the given duration and combining the sub samples
    rep.orchestrator.step(delta_time=duration, pause_timeline=False)

    # Restore the previous render and motion blur  settings
    print(f"[SDG] Restoring render mode from 'PathTracing' to '{prev_render_mode}'")
    carb.settings.get_settings().set("/rtx/rendermode", prev_render_mode)


# Update the app until a given simulation duration has passed (simulate the world between captures)
def run_simulation_loop(duration):
    timeline = omni.timeline.get_timeline_interface()
    elapsed_time = 0.0
    previous_time = timeline.get_current_time()
    if not timeline.is_playing():
        timeline.play()
    app_updates_counter = 0
    while elapsed_time <= duration:
        simulation_app.update()
        elapsed_time += timeline.get_current_time() - previous_time
        previous_time = timeline.get_current_time()
        app_updates_counter += 1
        print(
            f"\t Simulation loop at {timeline.get_current_time():.2f}, current elapsed time: {elapsed_time:.2f}, counter: {app_updates_counter}"
        )
    print(
        f"[SDG] Simulation loop finished in {elapsed_time:.2f} seconds at {timeline.get_current_time():.2f} with {app_updates_counter} app updates."
    )



# Isaac nucleus assets root path
assets_root_path = get_assets_root_path()

# ENVIRONMENT
# Create an empty or load a custom stage (clearing any previous semantics)
env_url = config.get("env_url", "")
if env_url:
    omni.usd.get_context().open_stage(prefix_with_isaac_asset_server(env_url))
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

# REPLICATOR
# Disable capturing every frame (capture will be triggered manually using the step function)
rep.orchestrator.set_capture_on_play(False)

# Create the camera prims and their properties
cam = rep.create.camera()
cam_prim = prims_utils.get_prim_at_path('/Replicator/Camera_Xform')
camera_properties_kwargs = config.get("camera_properties_kwargs", {})
# TODO bug here, cam_prim is actaully its xform, not the actual camera
for key, value in camera_properties_kwargs.items():
    if cam_prim.HasAttribute(key):
        cam_prim.GetAttribute(key).Set(value)
    else:
        print(f"Unknown camera attribute with {key}:{value}")


# Wait an app update to ensure the prim changes are applied
simulation_app.update()

# Create render products using the cameras
resolution = config.get("resolution", (640, 480))
rp = rep.create.render_product(cam_prim.GetPath(), resolution)
# Create the writer and attach the render products
writer_type = config.get("writer_type", "BasicWriter")
writer_kwargs = config.get("writer_kwargs", {})
# If not an absolute path, set it relative to the current working directory
if out_dir := writer_kwargs.get("output_dir"):
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(os.getcwd(), out_dir)
        writer_kwargs["output_dir"] = out_dir
    print(f"[SDG] Writing data to: {out_dir}")

if writer_type is not None:
    writer = rep.writers.get(writer_type)
    writer.initialize(**writer_kwargs)
    writer.attach(rp)


# SDG
# Number of frames to capture
num_frames = config.get("num_frames", 10)

# Increase subframes if materials are not loaded on time, or ghosting artifacts appear on moving objects,
# see: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/subframes_examples.html
rt_subframes = config.get("rt_subframes", -1) 
# Because we are only using path tracing mode, otherwise we need to call 
# rep.orchestrator.step(delta_time=duration, rt_subframes=st_subframes, pause_timeline=False) 
# to suppress ghosting


# Amount of simulation time to wait between captures
sim_duration_between_captures = config.get("simulation_duration_between_captures", 0.0)
# Increase maximum assets loading time in case assets are too many
carb.settings.get_settings().set('/exts/omni.replicator.core/maxAssetLoadingTime', 1000)

for _ in range(5):
    simulation_app.update()

# Set up objects in the scene
textures = full_textures_list()
rep_palletjack_group = add_palletjacks()
rep_distractor_group = add_distractors(distractor_type=config.get('distractors', "warehouse"))

# Set the timeline parameters (start, end, no looping) and start the timeline
timeline = omni.timeline.get_timeline_interface()
timeline.set_start_time(0)
timeline.set_end_time(1000000)
timeline.set_looping(False)

# If no custom physx scene is created, a default one will be created by the physics engine once the timeline starts
timeline.play()
timeline.commit()
simulation_app.update()

# Store the wall start time for stats
wall_time_start = time.perf_counter()


# Run the simulation and capture data triggering randomizations and actions at custom frame intervals
for i in range(num_frames):
    
    # Move the camera around in the scene, focus on the center of warehouse
    with cam:
        rep.modify.pose(
            position=rep.distribution.uniform((-9.2, -11.8, 0.4), (7.2, 15.8, 4)),
            look_at=(0, 0, 0),
        )

    # Get the Palletjack body mesh and modify its color
    with rep.get.prims(path_pattern="SteerAxles"):
        rep.randomizer.color(colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))

    # Randomize the pose of all the added palletjacks
    with rep_palletjack_group:
        rep.modify.pose(
            position=rep.distribution.uniform((-6, -6, 0), (6, 12, 0)),
            rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 360)),
            scale=rep.distribution.uniform((0.01, 0.01, 0.01), (0.01, 0.01, 0.01)),
        )

    # Modify the pose of all the distractors in the scene
    with rep_distractor_group:
        rep.modify.pose(
            position=rep.distribution.uniform((-6, -6, 0), (6, 12, 0)),
            rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 360)),
            scale=rep.distribution.uniform(1, 1.5),
        )

    # Randomize the lighting of the scene
    with rep.get.prims(path_pattern="RectLight"):
        rep.modify.attribute(
            "color", rep.distribution.uniform((0, 0, 0), (1, 1, 1))
        )
        rep.modify.attribute(
            "intensity", rep.distribution.normal(100000.0, 600000.0)
        )
        rep.modify.visibility(
            rep.distribution.choice(
                [True, False, False, False, False, False, False]
            )
        )

    # select floor material
    random_mat_floor = rep.create.material_omnipbr(
        diffuse_texture=rep.distribution.choice(textures),
        roughness=rep.distribution.uniform(0, 1),
        metallic=rep.distribution.choice([0, 1]),
        emissive_texture=rep.distribution.choice(textures),
        emissive_intensity=rep.distribution.uniform(0, 1000),
    )

    with rep.get.prims(path_pattern="SM_Floor"):
        rep.randomizer.materials(random_mat_floor)

    # select random wall material
    random_mat_wall = rep.create.material_omnipbr(
        diffuse_texture=rep.distribution.choice(textures),
        roughness=rep.distribution.uniform(0, 1),
        metallic=rep.distribution.choice([0, 1]),
        emissive_texture=rep.distribution.choice(textures),
        emissive_intensity=rep.distribution.uniform(0, 1000),
    )

    with rep.get.prims(path_pattern="SM_Wall"):
        rep.randomizer.materials(random_mat_wall)
        

    # Capture the current frame
    print(f"[SDG] Capturing frame {i}/{num_frames}, at simulation time: {timeline.get_current_time():.2f}")
    capture_pathtracing()
    
    # Run the simulation for a given duration between frame captures
    if sim_duration_between_captures > 0:
        run_simulation_loop(duration=sim_duration_between_captures)
    else:
        simulation_app.update()

# Wait for the data to be written (default writer backends are asynchronous)
rep.orchestrator.wait_until_complete()






# Get the stats
wall_duration = time.perf_counter() - wall_time_start
sim_duration = timeline.get_current_time()
avg_frame_fps = num_frames / wall_duration

print(
    f"[SDG] Captured {num_frames} frames in {wall_duration:.2f} seconds.\n"
    f"\t Simulation duration: {sim_duration:.2f}\n"
    f"\t Simulation duration between captures: {sim_duration_between_captures:.2f}\n"
    f"\t Average frame FPS: {avg_frame_fps:.2f}\n"
)

# Unsubscribe the physics overlap checks and stop the timeline
simulation_app.update()
timeline.stop()

simulation_app.close()




