__copyright__ = "Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved."
__license__ = """
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

import csv
import io
import json
import os
import threading
from typing import List

import carb
import numpy as np
from omni.syntheticdata.scripts.SyntheticData import SyntheticData
import omni.replicator.core.scripts.functional as F
from omni.replicator.core import AnnotatorRegistry
from omni.replicator.core import BackendDispatch
from omni.replicator.core.scripts.writers import Writer

EPS = 1e-5
# Procuring standard KITTI Labels for objects annotated in the KITTI-format
# The dictionary is ordered where label idx corresponds to semantic ID
# See https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
KITTI_LABELS = {
    "unlabelled": (0, 0, 0, 0),
    "ego vehicle": (0, 0, 0, 0),
    "rectification border": (0, 0, 0, 0),
    "out of roi": (0, 0, 0, 0),
    "static": (0, 0, 0, 0),
    "dynamic": (111, 74, 0, 255),
    "ground": (81, 0, 81, 255),
    "road": (128, 64, 128, 255),
    "sidewalk": (244, 35, 232, 255),
    "parking": (250, 170, 160, 255),
    "rail track": (230, 150, 140, 255),
    "building": (70, 70, 70, 255),
    "wall": (102, 102, 156, 255),
    "fence": (190, 153, 153, 255),
    "guard rail": (180, 165, 180, 255),
    "bridge": (150, 100, 100, 255),
    "tunnel": (150, 120, 90, 255),
    "pole": (153, 153, 153, 255),
    "polegroup": (153, 153, 153, 255),
    "traffic light": (250, 170, 30, 255),
    "traffic sign": (220, 220, 0, 255),
    "vegetation": (107, 142, 35, 255),
    "terrain": (152, 251, 152, 255),
    "background": (70, 130, 180, 255),  # Sky is always labelled as BACKGROUND
    "person": (220, 20, 60, 255),
    "rider": (255, 0, 0, 255),
    "car": (0, 0, 142, 255),
    "truck": (0, 0, 70, 255),
    "bus": (0, 60, 100, 255),
    "caravan": (0, 0, 90, 255),
    "trailer": (0, 0, 110, 255),
    "train": (0, 80, 100, 255),
    "motorcycle": (0, 0, 230, 255),
    "bicycle": (119, 11, 32, 255),
    "license plate": (0, 0, 142, 255),
}


__version__ = "0.0.1"


class FLS_KittiWriter(Writer):
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
    
    .. my note::
        - occlusion in sonar image can not be achieved now 
        (future: maybe initialize a larger chunck of memory at each pixel when processing semantic information from every rayquery? )
        - dimensions and locations of the object should be possible. (though this is priviledged information for sonar, but useful for sensor fusion)
        By Transferring 3D bbox information to local frame 
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
        self._sqlite_lock = threading.Lock()
        self._omit_semantic_type = omit_semantic_type
        self._bbox_height_threshold = bbox_height_threshold
        self._partly_occluded_threshold = partly_occluded_threshold
        self._fully_visible_threshold = fully_visible_threshold
        self._render_product_idxs = renderproduct_idxs
        self._use_kitti_dir_names = use_kitti_dir_names
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
            "rgb",
            "bounding_box_2d_tight_fast",
            "bounding_box_2d_loose_fast",
            AnnotatorRegistry.get_annotator(
                "semantic_segmentation", init_params={"mapping": self._get_anno_semantic_mapping()}
            ),
            AnnotatorRegistry.get_annotator(
                "instance_segmentation_fast", init_params={"colorize": colorize_instance_segmentation}
            ),
            "distance_to_camera",
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

    def _write_rgb(self, data, sub_dir: str, annotator: str):
        # Save the rgb data under the correct path
        rgb_dir_name = "image_02" if self._use_kitti_dir_names else "rgb"
        rgb_file_path = os.path.join(sub_dir, rgb_dir_name, f"{self._frame_id}.png")
        self._backend.schedule(F.write_image, data=data[annotator], path=rgb_file_path)

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
        distance_to_camera_metres = data[annotator]
        distance_to_camera_metres = np.nan_to_num(distance_to_camera_metres, posinf=0.0)
        distance_to_camera_uint16 = (distance_to_camera_metres * 256).astype(np.uint16)
        file_path = os.path.join(sub_dir, "depth", f"{self._frame_id}.png")
        self._backend.schedule(F.write_image, data=distance_to_camera_uint16, path=file_path)

    def write(self, data):
        render_products = [k for k in data.keys() if k.startswith("rp_")]
        if len(render_products) == 1:
            sub_dir = data[render_products[0]]["camera"].split("/")[-1]
            self._write_rgb(data, sub_dir, "rgb")
            self._write_segmentation(data, sub_dir, "semantic_segmentation", "instance_segmentation_fast")
            self._write_object_detection(
                data, sub_dir, render_products[0], "bounding_box_2d_tight_fast", "bounding_box_2d_loose_fast"
            )
            self._write_distance_to_camera(data, sub_dir, "distance_to_camera")
        else:
            for render_product in render_products:
                render_product_name = render_product[3:]
                sub_dir = os.path.join(render_product_name, data[render_product]["camera"].split("/")[-1])
                self._write_rgb(data, sub_dir, f"rgb-{render_product_name}")
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

        self._frame_id += 1
