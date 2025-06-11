# ./python.sh /isaac-sim/palletjack_sdg/standalone_palletjack_sdg.py --headless True --height 544 --width 960 --num_frames 1000 --distractors None --data_dir /isaac-sim/palletjack_sdg/palletjack_data/no_distractors

# Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
#
#  SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from omni.isaac.kit import SimulationApp
import os
import argparse

parser = argparse.ArgumentParser("Dataset generator")
parser.add_argument(
    "--headless",
    type=bool,
    default=False,
    help="Launch script headless, default is False",
)
parser.add_argument("--height", type=int, default=544, help="Height of image")
parser.add_argument("--width", type=int, default=960, help="Width of image")
parser.add_argument(
    "--num_frames", type=int, default=10, help="Number of frames to record"
)
parser.add_argument(
    "--distractors",
    type=str,
    default="warehouse",
    help="Options are 'warehouse' (default), 'additional' or None",
)
parser.add_argument(
    "--data_dir",
    type=str,
    default=os.getcwd() + "/_palletjack_data",
    help="Location where data will be output",
)

args, unknown_args = parser.parse_known_args()

# This is the config used to launch simulation.
CONFIG = {
    "renderer": "RayTracedLighting",
    "headless": args.headless,
    "width": args.width,
    "height": args.height,
    "num_frames": args.num_frames,
}

simulation_app = SimulationApp(launch_config=CONFIG)
import isaacsim.core.utils.extensions as extensions_utils
value = extensions_utils.enable_extension(extension_name='isaacsim.oceansim')

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
from omni.replicator.core import AnnotatorRegistry, WriterRegistry, BackendDispatch
from omni.replicator.core.scripts.writers import Writer

# Import sonart rendering kernel
from isaacsim.oceansim.utils.ImagingSonar_kernels import *

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
    TODO:
    - 3d bbox of object in sensor frame -> dimensions, location, rotation_y (OceanSim) why Nvidia doesn't implement this writing even they have 3DBbox annotator

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
        (future: process occlusion can only be done without parallelism ie. can not be on cuda, 
        otherwise to do in cuda, we have to initialize a really big memory by extending one dimention equal to the total number of semantics )
        - dimensions and locations of the object should be possible. 
        by Transferring 3D bbox information to local frame 
    """

    def __init__(
        self,
        output_dir: str,
        # configuration of sonar renderings
        sonar_param: dict = {"max_range": 3, 
                             "min_range": 0.2,
                             "hori_fov": 130, # Notice: on camera end, hori_fov and vert_fov is required to 
                             "vert_fov": 20,  # compute camera AR and vert_res given arbitrary hori_res
                             "range_res": 0.005, 
                             "angular_res": 0.25,
                             "normalizing_method": "range",
                             "query_prop": "reflectivity", # bit wanky, leave this for now
                             "attenuation": 0.1,
                             "gau_noise_param": 0.2,
                             "ray_noise_param": 0.05,
                             "intensity_offset": 0.0,
                             "intensity_gain": 1.0,
                             "central_peak": 2,
                             "central_std": 0.001},
        # extra config for data writing
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
        include_unlabelled: bool = True,
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
        self.include_unlabelled = include_unlabelled
        self._device = str(wp.get_preferred_device())

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
            # We don't need these three annotators for they are for camera rendering 
            # "rgb",
            # "bounding_box_2d_tight_fast",
            # "bounding_box_2d_loose_fast",
            
            # We need pointcloud data as the result of rayquest
            AnnotatorRegistry.get_annotator(
                "pointcloud", init_params={"includeUnlabelled": include_unlabelled}, device=self._device
            ),
            # AnnotatorRegistry.get_annotator(
            #     "semantic_segmentation", init_params={"mapping": self._get_anno_semantic_mapping()}
            # ),
            AnnotatorRegistry.get_annotator(
                "semantic_segmentation",
            ),
            AnnotatorRegistry.get_annotator(
                "instance_segmentation_fast", init_params={"colorize": colorize_instance_segmentation}
            ),
            # "distance_to_camera",
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
        self.bin_semantics = wp.empty(shape=self.r.shape, dtype=wp.uint32)
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
        viewTransform = data[cameraParams_annot]['cameraViewTransform'].reshape(4,4).T # 4 by 4 np.ndarray extrinsic matrix
        idToLabels = data[semantic_seg_annot]["info"]["idToLabels"] # dict 
        print(pcl.shape)
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
        
        wp.launch(kernel=bin_semantics_process,
                dim=num_points,
                inputs=[
                    pcl_local_spher,
                    semantics,
                    pcl_bin_idx,
                    self.bin_min_zenith
                ],
                outputs=[
                    self.bin_semantics
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

    def _write_sonar(self, sub_dir: str):
        # Save the rgb data under the correct path
        rgb_dir_name = "image_02" if self._use_kitti_dir_names else "rgb"
        rgb_file_path = os.path.join(sub_dir, rgb_dir_name, f"{self._frame_id}.png")
        self._backend.schedule(F.write_image, data=self.sonar_image, path=rgb_file_path)


    # def _get_anno_semantic_mapping(self):
    #     anno_semantic_mapping = {}
    #     for k, v in self.mapping_dict.items():
    #         is_valid_id = isinstance(v, int)
    #         is_valid_colour = isinstance(v, (list, tuple)) and len(v) == 4 and all(isinstance(e, int) for e in v)
    #         if not is_valid_id and not is_valid_colour:
    #             raise ValueError(
    #                 f"Provided mapping maps to invalid values. All target values must be an integer ID or integer RGBA values"
    #             )
    #         if ":" in k:
    #             anno_semantic_mapping[k] = v
    #         else:
    #             # fallback on `class` semantic type
    #             anno_semantic_mapping[f"class:{k}"] = v
    #     return json.dumps(anno_semantic_mapping)

    
    # def _write_rgb(self, data, sub_dir: str, annotator: str):
    #     # Save the rgb data under the correct path
    #     rgb_dir_name = "image_02" if self._use_kitti_dir_names else "rgb"
    #     rgb_file_path = os.path.join(sub_dir, rgb_dir_name, f"{self._frame_id}.png")
    #     self._backend.schedule(F.write_image, data=data[annotator], path=rgb_file_path)

    # def _write_object_detection(
    #     self,
    #     data,
    #     sub_dir: str,
    #     render_product_annotator: str,
    #     bbox_2d_tight_annotator: str,
    #     bbox_2d_loose_annotator: str,
    # ):
    #     r"""
    #     Saves the labels for the object detection data in Kitti format.

    #     Unsupported fields: alpha, rotation_y, truncated (all set to default values of 0.0)

    #     Notes on occlusion:
    #     # This estimation relies on the ratio between loose (unoccluded) and tight bounding boxes
    #     # and may produce unexpected results in certain cases:
    #     #
    #     #        //           XXXX                 //  XXXX
    #     #  _____//____/_______XXXX          ______//___XXXX______
    #     # )   __          __  XXXX         )   __      XXXX_     \
    #     # |__/  \________/  \_XXXX         |__/  \_____XXXX \____|
    #     # ___\__/________\__/_XXXX__      ____\_ /_____XXXX_/______
    #     # PARTLY OCCLUDED (OK!)           FULLY VISIBLE (INCORRECT)
    #     """
    #     label_set = []

    #     rp_width = data[render_product_annotator]["resolution"][0]
    #     rp_height = data[render_product_annotator]["resolution"][1]

    #     bbox_tight = data[bbox_2d_tight_annotator]["data"]
    #     bbox_loose = data[bbox_2d_loose_annotator]["data"]

    #     bbox_tight_bbox_ids = data[bbox_2d_tight_annotator]["info"]["bboxIds"]
    #     bbox_loose_bbox_ids = data[bbox_2d_loose_annotator]["info"]["bboxIds"]

    #     # For box in tight, find the corresponding index of box in loose
    #     bbox_loose_indices = np.where(np.isin(bbox_loose_bbox_ids, bbox_tight_bbox_ids))[0]
    #     selected_bbox_loose = bbox_loose[bbox_loose_indices]

    #     for box_tight, box_loose in zip(bbox_tight, selected_bbox_loose):

    #         label = []

    #         # Skip boxes shorter than threshold pixels in height
    #         if box_tight["y_max"] - box_tight["y_min"] < self._bbox_height_threshold:
    #             continue

    #         area_tight = (box_tight["x_max"] - box_tight["x_min"]) * (box_tight["y_max"] - box_tight["y_min"])
    #         area_loose = (box_loose["x_max"] - box_loose["x_min"]) * (box_loose["y_max"] - box_loose["y_min"])
    #         area_ratio = area_tight / (area_loose + EPS)

    #         if area_ratio >= self._fully_visible_threshold:
    #             occlusion_estimation = 0
    #         elif area_ratio >= self._partly_occluded_threshold:
    #             occlusion_estimation = 1
    #         else:
    #             occlusion_estimation = 2

    #         # Check if bounding boxes are in the viewport
    #         if (
    #             box_tight["x_min"] < 0
    #             or box_tight["y_min"] < 0
    #             or box_tight["x_max"] > rp_width
    #             or box_tight["y_max"] > rp_height
    #             or box_tight["x_min"] > rp_width
    #             or box_tight["y_min"] > rp_height
    #             or box_tight["y_max"] < 0
    #             or box_tight["x_max"] < 0
    #         ):
    #             continue

    #         semantic_label = data[bbox_2d_tight_annotator]["info"]["idToLabels"].get(box_tight["semanticId"])

    #         if self._omit_semantic_type:
    #             # omit semantic type
    #             semantic_label = semantic_label.get("class", "Unlabelled")

    #         # Adding Kitti Data,  NOTE: Only class and 2d bbox coordinates are filled in
    #         label.append(semantic_label)  # semantic
    #         label.append(f"{0.00:.2f}")  # truncated (not supported)
    #         label.append(occlusion_estimation)  # occluded (estimation)
    #         label.append(f"{0.00:.2f}")  # alpha (not supported)
    #         label.append(box_tight["x_min"])  # x min
    #         label.append(box_tight["y_min"])  # y min
    #         label.append(box_tight["x_max"])  # x max
    #         label.append(box_tight["y_max"])  # y max
    #         for _ in range(7):
    #             label.append(f"{0.00:.2f}")  # dimensions, location, rotation_y, score

    #         label_set.append(label)

    #     det_dir_name = "label_02" if self._use_kitti_dir_names else "object_detection"
    #     kitti_filepath = os.path.join(sub_dir, det_dir_name, f"{self._frame_id}.txt")
    #     buf = io.StringIO()

    #     writer = csv.writer(buf, delimiter=" ")
    #     writer.writerows(label_set)

    #     self._backend.schedule(self._backend.write_blob, data=bytes(buf.getvalue(), "utf-8"), path=kitti_filepath)

    # def _procure_labels_from_json(self, json_path):
    #     with open(json_path, "r") as f:
    #         labels_dict = json.load(f)
    #     return labels_dict

    # def _write_segmentation(self, data, sub_dir: str, sem_annotator: str, inst_annotator: str):
    #     """
    #     Instance segmentation follows the format specified here: https://www.vision.rwth-aachen.de/page/mots
    #     """
    #     sem_rgb_dir_name = "semantic_rgb" if self._use_kitti_dir_names else "semantic_segmentation"
    #     inst_dir_name = "instance" if self._use_kitti_dir_names else "instance_segmentation"
    #     seg_filepath = os.path.join(sub_dir, "semantic", f"{self._frame_id}.png")
    #     seg_col_filepath = os.path.join(sub_dir, sem_rgb_dir_name, f"{self._frame_id}.png")
    #     inst_filepath = os.path.join(sub_dir, inst_dir_name, f"{self._frame_id}.png")
    #     inst_col_filepath = os.path.join(sub_dir, "instance_rgb", f"{self._frame_id}.png")

    #     inst_id_to_labels = data[inst_annotator]["info"]["idToSemantics"]
    #     self._backend.schedule(F.write_image, data=data[sem_annotator]["data"], path=seg_col_filepath)

    #     inst_seg_img = data[inst_annotator]["data"]
    #     height, width = inst_seg_img.shape[:2]

    #     if self.colorize_instance_segmentation:
    #         inst_seg_img_colorized = inst_seg_img.view(np.uint8)
    #         inst_seg_img_colorized = inst_seg_img_colorized.reshape(height, width, -1)
    #         self._backend.schedule(F.write_image, data=inst_seg_img_colorized, path=inst_col_filepath)

    #     # Re-label instances to be sequentially numbered
    #     # The instance segmentation is a 16bit png where the lower 8 bit contain the semantic ID and the higher 8 bits
    #     # contain the instance ID
    #     # Semantic segmentation is saved as a 3 channel image where each channel is the same 8 bit semantic ID
    #     # Instance IDs start from 1
    #     cur_idx = {}
    #     if self.colorize_instance_segmentation:
    #         # convert ids to uint32
    #         inst_id_to_labels = {
    #             (iid[0] | iid[1] << 8 | iid[2] << 16 | iid[3] << 24): v for iid, v in inst_id_to_labels.items()
    #         }

    #     instance_ids = list(inst_id_to_labels.keys())
    #     semantic_classes = list(self.mapping_dict.keys())
    #     inst_seg_uint32 = inst_seg_img.view(np.uint32).squeeze()
    #     inst_seg_img_renumbered = np.zeros((height, width), dtype=np.uint16)
    #     sem_seg_img_renumbered = np.zeros((height, width), dtype=np.uint8)
    #     for i, iid in enumerate(instance_ids):
    #         semantic_class = inst_id_to_labels[iid].get("class", "unlabelled")
    #         is_unlabelled = semantic_class.lower() == "unlabelled"
    #         is_in_mapping = semantic_class in self.mapping_dict
    #         if not is_in_mapping or is_unlabelled:
    #             inst_seg_img_renumbered[inst_seg_uint32 == iid] = 0
    #         else:
    #             cur_semantics = str(inst_id_to_labels[iid])
    #             cur_idx.setdefault(cur_semantics, 0)
    #             cur_idx[cur_semantics] += 1
    #             semantics_renumbered = semantic_classes.index(semantic_class)
    #             inst_seg_img_renumbered[inst_seg_uint32 == iid] = cur_idx[cur_semantics] + semantics_renumbered * 256
    #             sem_seg_img_renumbered[inst_seg_uint32 == iid] = semantics_renumbered

    #     self._backend.schedule(F.write_image, data=inst_seg_img_renumbered, path=inst_filepath)
    #     self._backend.schedule(F.write_image, data=sem_seg_img_renumbered, path=seg_filepath)

    # def _write_distance_to_camera(self, data, sub_dir: str, annotator: str):
    #     distance_to_camera_metres = data[annotator]
    #     distance_to_camera_metres = np.nan_to_num(distance_to_camera_metres, posinf=0.0)
    #     distance_to_camera_uint16 = (distance_to_camera_metres * 256).astype(np.uint16)
    #     file_path = os.path.join(sub_dir, "depth", f"{self._frame_id}.png")
    #     self._backend.schedule(F.write_image, data=distance_to_camera_uint16, path=file_path)

    def write(self, data):
        render_products = [k for k in data.keys() if k.startswith("rp_")]
        if len(render_products) == 1:
            sub_dir = data[render_products[0]]["camera"].split("/")[-1]
            self._render_sonar(data, "pointcloud", "camera_params", "semantic_segmentation")
            self._write_sonar(sub_dir)
            # self._write_rgb(data, sub_dir, "rgb")
            # self._write_segmentation(data, sub_dir, "semantic_segmentation", "instance_segmentation_fast")
            # self._write_object_detection(
            #     data, sub_dir, render_products[0], "bounding_box_2d_tight_fast", "bounding_box_2d_loose_fast"
            # )
            # self._write_distance_to_camera(data, sub_dir, "distance_to_camera")
        else:
            for render_product in render_products:
                render_product_name = render_product[3:]
                sub_dir = os.path.join(render_product_name, data[render_product]["camera"].split("/")[-1])
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

WriterRegistry.register(FLS_KittiWriter)

## This is the path which has the background scene in which objects will be added.
ENV_URL = "/Isaac/Environments/Simple_Warehouse/warehouse.usd"

import carb
import omni
import omni.usd
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import get_current_stage, open_stage
from pxr import Semantics
import omni.replicator.core as rep
from omni.isaac.core.utils.semantics import get_semantics
import numpy as np

# Increase subframes if shadows/ghosting appears of moving objects
# See known issues: https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_replicator.html#known-issues
rep.settings.carb_settings("/omni/replicator/RTSubframes", 4)


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


def update_semantics(stage, keep_semantics=[]):
    """Remove semantics from the stage except for keep_semantic classes"""
    for prim in stage.Traverse():
        if prim.HasAPI(Semantics.SemanticsAPI):
            processed_instances = set()
            for property in prim.GetProperties():
                is_semantic = Semantics.SemanticsAPI.IsSemanticsAPIPath(
                    property.GetPath()
                )
                if is_semantic:
                    instance_name = property.SplitName()[1]
                    if instance_name in processed_instances:
                        # Skip repeated instance, instances are iterated twice due to their two semantic properties (class, data)
                        continue

                    processed_instances.add(instance_name)
                    sem = Semantics.SemanticsAPI.Get(prim, instance_name)
                    type_attr = sem.GetSemanticTypeAttr()
                    data_attr = sem.GetSemanticDataAttr()

                    for semantic_class in keep_semantics:
                        # Check for our data classes needed for the model
                        if data_attr.Get() == semantic_class:
                            continue
                        else:
                            # remove semantics of all other prims
                            prim.RemoveProperty(type_attr.GetName())
                            prim.RemoveProperty(data_attr.GetName())
                            prim.RemoveAPI(Semantics.SemanticsAPI, instance_name)


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


# This will handle replicator
def run_orchestrator():

    rep.orchestrator.run()

    # Wait until started
    while not rep.orchestrator.get_is_started():
        simulation_app.update()

    # Wait until stopped
    while rep.orchestrator.get_is_started():
        simulation_app.update()

    rep.BackendDispatch.wait_until_done()
    rep.orchestrator.stop()


def main():
    # Open the environment in a new stage
    print(f"Loading Stage {ENV_URL}")
    open_stage(prefix_with_isaac_asset_server(ENV_URL))
    stage = get_current_stage()

    # Run some app updates to make sure things are properly loaded
    for i in range(100):
        if i % 10 == 0:
            print(f"App uppdate {i}..")
        simulation_app.update()

    textures = full_textures_list()
    rep_palletjack_group = add_palletjacks()
    rep_distractor_group = add_distractors(distractor_type=args.distractors)

    # We only need labels for the palletjack objects
    update_semantics(stage=stage, keep_semantics=["palletjack"])
    
    # Load sonar params for adjust camera settings
    sonar_param = {"max_range": 3, 
                    "min_range": 0.2,
                    "hori_fov": 130, # Notice: on camera end, hori_fov and vert_fov is required to 
                    "vert_fov": 20,  # compute camera AR and vert_res given arbitrary hori_res
                    "range_res": 0.005, 
                    "angular_res": 0.25,
                    "normalizing_method": "range",
                    "query_prop": "reflectivity", # bit wanky, leave this for now
                    "attenuation": 0.1,
                    "gau_noise_param": 0.2,
                    "ray_noise_param": 0.05,
                    "intensity_offset": 0.0,
                    "intensity_gain": 1.0,
                    "central_peak": 2,
                    "central_std": 0.001}
    
    hori_res = 5000
    AR = sonar_param['hori_fov'] / sonar_param['vert_fov']
    vert_res = int(hori_res / AR)
    # By doing this, I am assuming the vertical beam separation
    # is the same as the beam horizontal separation. 
    # This is bacause replicator raytracing is specified as resolutions
    # while non-squre pixel is not supported in Isaac sim. See details below.
    

    # Assume the default focal length to compute the desired horizontal aperture
    # The reason why we are doing this is because Isaac sim will fix vertical aperture
    # given aspect ratio for mandating square pixles
    # https://forums.developer.nvidia.com/t/how-to-modify-the-cameras-field-of-view/278427/5

    focal_length = 24.0 # This is default when create a camera in Isaac Sim
    horizontal_aper = 2 * focal_length * np.tan(np.deg2rad(sonar_param['hori_fov']) / 2)
    # Create camera with Replicator API for gathering data
    cam = rep.create.camera(clipping_range=(sonar_param['min_range'], sonar_param['max_range']),
                            horizontal_aperture=horizontal_aper)

    # trigger replicator pipeline
    with rep.trigger.on_frame(max_execs=CONFIG["num_frames"]):

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

    # Set up the writer
    writer = rep.WriterRegistry.get("FLS_KittiWriter")

    # output directory of writer
    output_directory = args.data_dir
    print("Outputting data to ", output_directory)

    # use writer for bounding boxes, rgb and segmentation
    writer.initialize(
        output_dir=output_directory,
        omit_semantic_type=True,
    )

    # attach camera render products to writer so that data is outputted
    RESOLUTION = (hori_res, vert_res)
    render_product = rep.create.render_product(cam, RESOLUTION)
    writer.attach(render_product)

    # run rep pipeline
    run_orchestrator()
    simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        carb.log_error(f"Exception: {e}")
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()
