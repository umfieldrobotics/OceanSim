# Omniverse Import
import copy

import carb
import numpy as np

# Import OmniGraph Controller
import omni.graph.core as og
import omni.replicator.core as rep
import omni.syntheticdata
import omni.ui as ui
import warp as wp
import yaml

# Custom import
from isaacsim.oceansim.utils.UWrenderer_utils import (
    UW_depth_turbidity_attenuator,
    UW_render,
)

# Isaac sim import
# from isaacsim.sensors.camera import Camera
from isaacsim.sensors.camera import Camera
from omni.replicator.core.scripts.annotators import Augmentation
from omni.replicator.core.scripts.functional import write_image

# ROS2 bridge extension
# from isaacsim.core.utils.extensions import enable_extension
# enable_extension("isaacsim.ros2.bridge")
# from isaacsim.oceansim.sensors.ros2_helpers # move to utils
# import ros2_helpers


# def uw_augmentation( raw_image, depth_image, backscatter_value, atten_coeff, backscatter_coeff):
#     import warp as wp
#     uw_image_ = wp.zeros_like(raw_image)
#     bs_vec = wp.vec3(backscatter_value[0], backscatter_value[1], backscatter_value[2])
#     at_vec = wp.vec3(atten_coeff[0], atten_coeff[1], atten_coeff[2])
#     bsc_vec = wp.vec3(backscatter_coeff[0], backscatter_coeff[1], backscatter_coeff[2])
#     wp.launch(
#         dim=np.flip(raw_image.shape[:2]),
#         kernel=UW_render,
#         inputs=[raw_image, depth_image, bs_vec, at_vec, bsc_vec],
#         outputs=[uw_image_]
#     )
#    return uw_image_
#
# TODO: grab this as a param
MAX_RANGE = 20.0
USE_NAIVE_RANGE = True


class UW_Camera_ROS(Camera):
    def __init__(
        self,
        prim_path,
        name="UW_Camera_ROS",
        frequency=None,
        dt=None,
        resolution=None,
        position=None,
        orientation=None,
        translation=None,
        render_product_path=None,
    ):
        """Initialize an underwater camera sensor.

        Args:
            prim_path (str): prim path of the Camera Prim to encapsulate or create.
            name (str, optional): shortname to be used as a key by Scene class.
                                    Note: needs to be unique if the object is added to the Scene.
                                    Defaults to "UW_Camera".
            frequency (Optional[int], optional): Frequency of the sensor (i.e: how often is the data frame updated).
                                                Defaults to None.
            dt (Optional[str], optional): dt of the sensor (i.e: period at which a the data frame updated). Defaults to None.
            resolution (Optional[Tuple[int, int]], optional): resolution of the camera (width, height). Defaults to None.
            position (Optional[Sequence[float]], optional): position in the world frame of the prim. shape is (3, ).
                                                        Defaults to None, which means left unchanged.
            translation (Optional[Sequence[float]], optional): translation in the local frame of the prim
                                                            (with respect to its parent prim). shape is (3, ).
                                                            Defaults to None, which means left unchanged.
            orientation (Optional[Sequence[float]], optional): quaternion orientation in the world/ local frame of the prim
                                                            (depends if translation or position is specified).
                                                            quaternion is scalar-first (w, x, y, z). shape is (4, ).
                                                            Defaults to None, which means left unchanged.
            render_product_path (str): path to an existing render product, will be used instead of creating a new render product
                                    the resolution and camera attached to this render product will be set based on the input arguments.
                                    Note: Using same render product path on two Camera objects with different camera prims, resolutions is not supported
                                    Defaults to None
        """
        self._name = name
        self._prim_path = prim_path
        self._res = resolution
        self._writing = False
        self._uw_frame = None
        self._degraded_depth_frame = None
        self._og_node = None
        self._depth_og_node = None
        self._pointcloud_og_node = None
        self._camera_k = None

        super().__init__(
            prim_path,
            name,
            frequency,
            dt,
            resolution,
            position,
            orientation,
            translation,
            render_product_path,
        )

    def initialize(
        self,
        UW_param: np.ndarray = np.array(
            [0.0, 0.31, 0.24, 0.05, 0.05, 0.2, 0.05, 0.05, 0.05]
        ),
        depth_noise_sigma: float = 0.01,
        viewport: bool = True,
        writing_dir: str = None,
        UW_yaml_path: str = None,
        physics_sim_view=None,
        og_node=None,
        depth_og_node=None,
        pointcloud_og_node=None,
    ):
        """Configure underwater rendering properties and initialize pipelines.

        Args:
            UW_param (np.ndarray, optional): Underwater parameters array:
                [0:3] - Backscatter value (RGB)
                [3:6] - Attenuation coefficients (RGB)
                [6:9] - Backscatter coefficients (RGB)
                Defaults to typical coastal water values.
            depth_noise_sigma (float, optional): Standard deviation of Gaussian
                depth noise applied after turbidity attenuation. Defaults to 0.01.
            viewport (bool, optional): Enable viewport visualization. Defaults to True.
            writing_dir (str, optional): Directory to save rendered images. Defaults to None.
            UW_yaml_path (str, optional): Path to YAML file with water properties. Defaults to None.
            physics_sim_view (_type_, optional): _description_. Defaults to None.

        """
        self._og_node = og_node
        self._depth_og_node = depth_og_node
        self._pointcloud_og_node = pointcloud_og_node
        self._id = 0
        self._viewport = viewport
        self._device = wp.get_preferred_device()
        self._depth_noise_sigma = wp.float32(depth_noise_sigma)
        super().initialize(physics_sim_view)

        if UW_yaml_path is not None:
            with open(UW_yaml_path, "r") as file:
                try:
                    # Load the YAML content
                    yaml_content = yaml.safe_load(file)
                    self._backscatter_value = wp.vec3f(
                        *yaml_content["backscatter_value"]
                    )
                    self._atten_coeff = wp.vec3f(*yaml_content["atten_coeff"])
                    self._backscatter_coeff = wp.vec3f(
                        *yaml_content["backscatter_coeff"]
                    )
                    print(
                        f"[{self._name}] On {str(self._device)}. Using loaded render parameters:"
                    )
                    print(f"[{self._name}] Render parameters: {yaml_content}")
                except yaml.YAMLError as exc:
                    carb.log_error(f"[{self._name}] Error reading YAML file: {exc}")
        else:
            self._backscatter_value = wp.vec3f(*UW_param[0:3])
            self._atten_coeff = wp.vec3f(*UW_param[6:9])
            self._backscatter_coeff = wp.vec3f(*UW_param[3:6])
            print(
                f"[{self._name}] On {str(self._device)}. Using default render parameters."
            )

        self._rgba_annot = rep.AnnotatorRegistry.get_annotator(
            "LdrColor", device=str(self._device)
        )
        self._depth_annot = rep.AnnotatorRegistry.get_annotator(
            "distance_to_camera", device=str(self._device)
        )

        self._depth_annot.attach(self._render_product_path)
        self._rgba_annot.attach(self._render_product_path)
        # rep.AnnotatorRegistry.register_augmentation("UnderwaterAug", uw_augmentation)
        #     self._rgba_annot.augment(
        # uw_augmentation,
        #     raw_image=self._rgba_annot,
        #     depth_image=self._depth_annot,
        #     backscatter_value=list(self._backscatter_value),
        #     atten_coeff=list(self._atten_coeff),
        #     backscatter_coeff=list(self._backscatter_coeff)
        # )

        if self._viewport:
            self.make_viewport()

        if writing_dir is not None:
            self._writing = True
            self._writing_backend = rep.BackendDispatch(
                {"paths": {"out_dir": writing_dir}}
            )

        if self._depth_og_node:
            frame_id = self.prim_path.split("/")[-1]
            og.Controller.attribute(
                self._depth_og_node.get_attribute("inputs:frameId")
            ).set(frame_id)
            og.Controller.attribute(
                self._depth_og_node.get_attribute("inputs:topicName")
            ).set(f"{self.name}_depth")
            og.Controller.attribute(
                self._depth_og_node.get_attribute("inputs:encoding")
            ).set("32FC1")

        if self._pointcloud_og_node:
            from isaacsim.ros2.bridge import read_camera_info

            frame_id = self.prim_path.split("/")[-1]
            camera_info, _ = read_camera_info(
                render_product_path=self._render_product_path
            )
            self._camera_k = np.asarray(camera_info.k, dtype=np.float32).reshape(3, 3)
            og.Controller.attribute(
                self._pointcloud_og_node.get_attribute("inputs:frameId")
            ).set(frame_id)
            og.Controller.attribute(
                self._pointcloud_og_node.get_attribute("inputs:topicName")
            ).set(f"{self.name}_pointcloud")

        print(f"[{self._name}] Initialized successfully. Data writing: {self._writing}")

    def render(self):
        """Process and display a single frame with underwater effects.

        Note:
            - Updates viewport display if enabled
            - Saves image to disk if writing_dir was specified
        """
        raw_rgba = self._rgba_annot.get_data()
        depth = self._depth_annot.get_data()
        if raw_rgba.size != 0:
            uw_image = wp.zeros_like(raw_rgba)

            # RGB
            wp.launch(
                dim=np.flip(self.get_resolution()),
                kernel=UW_render,
                inputs=[
                    raw_rgba,
                    depth,
                    self._backscatter_value,
                    self._atten_coeff,
                    self._backscatter_coeff,
                ],
                outputs=[uw_image],
            )
            degraded_depth = wp.zeros_like(depth)

            # Depth
            wp.launch(
                dim=np.flip(self.get_resolution()),
                kernel=UW_depth_turbidity_attenuator,
                inputs=[
                    depth,
                    degraded_depth,
                    MAX_RANGE,
                    self._backscatter_value,
                    self._atten_coeff,
                    self._backscatter_coeff,
                    self._depth_noise_sigma,
                    int(self._id),
                ],
                outputs=[],
            )

            #     #for raw ros2 msg ocnstruction
            self._uw_frame = uw_image.numpy()
            self._degraded_depth_frame = np.ascontiguousarray(
                degraded_depth.numpy(), dtype=np.float32
            )
            #    wp.copy(raw_rgba, uw_image)

            # ros pub:
            sim_time = omni.timeline.get_timeline_interface().get_current_time()

            if self._og_node:
                # Set the data on the node
                # og.Controller.attribute(self._og_node.get_attribute("inputs:width")).set(self._uw_frame.shape[0])
                # og.Controller.attribute(self._og_node.get_attribute("inputs:height")).set(self._uw_frame.shape[1])
                og.Controller.attribute(
                    self._og_node.get_attribute("inputs:width")
                ).set(self._res[0])
                og.Controller.attribute(
                    self._og_node.get_attribute("inputs:height")
                ).set(self._res[1])
                og.Controller.attribute(
                    self._og_node.get_attribute("inputs:timeStamp")
                ).set(sim_time)
                og.Controller.attribute(self._og_node.get_attribute("inputs:data")).set(
                    self._uw_frame
                )
            if self._depth_og_node:
                depth_bytes = self._degraded_depth_frame.view(np.uint8).reshape(-1)
                og.Controller.attribute(
                    self._depth_og_node.get_attribute("inputs:timeStamp")
                ).set(sim_time)
                og.Controller.attribute(
                    self._depth_og_node.get_attribute("inputs:width")
                ).set(self._degraded_depth_frame.shape[1])
                og.Controller.attribute(
                    self._depth_og_node.get_attribute("inputs:height")
                ).set(self._degraded_depth_frame.shape[0])
                og.Controller.attribute(
                    self._depth_og_node.get_attribute("inputs:bufferSize")
                ).set(self._degraded_depth_frame.nbytes)
                og.Controller.attribute(
                    self._depth_og_node.get_attribute("inputs:data")
                ).set(depth_bytes)
            if self._pointcloud_og_node and self._camera_k is not None:
                stride = 4
                fx = self._camera_k[0, 0]
                fy = self._camera_k[1, 1]
                cx = self._camera_k[0, 2]
                cy = self._camera_k[1, 2]
                height, width = self._degraded_depth_frame.shape
                sampled_depth = self._degraded_depth_frame[::stride, ::stride]
                u, v = np.meshgrid(
                    np.arange(0, width, stride, dtype=np.float32),
                    np.arange(0, height, stride, dtype=np.float32),
                )
                x = (u - cx) / fx
                y = (v - cy) / fy
                norm = np.sqrt(x * x + y * y + 1.0)
                valid = np.isfinite(sampled_depth) & (sampled_depth > 0.0)
                if np.any(valid):
                    scale = sampled_depth[valid] / norm[valid]
                    pointcloud = np.ascontiguousarray(
                        np.column_stack(
                            (
                                x[valid] * scale,
                                y[valid] * scale,
                                scale,
                            )
                        ).astype(np.float32)
                    )
                else:
                    pointcloud = np.empty((0, 3), dtype=np.float32)
                og.Controller.attribute(
                    self._pointcloud_og_node.get_attribute("inputs:timeStamp")
                ).set(sim_time)
                og.Controller.attribute(
                    self._pointcloud_og_node.get_attribute("inputs:bufferSize")
                ).set(pointcloud.nbytes)
                og.Controller.attribute(
                    self._pointcloud_og_node.get_attribute("inputs:data")
                ).set(pointcloud)
            if self._viewport:
                self._provider.set_bytes_data_from_gpu(
                    uw_image.ptr, self.get_resolution()
                )
            if self._writing:
                self._writing_backend.schedule(
                    write_image, path=f"UW_image_{self._id}.png", data=uw_image
                )
                print(
                    f"[{self._name}] [{self._id}] Rendered image saved to {self._writing_backend.output_dir}"
                )

            self._id += 1

    def make_viewport(self):
        """Create a viewport window for real-time visualization.

        Note:
            - Window size fixed at 1280x760 pixels
        """

        self.wrapped_ui_elements = []
        self.window = ui.Window(self._name, width=1280, height=720 + 40, visible=True)
        self._provider = ui.ByteImageProvider()
        with self.window.frame:
            with ui.ZStack(height=720):
                ui.Rectangle(style={"background_color": 0xFF000000})
                ui.Label(
                    "Run the scenario for image to be received",
                    style={"font_size": 55, "alignment": ui.Alignment.CENTER},
                    word_wrap=True,
                )
                image_provider = ui.ImageWithProvider(
                    self._provider,
                    width=1280,
                    height=720,
                    style={
                        "fill_policy": ui.FillPolicy.PRESERVE_ASPECT_FIT,
                        "alignment": ui.Alignment.CENTER,
                    },
                )

        self.wrapped_ui_elements.append(image_provider)
        self.wrapped_ui_elements.append(self._provider)
        self.wrapped_ui_elements.append(self.window)

    # Detach the annotator from render product and clear the data cache
    def close(self):
        """Clean up resources by detaching annotators and clearing caches.

        Note:
            - Required for proper shutdown when done using the sensor
            - Also closes viewport window if one was created
        """
        self._rgba_annot.detach(self._render_product_path)
        self._depth_annot.detach(self._render_product_path)

        rep.AnnotatorCache.clear(self._rgba_annot)
        rep.AnnotatorCache.clear(self._depth_annot)

        if self._viewport:
            self.ui_destroy()

        print(f"[{self._name}] Annotator detached. AnnotatorCache cleaned.")

    def ui_destroy(self):
        """Explicitly destroy viewport UI elements.

        Note:
            - Called automatically by close()
            - Only needed if manually managing UI lifecycle
        """
        for elem in self.wrapped_ui_elements:
            elem.destroy()


# ROS2 Integration
# approx_freq = 30
# ros2_helpers.publish_camera_info(camera, approx_freq)
# ros2_helpers.publish_rgb(camera, approx_freq)
# ros2_helpers.publish_depth(camera, approx_freq)
# ros2_helpers.publish_pointcloud_from_depth(camera, approx_freq)
# ros2_helpers.publish_camera_tf(camera)
