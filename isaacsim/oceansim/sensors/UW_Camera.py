# Omniverse Import
import omni.replicator.core as rep
from omni.replicator.core.scripts.functional import write_image
import omni.ui as ui
import json
# Isaac sim import
from isaacsim.sensors.camera import Camera
import numpy as np
import warp as wp
import yaml
import carb

# Custom import
from isaacsim.oceansim.utils.UWrenderer_utils import *


class UW_Camera(Camera):

    def __init__(self, 
                 prim_path, 
                 name = "UW_Camera", 
                 frequency = None, 
                 dt = None, 
                 resolution = None, 
                 position = None, 
                 orientation = None, 
                 translation = None, 
                 render_product_path = None):
        
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
        super().__init__(prim_path, name, frequency, dt, resolution, position, orientation, translation, render_product_path)

    def initialize(self, 
                   UW_param: np.ndarray = np.array([0.0, 0.31, 0.24, 0.05, 0.05, 0.2, 0.05, 0.05, 0.05 ]),
                   viewport: bool = True,
                   writing_dir: str = None,
                   UW_yaml_path: str = None,
                   physics_sim_view=None):
        
        """Configure underwater rendering properties and initialize pipelines.
    
        Args:
            UW_param (np.ndarray, optional): Underwater parameters array:
                [0:3] - Backscatter value (RGB)
                [3:6] - Attenuation coefficients (RGB)
                [6:9] - Backscatter coefficients (RGB)
                Defaults to typical coastal water values.
            viewport (bool, optional): Enable viewport visualization. Defaults to True.
            writing_dir (str, optional): Directory to save rendered images. Defaults to None.
            UW_yaml_path (str, optional): Path to YAML file with water properties. Defaults to None.
            physics_sim_view (_type_, optional): _description_. Defaults to None.            
    
        """
        self._id = 0
        self._time = 0.0
        self._timeSpeed = 2.0

        self._viewport = viewport
        self._device = wp.get_preferred_device()
        self._uw_image = wp.empty(shape=(self._resolution[1], self._resolution[0], 4), dtype=wp.uint8)
        self._caustics = wp.empty(shape=(self._resolution[1], self._resolution[0], 4), dtype=wp.uint8)
        super().initialize(physics_sim_view)

        if UW_yaml_path is not None:
            with open(UW_yaml_path, 'r') as file:
                try:
                    # Load the YAML content
                    yaml_content = yaml.safe_load(file)
                    self._backscatter_value = wp.vec3f(*yaml_content['backscatter_value'])
                    self._atten_coeff = wp.vec3f(*yaml_content['atten_coeff'])
                    self._backscatter_coeff = wp.vec3f(*yaml_content['backscatter_coeff'])
                    print(f"[{self._name}] On {str(self._device)}. Using loaded render parameters:")
                    print(f"[{self._name}] Render parameters: {yaml_content}")
                except yaml.YAMLError as exc:
                    carb.log_error(f"[{self._name}] Error reading YAML file: {exc}")
        else:
            self._backscatter_value = wp.vec3f(*UW_param[0:3])
            self._atten_coeff = wp.vec3f(*UW_param[6:9])
            self._backscatter_coeff = wp.vec3f(*UW_param[3:6])
            print(f'[{self._name}] On {str(self._device)}. Using default render parameters.')

        
        self._rgba_annot = rep.AnnotatorRegistry.get_annotator('LdrColor', device=str(self._device))
        self._depth_annot = rep.AnnotatorRegistry.get_annotator('distance_to_camera', device=str(self._device))
        self._normalAnnot = rep.AnnotatorRegistry.get_annotator('normals', device=str(self._device))
        self._cameraParamAnnot = rep.AnnotatorRegistry.get_annotator("CameraParams")
        
        self._normalAnnot.attach(self._render_product_path)
        self._rgba_annot.attach(self._render_product_path)
        self._depth_annot.attach(self._render_product_path)
        self._cameraParamAnnot.attach(self._render_product_path)

        if self._viewport:
            self.make_viewport()

        if writing_dir is not None:
            self._writing = True
            self._writing_backend = rep.BackendDispatch({"paths": {"out_dir": writing_dir}})
        
        print(f'[{self._name}] Initialized successfully. Data writing: {self._writing}')
        self._world_points = wp.empty(shape=(self._resolution[1], self._resolution[0], 3), dtype=wp.float32)
    
    def render(self):
        """Process and display a single frame with underwater effects.
        Note:
            - Updates viewport display if enabled
            - Saves image to disk if writing_dir was specified
        """
        if self._rgba_annot.get_data().size !=0:
            
            # Extract camera position from view transform matrix
            # For view transform: camera_pos = -view_matrix[3,0:3] (last row, first 3 elements)

            # wp.launch(
            #     kernel=water_caustics,
            #     dim=self.get_resolution(),  # (x, y)
            #     inputs=[self._caustics, self._resolution[0], self._resolution[1], self._time, self._timeSpeed],
            # )


            # # Launch depth to world kernel once (this doesn't change)
            # wp.launch(
            #     kernel=depth_to_world_pos,
            #     dim=self.get_resolution(),  # Launch dimensions (width, height)
            #     inputs=[
            #         self._depth_annot.get_data(),
            #         wp.mat44(self._cameraParamAnnot.get_data()["cameraProjection"].reshape(4, 4)),
            #         wp.mat44(self._cameraParamAnnot.get_data()["cameraViewTransform"].reshape(4, 4)),
            #         self.get_resolution()[0],
            #         self.get_resolution()[1]
            #     ],
            #     outputs=[self._world_points],
            #     device=str(self._device)
            # )
            # wp.launch(
            #     kernel=blend_caustics,
            #     dim=self.get_resolution(),
            #     inputs=[
            #         self._rgba_annot.get_data(),
            #         self._world_points,
            #         self._normalAnnot.get_data(),
            #         self._caustics,
            #         wp.vec3f(0.0, 0.0, 1.0),
            #         1.0,       # blend_weight
            #         1.0,       # uv_scale_x (horizontal scaling)
            #         1.0,       # uv_scale_y (vertical scaling)
            #         0.0,       # depth_min
            #         10000.0,   # depth_max
            #         self._resolution[0],         # tex_w
            #         self._resolution[1],         # tex_h
            #     ],
            #     outputs=[self._uw_image],
            #     device=str(self._device)
            # )
            wp.launch(
                dim=(self._resolution[1], self._resolution[0]),  # Note the (height, width) order for 2D launch
                kernel=UW_render_2,
                inputs=[
                    self._rgba_annot.get_data(),
                    self._depth_annot.get_data(),
                    1.0,
                    self._backscatter_value,
                    self._atten_coeff,
                    self._backscatter_coeff
                ],
                outputs=[
                    self._uw_image
                ]
            )  
            
            if self._viewport:
                self._provider.set_bytes_data_from_gpu(self._uw_image.ptr, self.get_resolution())
            if self._writing:
                self._writing_backend.schedule(write_image, path=f'UW_image_{self._id}.png', data=self._uw_image)
                print(f'[{self._name}] [{self._id}] Rendered image saved to {self._writing_backend.output_dir}')

            self._id += 1
            self._time += self.get_dt()

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
                ui.Label('Run the scenario for image to be received',
                         style={'font_size': 55,'alignment': ui.Alignment.CENTER},
                         word_wrap=True)
                image_provider = ui.ImageWithProvider(self._provider, width=1280, height=720,
                                     style={'fill_policy': ui.FillPolicy.PRESERVE_ASPECT_FIT,
                                    'alignment' :ui.Alignment.CENTER})
        
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
            
        print(f'[{self._name}] Annotator detached. AnnotatorCache cleaned.')
    
    
    def ui_destroy(self):
        """Explicitly destroy viewport UI elements.
    
        Note:
            - Called automatically by close()
            - Only needed if manually managing UI lifecycle
        """
        for elem in self.wrapped_ui_elements:
            elem.destroy()

        
       