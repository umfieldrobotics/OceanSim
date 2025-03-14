# Omniverse Import
import omni.replicator.core as rep
from omni.replicator.core.scripts.functional import write_image
import omni.ui as ui

# Isaac sim import
from isaacsim.sensors.camera import Camera
import numpy as np
import warp as wp
import yaml
import carb

# Custom import
from isaacsim.OceanSim.utils.UWrenderer_utils import UW_render


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
        self._id = 0
        self._viewport = viewport
        self._device = wp.get_preferred_device()
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

        self._rgba_annot.attach(self._render_product_path)
        self._depth_annot.attach(self._render_product_path)

        if self._viewport:
            self.make_viewport()

        if writing_dir is not None:
            self._writing = True
            self._writing_backend = rep.BackendDispatch({"paths": {"out_dir": writing_dir}})
        
        print(f'[{self._name}] Initialized successfully. Data writing: {self._writing}')
    
    def render(self):
        raw_rgba = self._rgba_annot.get_data()
        depth = self._depth_annot.get_data()
        if raw_rgba.size !=0:
            uw_image = wp.zeros_like(raw_rgba)
            wp.launch(
                dim=np.flip(self.get_resolution()),
                kernel=UW_render,
                inputs=[
                    raw_rgba,
                    depth,
                    self._backscatter_value,
                    self._atten_coeff,
                    self._backscatter_coeff
                ],
                outputs=[
                    uw_image
                ]
            )  
            
            if self._viewport:
                self._provider.set_bytes_data_from_gpu(uw_image.ptr, self.get_resolution())
            if self._writing:
                self._writing_backend.schedule(write_image, path=f'UW_image_{self._id}.png', data=uw_image)
                print(f'[{self._name}] [{self._id}] Rendered image saved to {self._writing_backend.output_dir}')

            self._id += 1

    def make_viewport(self):
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
        self._rgba_annot.detach(self._render_product_path)
        self._depth_annot.detach(self._render_product_path)

        rep.AnnotatorCache.clear(self._rgba_annot)
        rep.AnnotatorCache.clear(self._depth_annot)

        if self._viewport:
            self.ui_destroy()
            
        print(f'[{self._name}] Annotator detached. AnnotatorCache cleaned.')
    
    
    def ui_destroy(self):
        for elem in self.wrapped_ui_elements:
            elem.destroy()

        
       