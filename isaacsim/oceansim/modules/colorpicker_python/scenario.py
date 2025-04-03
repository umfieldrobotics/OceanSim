# Omniverse import
import numpy as np
import omni.replicator.core as rep
import omni.ui as ui
import warp as wp
from omni.kit.viewport.utility import get_active_viewport

# Custom import
from isaacsim.oceansim.utils.UWrenderer_utils import UW_render

class Colorpicker_Scenario():
    def __init__(self):

        self.raw_rgba = None
        self.depth_image = None
        self._running_scenario = False
        self._time = 0.0
        self._id = 0
        self._device = wp.get_preferred_device()

        self._viewport = None
        self._viewport_rgba_annot = None
        self._viewport_depth_annot = None
    


    def setup_scenario(self):


        self._running_scenario = True
        
        self._viewport = get_active_viewport()
        self._viewport_rgba_annot = rep.AnnotatorRegistry.get_annotator(name='LdrColor', device=str(self._device))
        self._viewport_depth_annot = rep.AnnotatorRegistry.get_annotator(name="distance_to_camera", device=str(self._device))
        self._viewport_rgba_annot.attach(self._viewport.render_product_path)
        self._viewport_depth_annot.attach(self._viewport.render_product_path)
        
        self.make_window()
      

    def teardown_scenario(self):
        self._running_scenario = False
        self._time = 0.0
        self._id = 0

        if self._viewport is not None:
            self._viewport_rgba_annot.detach(self._viewport.render_product_path)
            self._viewport_depth_annot.detach(self._viewport.render_product_path)
            rep.AnnotatorCache.clear(self._viewport_rgba_annot)
            rep.AnnotatorCache.clear(self._viewport_depth_annot)
            self.ui_destroy()
        
        self._viewport = None
        self._viewport_rgba_annot = None
        self._viewport_depth_annot = None


    def update_scenario(self, step: float, render_param: np.ndarray):

        
        if not self._running_scenario:
            return
        self._time += step
        if self._viewport_rgba_annot.get_data().size == 0:
            return
        self.raw_rgba = self._viewport_rgba_annot.get_data()
        self.depth_image = self._viewport_depth_annot.get_data()        
        
        self.update_render(render_param)

        self._id += 1

       
            
    
    def update_render(self, render_param: np.ndarray):
        if self.raw_rgba is not None:
            if self.raw_rgba.size !=0:
                backscatter_value = wp.vec3f(*render_param[0:3])
                atten_coeff = wp.vec3f(*render_param[6:9])
                backscatter_coeff = wp.vec3f(*render_param[3:6])
                self.uw_image = wp.zeros_like(self.raw_rgba)
                wp.launch(
                    dim=(self.raw_rgba.shape[0], self.raw_rgba.shape[1]),
                    kernel=UW_render,
                    inputs=[
                        self.raw_rgba,
                        self.depth_image,
                        backscatter_value,
                        atten_coeff,
                        backscatter_coeff
                    ],
                    outputs=[
                        self.uw_image
                    ]
                )  
                
                self.image_provider.set_bytes_data_from_gpu(self.uw_image.ptr, [self.uw_image.shape[1], self.uw_image.shape[0]])

    def make_window(self):

        self.wrapped_ui_elements = []
        window = ui.Window("Render Result", width=1920, height=1080 + 40, visible=True)
        self.image_provider = ui.ByteImageProvider()
        with window.frame:
            with ui.ZStack(height=1080):
                ui.Rectangle(style={"background_color": 0xFF000000})
                ui.Label('Run the scenario for image to be received',
                            style={'font_size': 55,'alignment': ui.Alignment.CENTER},
                            word_wrap=True)
                render_result = ui.ImageWithProvider(self.image_provider, width=1920, height=1080,
                                        style={'fill_policy': ui.FillPolicy.PRESERVE_ASPECT_FIT,
                                    'alignment' :ui.Alignment.CENTER})
   
        self.wrapped_ui_elements.append(render_result)
        self.wrapped_ui_elements.append(window)
        self.wrapped_ui_elements.append(self.image_provider)
    
    def ui_destroy(self):
        for elem in self.wrapped_ui_elements:
            elem.destroy()
    
   
   
   
   
   
   
   
   
    # from omni.kit.viewport.utility import get_active_viewport
    # self.viewport_api = get_active_viewport()
    # capture = self.viewport_api.schedule_capture(ByteCapture(self.on_capture_completed, aov_name='LdrColor'))
    # def on_capture_completed(self, buffer, buffer_size, width, height, format):
    #     '''
    #     Example
    #     buffer: <capsule object NULL at 0x70805cd0c660>
    #     buffer_size: 3686400
    #     width: 1280
    #     height: 720
    #     format: TextureFormat.RGBA8_UNORM
    #     '''
    #     self.image_provider.set_raw_bytes_data(raw_bytes=buffer,
    #                                             sizes=[width, height],
    #                                             format=format)
