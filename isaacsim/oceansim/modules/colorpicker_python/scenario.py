# Omniverse import
import numpy as np
import omni.replicator.core as rep
import omni.ui as ui
import warp as wp
from omni.kit.viewport.utility import get_active_viewport

# Custom import
from isaacsim.oceansim.utils.UWrenderer_utils import *
from isaacsim.oceansim.watersurface import WaterSurface
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
        self._viewport_normalAnnot = None
        self._viewport_cameraParamAnnot = None
    
        self._water = None

    def setup_scenario(self, water:WaterSurface):


        self._running_scenario = True
        self._water = water
        self._viewport = get_active_viewport()
        (self.width, self.height) = self._viewport.get_texture_resolution()
        self._viewport_rgba_annot = rep.AnnotatorRegistry.get_annotator(name='LdrColor', device=str(self._device))
        self._viewport_depth_annot = rep.AnnotatorRegistry.get_annotator(name="distance_to_camera", device=str(self._device))
        self._viewport_normalAnnot = rep.AnnotatorRegistry.get_annotator('normals', device=str(self._device))
        self._viewport_cameraParamAnnot = rep.AnnotatorRegistry.get_annotator("CameraParams")
        
        
        
        self._viewport_cameraParamAnnot.attach(self._viewport.render_product_path)
        self._viewport_normalAnnot.attach(self._viewport.render_product_path)
        self._viewport_rgba_annot.attach(self._viewport.render_product_path)
        self._viewport_depth_annot.attach(self._viewport.render_product_path)
        self.make_window()
        self.caustics = wp.zeros((self.height, self.width, 4), dtype=wp.uint8)
        self.world_points = wp.zeros((self.height, self.width, 3), dtype=wp.float32)
        self.uw_image = wp.zeros((self.height, self.width, 4), dtype=wp.uint8)

    def teardown_scenario(self):
        self._running_scenario = False
        self._time = 0.0
        self._id = 0

        if self._viewport is not None:
            self._viewport_rgba_annot.detach(self._viewport.render_product_path)
            self._viewport_depth_annot.detach(self._viewport.render_product_path)
            self._viewport_normalAnnot.detach(self._viewport.render_product_path)
            self._viewport_cameraParamAnnot.detach(self._viewport.render_product_path)
            rep.AnnotatorCache.clear(self._viewport_rgba_annot)
            rep.AnnotatorCache.clear(self._viewport_depth_annot)
            rep.AnnotatorCache.clear(self._viewport_normalAnnot)
            rep.AnnotatorCache.clear(self._viewport_cameraParamAnnot)
            self.ui_destroy()
        
        if self._water is not None:
            self._water = None
        
        self._viewport = None
        self._viewport_rgba_annot = None
        self._viewport_depth_annot = None
        self._viewport_normalAnnot = None
        self._viewport_cameraParamAnnot = None


    def update_scenario(self, 
                        step: float, 
                        render_param: np.ndarray, 
                        water_surface_param: np.ndarray,
                        caustics_param: np.ndarray):

        
        if not self._running_scenario:
            return
        self._time += step
        if self._viewport_rgba_annot.get_data().size == 0:
            return
        self.raw_rgba = self._viewport_rgba_annot.get_data()
        self.depth_image = self._viewport_depth_annot.get_data()        
        self.normal_image = self._viewport_normalAnnot.get_data()
        self.camera_param = self._viewport_cameraParamAnnot.get_data()

        self.update_render(render_param, caustics_param)
        
        if self._water is not None:
            self._water.deform(time = self._time,
                               amplitude=water_surface_param[0],
                               clipmapCellSize=water_surface_param[1],
                               direction=water_surface_param[2],
                               directionality=water_surface_param[3],
                               scale=water_surface_param[4],
                               waterDepth=water_surface_param[5],
                               windSpeed=water_surface_param[6])
        
        self._id += 1

       
            
    
    def update_render(self, render_param: np.ndarray, caustics_param: np.ndarray):
        if self.raw_rgba is not None:
            if self.raw_rgba.size !=0:
                backscatter_value = wp.vec3f(*render_param[0:3])
                atten_coeff = wp.vec3f(*render_param[6:9])
                backscatter_coeff = wp.vec3f(*render_param[3:6])
                # If we are not blending caustics, we can just call UW_render shader
                # wp.launch(
                #     dim=(self.raw_rgba.shape[0], self.raw_rgba.shape[1]),
                #     kernel=UW_render,
                #     inputs=[
                #         self.raw_rgba,
                #         self.depth_image,
                #         backscatter_value,
                #         atten_coeff,
                #         backscatter_coeff
                #     ],
                #     outputs=[
                #         self.uw_image
                #     ]
                # )  
                wp.launch(
                    kernel=water_caustics,
                    dim=(self.width, self.height),  # (x, y)
                    inputs=[self.caustics, self.width, self.height, self._time, caustics_param[5]],
                )


                # Launch depth to world kernel once (this doesn't change)
                wp.launch(
                    kernel=depth_to_world_pos,
                    dim=(self.width, self.height),  # Launch dimensions (width, height)
                    inputs=[
                        self.depth_image,
                        wp.mat44f(self.camera_param["cameraProjection"].reshape(4, 4)),
                        wp.mat44f(self.camera_param["cameraViewTransform"].reshape(4, 4)),
                        self.width,
                        self.height
                    ],
                    outputs=[self.world_points],
                    device=str(self._device)
                )
                wp.launch(
                    kernel=blend_caustics,
                    dim=(self.width, self.height),
                    inputs=[
                        self.raw_rgba,
                        self.world_points,
                        self.normal_image,
                        self.caustics,
                        wp.vec3f(0.0, 0.0, 1.0),
                        caustics_param[0],       # blend_weight
                        caustics_param[3],       # uv_scale_x (horizontal scaling)
                        caustics_param[4],       # uv_scale_y (vertical scaling)
                        caustics_param[1],       # depth_min
                        caustics_param[2],   # depth_max
                        self.width,         # tex_w
                        self.height,         # tex_h
                    ],
                    outputs=[self.uw_image],
                    device=str(self._device)
                )


                wp.launch(
                dim=(self.height, self.width),
                kernel=UW_render_2,
                inputs=[
                    self.uw_image,
                    self.depth_image,
                    1.0,
                    backscatter_value,
                    atten_coeff,
                    backscatter_coeff
                ],
                outputs=[
                    self.uw_image
                ]
                )  
                
                self.image_provider.set_bytes_data_from_gpu(self.uw_image.ptr, [self.width, self.height])

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
