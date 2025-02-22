# Omniverse import
import numpy as np
import omni.replicator.core as rep
from omni.replicator.core.scripts.functional import write_np, write_image
import omni.ui as ui
import warp as wp

# Isaac sim import
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_path
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.viewports import set_camera_view

# Custom import
from ..utils.UWrenderer_utils import UW_render

class MHL_colorpicker_Scenario():
    def __init__(self):
        self._rob = None
        self._cam = None
        self.raw_rgba = None
        self.depth_image = None
        self._running_scenario = False
        self._time = 0.0



    def setup_scenario(self, rob, cam):
        self._rob = rob
        self._cam = cam

        # self._vel_buffer = []
        # self._range_buffer = []
        # self._singleBeam_buffer = []
        # self._output_dir = '/home/haoyu/Desktop//MHL_replica'
        # self._backend = rep.BackendDispatch({"paths": {"out_dir": self._output_dir}})

        self._running_scenario = True
        self._device = str(wp.get_preferred_device())

        SingleRigidPrim(prim_path=get_prim_path(self._rob),
                        translation=np.array([0.0, 0.0, 0.7]),
                        orientation=euler_angles_to_quat(np.array([0.0, 0.0, 0.0]), degrees=True))
        

        rp = rep.create.render_product(
            camera='/World/rob/Camera',
            resolution=(1920,1080),
            )
        
        self._ldr = rep.AnnotatorRegistry.get_annotator(name = "LdrColor", device=self._device)
        self._depth = rep.AnnotatorRegistry.get_annotator(name = 'distance_to_camera', device=self._device)
        self._ldr.attach(rp)
        self._depth.attach(rp)

        set_camera_view(eye=[-2.0, 0.0, 3.0], target=np.array([0.0, 1.25, 0.15]), camera_prim_path="/OmniverseKit_Persp")
        
        
        self.window = ui.Window("Camera", width=1280, height=720 + 40, visible=True)
        self.image_provider = ui.ByteImageProvider()
        with self.window.frame:
            with ui.ZStack(height=720):
                ui.Rectangle(style={"background_color": 0xFF000000})
                ui.Label('Run the scenario for image to be received',
                         style={'font_size': 55,'alignment': ui.Alignment.CENTER},
                         word_wrap=True)
                ui.ImageWithProvider(self.image_provider, width=1280, height=720,
                                     style={'fill_policy': ui.FillPolicy.PRESERVE_ASPECT_FIT,
                                    'alignment' :ui.Alignment.CENTER})

            

    def teardown_scenario(self):
        self._rob = None
        self._running_scenario = False
        self._time = 0.0
        self._id = 0


    def update_scenario(self, step: float, render_param: np.ndarray):

        
        if not self._running_scenario:
            return
        self._time += step
        if self._ldr.get_data().size == 0:
            return
        self.raw_rgba = self._ldr.get_data()
        self.depth_image = self._depth.get_data()        
        SingleRigidPrim(prim_path=get_prim_path(self._rob)).set_linear_velocity(np.array([0.5,0,0]))
        
        self.update_camera_render(render_param)
        # self._backend.schedule(write_image, path=f'cam/rgb_{self._id}.png', data=uw_image)
        # self._vel_buffer.append(self._DVL.get_linear_vel())
        # self._range_buffer.append(self._DVL.get_depth())
        # self._singleBeam_buffer.append(self._DVL.get_single_beam_range())
        self._id += 1

       
            
    
    def update_camera_render(self, render_param: np.ndarray):
        if self.raw_rgba is not None:
            if self.raw_rgba.size !=0:
                backscatter_value = wp.vec3f(*render_param[0:3])
                atten_coeff = wp.vec3f(*render_param[6:9])
                backscatter_coeff = wp.vec3f(*render_param[3:6])
                uw_image = wp.zeros_like(self.raw_rgba)
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
                        uw_image
                    ]
                )  
                
                self.image_provider.set_bytes_data_from_gpu(uw_image.ptr, [uw_image.shape[1], uw_image.shape[0]])

        
    
    def save(self):
        np.save(self._output_dir + "/vel.npy", self._vel_buffer)
        np.save(self._output_dir + "/range.npy", self._range_buffer)
        np.save(self._output_dir + "/singleBeam.npy", self._singleBeam_buffer)

        print(f'data has been saved to {self._output_dir}')

        
    
    
    
    
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
