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

@wp.kernel
def add_blue_tint(
    image: wp.array(ndim=3, dtype=wp.uint8),  # Input/output image (RGBA format)
    blue_intensity: wp.uint8,            # Intensity of the blue tint (0.0 to 1.0)
):
    # Get the thread indices
    i, j = wp.tid()

    # Get the pixel color (RGBA)
    blue_pixel = image[i, j, 2]

    # Add blue tint to the blue channel
    blue_pixel = wp.clamp(blue_pixel + blue_intensity, wp.uint8(0), wp.uint8(255))

    # Write the modified pixel back to the image
    image[i, j, 2] = blue_pixel

class MHL_colorpicker_Scenario():
    def __init__(self):
        self._rob = None
        self._cam = None
        self._running_scenario = False
        self._time = 0.0
        

        self._output_dir = '/home/haoyu/Desktop/MHL_replica'



    def setup_scenario(self, rob, cam):
        self._rob = rob
        self._cam = cam

        self._running_scenario = True
        self._device = str(wp.get_preferred_device())

        SingleRigidPrim(prim_path=get_prim_path(self._rob),
                        translation=np.array([0.0, 0.0, 2.5]),
                        orientation=euler_angles_to_quat(np.array([0.0, 0.0, 0.0]), degrees=True))
        

        self._backend = rep.BackendDispatch({"paths": {"out_dir": self._output_dir}})
        rp = rep.create.render_product(
            camera='/MHL/rob/Camera',
            resolution=(1920,1080),
            )
        
        self._ldr = rep.AnnotatorRegistry.get_annotator(name = "LdrColor", device=self._device)
        self._depth = rep.AnnotatorRegistry.get_annotator(name = 'distance_to_camera', device=self._device)
        self._ldr.attach(rp)
        self._depth.attach(rp)

        set_camera_view(eye=[-2.0, 0.0, 3.0], target=np.array([0.0, 1.25, 0.15]), camera_prim_path="/OmniverseKit_Persp")
        
        
        self.window = ui.Window("Camera", width=1280, height=720, visible=True)
        self.image_provider = ui.ByteImageProvider()
        with self.window.frame:
            ui.Image('/home/haoyu-ma/.local/share/ov/pkg/isaac-sim-4.5.0/extsUser/OceanSim/data/icon.png',
                            fill_policy=ui.FillPolicy.PRESERVE_ASPECT_FIT,
                            alignment=ui.Alignment.CENTER)
            

    def teardown_scenario(self):
        self._rob = None
        self._running_scenario = False
        self._time = 0.0
        self._id = 0


    def update_scenario(self, step: float):

        
        if not self._running_scenario:
            return
        self._time += step
        if self._ldr.get_data().size == 0:
            return
        
        SingleRigidPrim(prim_path=get_prim_path(self._rob)).set_linear_velocity(np.array([5,0,0]))

        rgba = self._ldr.get_data()
        wp.launch(
            kernel=add_blue_tint,
            dim=(rgba.shape[0], rgba.shape[1]),
            inputs=[
                rgba,
                50
            ]
        )


        self.image_provider.set_bytes_data_from_gpu(rgba.ptr, [rgba.shape[1], rgba.shape[0]])



        with self.window.frame:
            ui.ImageWithProvider(self.image_provider, width=1000, height=1000)

        self._id += 1


    
    
    
    
    
    
    
    
    
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
