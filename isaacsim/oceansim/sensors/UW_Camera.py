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
from isaacsim.oceansim.utils.UWrenderer_utils import UW_render

'''
Attention:

Before OceanSim extension being activated, the extension isaacsim.ros2.bridge should be activated, otherwise rclpy will
fail to be loaded.

so, we suggest that make sure the extension isaacsim.ros2.bridge is being setup to "AUTOLOADED" in Window->Extension.
'''
import rclpy
from sensor_msgs.msg import CompressedImage
import time
import cv2

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
                   physics_sim_view=None,
                   enable_ros2_pub=True, uw_img_topic="/oceansim/robot/uw_img", ros2_pub_frequency=5, ros2_pub_jpeg_quality=50):
        
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
            enable_ros2_pub (bool, optional): Enable ROS2 communication. Defaults to True.
            uw_img_topic (str, optional): ROS2 topic name for UW image. Defaults to "/oceansim/robot/uw_img".
            ros2_pub_frequency (int, optional): ROS2 publish frequency. Defaults to 5.
            ros2_pub_jpeg_quality (int, optional): ROS2 publish jpeg quality. Defaults to 50.
    
        """
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

        # ROS2 configuration
        self._enable_ros2_pub = enable_ros2_pub
        self._uw_img_topic = uw_img_topic
        self._last_publish_time = 0.0
        self._ros2_pub_frequency = ros2_pub_frequency     # publish frequency, hz
        self._ros2_pub_jpeg_quality = ros2_pub_jpeg_quality
        self._setup_ros2_publisher()
        
        print(f'[{self._name}] Initialized successfully. Data writing: {self._writing}')
    
    def _setup_ros2_publisher(self):
        '''
        setup the publisher for uw image
        '''
        try:
            if not self._enable_ros2_pub:
                return
            
            # Initialize ROS2 context if not already done
            if not rclpy.ok():
                rclpy.init()
                print(f'[{self._name}] ROS2 context initialized')

            # Create uw image publisher node
            node_name = f'oceansim_rob_uw_img_pub_{self._name.lower()}'.replace(' ', '_')
            self._ros2_uw_img_node = rclpy.create_node(node_name)
            self._uw_img_pub = self._ros2_uw_img_node.create_publisher(
                CompressedImage, 
                self._uw_img_topic, 
                10
            )
        
        except Exception as e:
            print(f'[{self._name}] ROS2 uw image publisher setup failed: {e}')

    def _ros2_publish_uw_img(self, uw_img):
        """
        publish the uw image
        """
        try:
            if self._uw_img_pub is None:
                return

            # fps control
            current_time = time.time()
            if current_time - self._last_publish_time < (1.0 / self._frequency):
                return

            # Convert the image
            uw_image_cpu = uw_img.numpy()
            if uw_image_cpu.dtype != np.uint8:
                uw_image_cpu = uw_image_cpu.astype(np.uint8)    # UW_render return 'rgba'
            uw_image_bgr = cv2.cvtColor(uw_image_cpu, cv2.COLOR_RGBA2BGR)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self._ros2_pub_jpeg_quality]      # JPEG quality, default 90, can be 0-100
            result, compressed_img = cv2.imencode('.jpg', uw_image_bgr, encode_param)
            if not result:
                print(f'[{self._name}] Failed to compress image to JPEG')
                return

            # Create a ROS2 Image message
            msg = CompressedImage()
            msg.header.stamp = self._ros2_uw_img_node.get_clock().now().to_msg()
            msg.header.frame_id = 'uw_image'
            msg.format = 'jpeg'
            msg.data = compressed_img.tobytes()
            
            # Publish the message
            self._uw_img_pub.publish(msg)

            rclpy.spin_once(self._ros2_uw_img_node, timeout_sec=0.0)

            self._last_publish_time = current_time

            # debug
            # self._ros2_uw_img_node.get_logger().info(
            #     f'Published image: encoding={msg.encoding}, '
            #     f'width={msg.width}, height={msg.height}, step={msg.step}, '
            #     f'data_size={len(msg.data)}'
            # )

        except Exception as e:
            print(f'[{self._name}] ROS2 uw image publish failed: {e}')

    def render(self):
        """Process and display a single frame with underwater effects.
    
        Note:
            - Updates viewport display if enabled
            - Saves image to disk if writing_dir was specified
        """
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
            if self._enable_ros2_pub:
                self._ros2_publish_uw_img(uw_image)
                pass

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

        
       