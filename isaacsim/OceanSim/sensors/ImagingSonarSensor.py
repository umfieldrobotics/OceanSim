from isaacsim.sensors.camera import Camera
import omni.replicator.core as rep
import omni.ui as ui
import numpy as np
from omni.replicator.core.scripts.functional import write_np
import warp as wp
# Future TODO
# In future release, wrap this class around RTX lidar


@wp.func
def cartesian_to_spherical(cart: wp.vec3) -> wp.vec3:
    r = wp.sqrt(cart[0]*cart[0] + cart[1]*cart[1] + cart[2]*cart[2])
    return wp.vec3(r,
                wp.atan2(cart[1], cart[0]),
                wp.acos(cart[2] / r)
                )
                                    

@wp.kernel
def compute_intensity(pcl: wp.array(ndim=2, dtype=wp.float32),
                    normals: wp.array(ndim=2, dtype=wp.float32),
                    viewTransform: wp.mat44,
                    semantics: wp.array(ndim=1, dtype=wp.uint32),
                    indexToRefl: wp.array(dtype=wp.float32),
                    attenuation: float,
                    intensity: wp.array(dtype=wp.float32)
                    ):
    tid = wp.tid()
    pcl_vec = wp.vec3(pcl[tid,0], pcl[tid,1], pcl[tid,2])
    normal_vec = wp.vec3(normals[tid,0], normals[tid,1],normals[tid,2])
    R = wp.mat33(viewTransform[0,0], viewTransform[0,1], viewTransform[0,2],
                 viewTransform[1,0], viewTransform[1,1], viewTransform[1,2],
                 viewTransform[2,0], viewTransform[2,1], viewTransform[2,2])
    T = wp.vec3(viewTransform[0,3], viewTransform[1,3], viewTransform[2,3])
    sensor_loc = - (wp.transpose(R) @ T)
    incidence = pcl_vec - sensor_loc
    # Will use warp.math.norm_l2() in future release
    dist = wp.sqrt(incidence[0]*incidence[0] + incidence[1]*incidence[1] + incidence[2]*incidence[2])
    unit_directs = wp.normalize(pcl_vec - sensor_loc)
    cos_theta = wp.dot(-unit_directs, normal_vec)
    reflectivity = indexToRefl[semantics[tid]]
    intensity[tid] = reflectivity * cos_theta * wp.exp(-attenuation * dist)

@wp.kernel
def world2local(viewTransform: wp.mat44,
                pcl_world: wp.array(ndim=2, dtype=wp.float32),
                pcl_local: wp.array(dtype=wp.vec3),
                pcl_local_spher: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    pcl_world_homogeneous = wp.vec4(pcl_world[tid,0],
                          pcl_world[tid,1],
                          pcl_world[tid,2],
                          wp.float32(1.0)
                          )
    pcl_local_homogeneous = viewTransform @ pcl_world_homogeneous
    # Rotate axis such that y axis pointing forward for sonar data plotting
    pcl_local[tid] = wp.vec3(pcl_local_homogeneous[0], -pcl_local_homogeneous[2], pcl_local_homogeneous[1])
    pcl_local_spher[tid] = cartesian_to_spherical(pcl_local[tid])


@wp.kernel
def bin_intensity(pcl: wp.array(dtype=wp.vec3),
                  intensity: wp.array(dtype=wp.float32),
                  x_offset: wp.float32,
                  y_offset: wp.float32,
                  x_res: wp.float32,
                  y_res: wp.float32,
                  bin_sum: wp.array(ndim=2, dtype=wp.float32),
                  bin_count: wp.array(ndim=2, dtype=wp.int32)
                  ):
    tid = wp.tid()

    # Get the range, azimuth, and intensity of the point
    x = pcl[tid][0]
    y = pcl[tid][1]

    # Calculate the bin indices for range and azimuth
    x_bin_idx = wp.int32((x - x_offset) / x_res)
    y_bin_idx = wp.int32((y - y_offset) / y_res)
    wp.atomic_add(bin_sum, x_bin_idx, y_bin_idx, intensity[tid])
    wp.atomic_add(bin_count, x_bin_idx, y_bin_idx, 1)

@wp.kernel 
def average(sum: wp.array(ndim=2, dtype=wp.float32),
            count: wp.array(ndim=2, dtype=wp.int32),
            avg: wp.array(ndim=2, dtype=wp.float32)):
    i, j = wp.tid()
    if count[i, j] > 0:
        avg[i, j] = sum[i, j] / wp.float32(count[i, j])


@wp.kernel
def all_max(array: wp.array(ndim=2, dtype=wp.float32), 
              max_value: wp.array(dtype=wp.float32)):
    i,j = wp.tid()  
    wp.atomic_max(max_value, 0, array[i, j])

@wp.kernel
def range_max(array: wp.array(ndim=2, dtype=wp.float32), 
              max_value: wp.array(dtype=wp.float32)):
    i, j = wp.tid()
    wp.atomic_max(max_value, i, array[i,j])

@wp.kernel 
def make_sonar_map_all(r: wp.array(ndim=2, dtype=wp.float32),
                       azi: wp.array(ndim=2, dtype=wp.float32),
                       intensity: wp.array(ndim=2, dtype=wp.float32),
                       max_intensity: wp.float32,
                       gau_noise: wp.array(ndim=2, dtype=wp.float32),
                       range_ray_noise: wp.array(ndim=2, dtype=wp.float32),
                       offset: wp.float32,
                       gain: wp.float32,
                       result: wp.array(ndim=2, dtype=wp.vec3)):
    i, j = wp.tid()
    intensity[i,j] = intensity[i,j]/max_intensity
    intensity[i,j] += offset
    intensity[i,j] *= gain
    intensity[i,j] *= (0.5 + gau_noise[i,j])
    intensity[i,j] += range_ray_noise[i,j]
    intensity[i,j] = wp.clamp(intensity[i,j], wp.float32(0.0), wp.float32(1.0))

    result[i,j] = wp.vec3(r[i,j] * wp.cos(azi[i,j]),
                          r[i,j] * wp.sin(azi[i,j]),
                          intensity[i,j])

@wp.kernel 
def make_sonar_map_range(r: wp.array(ndim=2, dtype=wp.float32),
                       azi: wp.array(ndim=2, dtype=wp.float32),
                       intensity: wp.array(ndim=2, dtype=wp.float32),
                       max_intensity: wp.array(ndim=1, dtype=wp.float32),
                       gau_noise: wp.array(ndim=2, dtype=wp.float32),
                       range_ray_noise: wp.array(ndim=2, dtype=wp.float32),
                       offset: wp.float32,
                       gain: wp.float32,
                       result: wp.array(ndim=2, dtype=wp.vec3)):
    i, j = wp.tid()

    if max_intensity[i] !=0:
        intensity[i,j] = intensity[i,j]/max_intensity[i]

    intensity[i,j] *= (0.5 + gau_noise[i,j])
    intensity[i,j] += range_ray_noise[i,j]
    intensity[i,j] += offset
    intensity[i,j] *= gain
    intensity[i,j] = wp.clamp(intensity[i,j], wp.float32(0.0), wp.float32(1.0))

    result[i,j] = wp.vec3(r[i,j] * wp.cos(azi[i,j]),
                          r[i,j] * wp.sin(azi[i,j]),
                          intensity[i,j])
    
@wp.kernel
def make_sonar_image(sonar_data: wp.array(ndim=2, dtype=wp.vec3),
                     sonar_image: wp.array(ndim=3, dtype=wp.uint8)):
    i, j = wp.tid()
    width = sonar_data.shape[1]
    sonar_rgb = wp.uint8(sonar_data[i,j][2] * wp.float32(255))
    sonar_image[i,width-j,0] = sonar_rgb
    sonar_image[i,width-j,1] = sonar_rgb
    sonar_image[i,width-j,2] = sonar_rgb
    sonar_image[i,width-j,3] = wp.uint8(255)


class ImagingSonarSensor(Camera):
    def __init__(self, 
                 prim_path, 
                 name = "ImagingSonar", 
                 frequency = None, 
                 dt = None, 
                 position = None, 
                 orientation = None, 
                 translation = None, 
                 render_product_path = None,
                 physics_sim_view = None,
                 min_range: float = 0.2, # m
                 max_range: float = 3.0, # m
                 range_res: float = 0.008, # deg
                 hori_fov: float = 130.0, # deg
                 vert_fov: float = 20.0, # deg
                 angular_res: float = 0.5, # deg
                 hori_res: int = 3000 # isaac camera render product only accepts square pixel, 
                                      # for now vertical res is automatically set with ratio of hori_fov vs.vert_fov 
                 ):
        self._name = name
        # Raw parameters from Oculus M370s\MT370s\MD370s
        self.max_range = max_range # m (max is 200 m in datasheet )
        self.min_range = min_range # m (min is 0.2 m in datasheet)
        self.range_res = range_res # m (datasheet is 0.008 m)
        self.hori_fov = hori_fov # degree (hori_fov is 130 degrees in datasheet)
        self.vert_fov = vert_fov # degree (vert_fov is 20 degrees in datasheet)
        self.angular_res = angular_res # degree (datasheet is 2 deg)
        self.hori_res= hori_res

        # self.beam_separation = 0.5 # degree (Not USED FOR NOW)!!
        # self.num_beams = 256 # (max number of beams) (NOT USED FOR NOW)!!
        # self.update_rate = 40 # Hz (max update rate) (NOT USED FOR NOW)!!


        # Generate sonar map's r and z meshgrid
        self.r, self.azi = np.meshgrid(np.arange(self.min_range,self.max_range,self.range_res),
                                       np.arange(np.deg2rad(90-self.hori_fov/2), np.deg2rad(90+self.hori_fov/2), np.deg2rad(self.angular_res)),
                                       indexing='ij')

        # Load array that doesn't change shapes to cuda for reusage memory
        # Users can also automatically see if they have set a reasonable parameter 
        # for sonar map bin size\resolution once load the sensor
        self.bin_sum = wp.zeros(shape=self.r.shape, dtype=wp.float32)
        self.bin_count = wp.zeros(shape=self.r.shape, dtype=wp.int32)
        self.binned_intensity = wp.zeros(shape=self.r.shape, dtype=wp.int32)
        self.sonar_map = wp.zeros(shape=self.r.shape, dtype=wp.vec3)
        self.sonar_image = wp.zeros(shape=(self.r.shape[0], self.r.shape[1], 4), dtype=wp.uint8)


        self.AR = self.hori_fov / self.vert_fov
        self.vert_res = int(self.hori_res / self.AR)
        # By doing this, I am assuming the vertical beam separation
        # is the same as the beam horizontal separation. 
        # This is bacause replicator raytracing is specified as resolutions
        # while non-squre pixel is not supported in Isaac sim. See details below.
        
        super().__init__(prim_path=prim_path, 
                         name=name, 
                         frequency=frequency,
                         dt=dt, 
                         resolution=[self.hori_res, self.vert_res],
                         position=position, 
                         orientation=orientation, 
                         translation=translation, 
                         render_product_path=render_product_path)

        self.set_clipping_range(
            near_distance=self.min_range,
            far_distance=self.max_range
        )
        # This is a bug. Needs to call initialize() before changing aperture
        # https://forums.developer.nvidia.com/t/error-when-setting-a-cameras-vertical-horizontal-aperture/271314
        # This line initialize the camera
        self.initialize(physics_sim_view)

        # Assume the default focal length to compute the desired horizontal aperture
        # The reason why we are doing this is because Isaac sim will fix vertical aperture
        # given aspect ratio for mandating square pixles
        # https://forums.developer.nvidia.com/t/how-to-modify-the-cameras-field-of-view/278427/5
        self.focal_length = self.get_focal_length()
        horizontal_aper = 2 * self.focal_length * np.tan(np.deg2rad(self.hori_fov) / 2)
        self.set_horizontal_aperture(horizontal_aper)
        # Notice if you would like to observe sonar view from linked viewport.
        # Only horizontal fov is displayed correctly while the vertical fov is
        # followed by your viewport aspect ratio settings.
        

    # Initialize the sensor so that annotator is 
    # loaded on cuda and ready to acquire data
    # Data is generated per simulation tick

    # do_array_copy: If True, retrieve a copy of the data array. 
    # This is recommended for workflows using asynchronous
    # backends to manage the data lifetime. 
    # Can be set to False to gain performance if the data is 
    # expected to be used immediately within the writer. Defaults to True.

    def sonar_initialize(self, output_dir : str = None, viewport: bool = True, include_unlabelled = False, if_array_copy: bool = True):
        self.writing = False
        self._viewport = viewport
        self._device = str(wp.get_preferred_device())
        self.scan_data = {}
        self.id = 0

        self.pointcloud_annot = rep.AnnotatorRegistry.get_annotator(
            name="pointcloud",
            init_params={"includeUnlabelled": include_unlabelled},
            do_array_copy=if_array_copy,
            device=self._device
            )
        
        self.cameraParams_annot = rep.AnnotatorRegistry.get_annotator(
            name="CameraParams",
            do_array_copy=if_array_copy,
            device=self._device
            )
        
        self.semanticSeg_annot = rep.AnnotatorRegistry.get_annotator(
            name='semantic_segmentation',
            init_params={"colorize": False},
            do_array_copy=if_array_copy,
            device=self._device
        )

        print(f'[{self._name}] Using {self._device}' )
        print(f'[{self._name}] Render query res: {self.hori_res} x {self.vert_res}. Binning res: {self.r.shape[0]} x {self.r.shape[1]}')

        self.pointcloud_annot.attach(self._render_product_path)
        self.cameraParams_annot.attach(self._render_product_path)
        self.semanticSeg_annot.attach(self._render_product_path)
        
        if output_dir is not None:
            self.writing = True
            self.backend = rep.BackendDispatch({"paths": {"out_dir": output_dir}})
        if self._viewport:
            self.make_sonar_viewport()
        
        print(f'[{self._name}] Initialized successfully. Data writing: {self.writing}')

        self.bin_sum.zero_()
        self.bin_count.zero_()
        self.binned_intensity.zero_()
        self.sonar_map.zero_()
        self.sonar_image.zero_()

        

    def scan(self):
        # Due to the time to load annotator to cuda, the first few simulation tick gives no annotation in memory.
        # This would also reult error when no mesh within the sonar fov
        # Ignore scan that gives empty data stream
        if len(self.semanticSeg_annot.get_data()['info']['idToLabels']) !=0:
            self.scan_data['pcl'] = self.pointcloud_annot.get_data(device=self._device)['data'][0]  # shape :(1,N,3) <class 'warp.types.array'>
            self.scan_data['normals'] = self.pointcloud_annot.get_data(device=self._device)['info']['pointNormals'][0] # shape :(1,N,4) <class 'warp.types.array'>
            self.scan_data['semantics'] = self.pointcloud_annot.get_data(device=self._device)['info']['pointSemantic'][0] # shape: (1, N) <class 'warp.types.array'>
            self.scan_data['viewTransform'] = self.cameraParams_annot.get_data()['cameraViewTransform'].reshape(4,4).T # 4 by 4 np.ndarray extrinsic matrix
            self.scan_data['idToLabels'] = self.semanticSeg_annot.get_data()['info']['idToLabels'] # dict 
            return True
        else:
            return False


    def make_sonar_data(self, 
                        binning_method: str = "sum", 
                        normalizing_method: str = "range",
                        query_prop: str ='reflectivity', # Do not modify this if not developing the sensor.
                        attenuation: float = 0.1, # Control the attentuation along distance when computing attenuation
                        gau_noise_param: float = 0.2, # multiplicative noise coefficient 
                        ray_noise_param: float = 0.05, # additive noise parameter
                        intensity_offset: float = 0.0, # offset intensity after normalization
                        intensity_gain: float = 1.0, # scale intensity after normalization
                        central_peak: float = 2, # control the strength of the streak
                        central_std: float = 0.001, # control the spread of the streak
                        ):
        # A utility function helps to convert idToLabels into indexToProp array
        # This manipulation facilitates warp computation framework
        # indexToProp is an 1-dim array where the values associated with the query property 
        # are placed at the index corresponding to the key
        # First two entry are always zero because {'0': {'class': 'BACKGROUND'}, '1': {'class': 'UNLABELLED'}}
        # eg: indexToProp = [0, 0, 0.1, 1 .....] 
        def make_indexToProp_array(idToLabels: dict, query_property: str):
            max_id = max(idToLabels.keys(), default=-1)
            indexToProp_array = np.ones((int(max_id)+1,))
            for id in idToLabels.keys():
                for property in idToLabels.get(id):
                    if property == query_property:
                        indexToProp_array[int(id)] = idToLabels.get(id).get(property)
            return indexToProp_array

        if self.scan():
            num_points = self.scan_data['pcl'].shape[0]
            # Load these small numpy arrays to cuda
            indexToRefl = wp.array(make_indexToProp_array(idToLabels=self.scan_data['idToLabels'],
                                                         query_property=query_prop),
                                                         dtype=wp.float32)
            viewTransform=wp.mat44(self.scan_data['viewTransform'])
            # directly use warp array loaded on cuda
            pcl = self.scan_data['pcl']
            normals = self.scan_data['normals']
            semantics = self.scan_data['semantics']
        else:
            return

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
                      attenuation,
                  ],
                  outputs=[
                      intensity
                  ]
                )
                
        # Transform pointcloud from world cooridates to sonar local
        pcl_local =wp.empty(shape=(num_points,), dtype=wp.vec3)
        pcl_spher = wp.empty(shape=(num_points,), dtype=wp.vec3)
        wp.launch(kernel=world2local,
                  dim=num_points,
                  inputs=[
                      viewTransform,
                      pcl
                  ],
                    outputs=[
                      pcl_local,
                      pcl_spher
                    ]
                )
        
        # Collapse three dimensional intensity data to 2D
        # Simply sum intensity return and compute number of return that falls into the same bin
        self.bin_sum.zero_()
        self.bin_count.zero_()
        self.binned_intensity.zero_()

        
        wp.launch(kernel=bin_intensity,
                  dim=num_points,
                  inputs=[
                      pcl_spher,
                      intensity,
                      self.r[0,0],
                      self.azi[0,0],
                      self.range_res,
                      wp.radians(self.angular_res),
                  ],
                  outputs=[
                      self.bin_sum,
                      self.bin_count
                  ]
                  )
        
        # Process intensity data by either sum as it is or averaging
        if binning_method == "mean":
            wp.launch(
                kernel=average,
                dim=self.bin_sum.shape,
                inputs=[
                    self.bin_sum,
                    self.bin_count
                ],
                outputs=[
                    self.binned_intensity,
                ]
                )
        
        if binning_method == "sum":
            self.binned_intensity = self.bin_sum


        # Calculate additive rayleigh noise (range dependent and mimic central beam)
        # Calculate multiplicative gaussian noise
        gau_noise = np.random.normal(loc=0, scale=gau_noise_param, size=self.bin_sum.shape)
        ray_noise = np.random.rayleigh(scale=ray_noise_param, size=self.bin_sum.shape)
        range_dependent_ray_noise = (self.r/self.max_range)**2*(1 + central_peak*np.exp(-(self.azi-np.pi/2)**2/central_std))*ray_noise 

        # Normalizing intensity at each bin either by global maximum or rangewise maximum

        self.sonar_map.zero_()

        # Compute global maximum
        if normalizing_method == "all":
            # warp.max(scalar, scalar) has bug. Now using the warp.atomic_max(array, i, value)
            maximum = wp.zeros(shape=(1,), dtype=wp.float32)
            wp.launch(
                dim=self.bin_sum.shape,
                kernel=all_max,
                inputs=[
                    self.binned_intensity,
                ],
                outputs=[
                    maximum # wp.array of shape (1,)
                ]
            )
            # TODO in future release, this will be fixed so everything stays on CUDA
            maximum = maximum.numpy()[0]
            # Apply noise, normalize by global maximum, and convert (r, azi) to (x,y) for plotting
            wp.launch(
                  kernel=make_sonar_map_all,
                  dim=self.sonar_map.shape,
                  inputs=[
                      wp.array(self.r, ndim=2, dtype=wp.float32),
                      wp.array(self.azi, ndim=2, dtype=wp.float32),
                      self.binned_intensity,
                      maximum,
                      wp.array(gau_noise, ndim=2, dtype=wp.float32),
                      wp.array(range_dependent_ray_noise, ndim=2, dtype=wp.float32),
                      intensity_offset,
                      intensity_gain
                  ],
                  outputs=[
                      self.sonar_map
                  ]
                  )
            
        if normalizing_method == "range":
            # Compute rangewise maximum
            maximum = wp.zeros(shape=(self.r.shape[0],), dtype=wp.float32)
            wp.launch(
                dim=self.bin_sum.shape,
                kernel=range_max,
                inputs=[
                    self.binned_intensity,
                ],
                outputs=[
                    maximum      # wp.array of shape (number of range bins, )
                ]
            )
            # Apply noise, normalize by range maximum, and convert (r, azi) to (x,y) for plotting
            wp.launch(
                  kernel=make_sonar_map_range,
                  dim=self.sonar_map.shape,
                  inputs=[
                      wp.array(self.r, ndim=2, dtype=wp.float32),
                      wp.array(self.azi, ndim=2, dtype=wp.float32),
                      self.binned_intensity,
                      maximum,
                      wp.array(gau_noise, ndim=2, dtype=wp.float32),
                      wp.array(range_dependent_ray_noise, ndim=2, dtype=wp.float32),
                      intensity_offset,
                      intensity_gain
                  ],
                  outputs=[
                      self.sonar_map
                  ]
                  )
        
        
        # Write data to the dir
        if self.writing:
            # self.backend.schedule(write_np, f"intensity_{self.id}.npy", data=intensity)
            # self.backend.schedule(write_np, f'pcl_local_{self.id}.npy', data=pcl_local)
            self.backend.schedule(write_np, f'sonar_data_{self.id}.npy', data=self.sonar_map)
            print(f"[{self._name}] [{self.id}] Writing sonar data to {self.backend.output_dir}")
        
        if self._viewport:
            self._sonar_provider.set_bytes_data_from_gpu(self.make_sonar_image().ptr, 
                                                    [self.sonar_map.shape[1], self.sonar_map.shape[0]])
            # self.backend.schedule(write_image, f'sonar_{self.id}.png', data = self.make_sonar_image())        
            
        self.id += 1
    

    # This is a utility function that converts sonar_data to grey scale sonar image for viewport visualization
    def make_sonar_image(self):
        self.sonar_image.zero_()
        wp.launch(
            dim=self.sonar_map.shape,
            kernel=make_sonar_image,
            inputs=[
                self.sonar_map
            ],
            outputs=[
                self.sonar_image
            ]
        )
        return self.sonar_image
    

    def make_sonar_viewport(self):
        self.wrapped_ui_elements = []

        range_tick_num = 10
        range_tick = np.round(np.linspace(self.min_range, self.max_range, range_tick_num), 2)

        azi_tick_num = 10
        azi_tick = np.round(np.linspace(90-self.hori_fov/2, 90+self.hori_fov/2, azi_tick_num))
        self._sonar_provider = ui.ByteImageProvider()
        self._window = ui.Window(self._name, width=800, height=800, visible=True)
        
        with self._window.frame:
            with ui.ZStack(height=720, width = 720):
                ui.Rectangle(widthstyle={"background_color": 0xFF000000})
                ui.Label('Run the scenario for image to be received',
                         style={'font_size': 55,'alignment': ui.Alignment.CENTER},
                         word_wrap=True)
                sonar_image_provider = ui.ImageWithProvider(self._sonar_provider, 
                                    style={"width": 720, 
                                        "height": 720, 
                                        "fill_policy" : ui.FillPolicy.STRETCH,
                                        'alignment': ui.Alignment.CENTER})
                
                # ui.Line(alignment=ui.Alignment.LEFT,
                #         style={'border_width': 2,
                #                 'color':ui.color.white })
                # with ui.VGrid(row_height = 720/(range_tick_num-1)):
                #     for i in range(range_tick_num-1):
                #         with ui.ZStack():
                #             ui.Rectangle(style={'border_color': ui.color.white, 'background_color': ui.color.transparent,'border_width': 0.05, 'margin': 0})
                #             ui.Label(str(range_tick[i]) + ' m',style={'font_size': 15,'alignment': ui.Alignment.LEFT, 'margin':2})
                # with ui.HGrid(column_width = 720/(azi_tick_num-1), direction=ui.Direction.RIGHT_TO_LEFT):
                #     for i in range(azi_tick_num-1):
                #         with ui.ZStack():
                #             ui.Rectangle(style={'border_color': ui.color.white, 'background_color': ui.color.transparent,'border_width': 0.05, 'margin': 0})
                #             ui.Label(str(azi_tick[i]) + "°",style={'font_size': 15,'alignment': ui.Alignment.RIGHT, 'margin':2})                           
                # ui.Label(str(range_tick[-1]) +" m", style={'font_size': 15, "alignment":ui.Alignment.LEFT_BOTTOM, 'margin':2})
        
        self.wrapped_ui_elements.append(sonar_image_provider)
        self.wrapped_ui_elements.append(self._sonar_provider)
        self.wrapped_ui_elements.append(self._window)

    def get_range(self):
        return [self.min_range, self.max_range]
    
    def get_fov(self):
        return [self.hori_fov, self.vert_fov]
    

    
    # Detach the annotator from render product and clear the data cache
    def close(self):
        self.pointcloud_annot.detach(self._render_product_path)
        self.cameraParams_annot.detach(self._render_product_path)
        self.semanticSeg_annot.detach(self._render_product_path)

        rep.AnnotatorCache.clear(self.pointcloud_annot)
        rep.AnnotatorCache.clear(self.cameraParams_annot)
        rep.AnnotatorCache.clear(self.semanticSeg_annot)


        print(f'[{self._name}] Annotator detached. AnnotatorCache cleaned.')

        if self._viewport:
            self.ui_destroy()


    def ui_destroy(self):
        for elem in self.wrapped_ui_elements:
            elem.destroy()