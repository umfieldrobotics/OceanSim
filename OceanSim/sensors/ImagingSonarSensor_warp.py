from isaacsim.sensors.camera import Camera
import omni.replicator.core as rep
import isaacsim.core.utils.rotations as rotations_utils
import numpy as np
from omni.replicator.core.scripts.functional import write_np
import warp as wp



@wp.func
def cartesian_to_spherical(cart: wp.vec3) -> wp.vec3:
    r = wp.sqrt(cart[0]*cart[0] + cart[1]*cart[1] + cart[2]*cart[2])
    return wp.vec3(r,
                wp.atan2(cart[1], cart[0]),
                wp.acos(cart[2] / r)
                )
                                    

@wp.kernel
def compute_intensity(pcl: wp.array(dtype=wp.vec3),
                    normals: wp.array(dtype=wp.vec3),
                    viewTransform: wp.mat44,
                    semantics: wp.array(dtype=wp.int8),
                    indexToRefl: wp.array(dtype=wp.float32),
                    attenuation: float,
                    intensity: wp.array(dtype=wp.float32)
                    ):
    tid = wp.tid()
    R = wp.mat33(viewTransform[0,0], viewTransform[0,1], viewTransform[0,2],
                 viewTransform[1,0], viewTransform[1,1], viewTransform[1,2],
                 viewTransform[2,0], viewTransform[2,1], viewTransform[2,2])
    T = wp.vec3(viewTransform[0,3], viewTransform[1,3], viewTransform[2,3])
    sensor_loc = - (wp.transpose(R) @ T)
    incidence = pcl[tid] - sensor_loc
    # Will use warp.math.norm_l2() in future release
    dist = wp.sqrt(incidence[0]*incidence[0] + incidence[1]*incidence[1] + incidence[2]*incidence[2])
    unit_directs = wp.normalize(pcl[tid] - sensor_loc)
    cos_theta = wp.dot(-unit_directs, normals[tid])
    reflectivity = indexToRefl[semantics[tid]]
    intensity[tid] = reflectivity * cos_theta * wp.exp(-attenuation * dist)

@wp.kernel
def world2local(viewTransform: wp.mat44,
                pcl_world: wp.array(dtype=wp.vec3),
                pcl_local: wp.array(dtype=wp.vec3),
                pcl_local_spher: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    pcl_world_homogeneous = wp.vec4(pcl_world[tid][0],
                          pcl_world[tid][1],
                          pcl_world[tid][2],
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
    intensity[i,j] += offset
    intensity[i,j] *= gain
    intensity[i,j] *= (0.5 + gau_noise[i,j])
    intensity[i,j] += range_ray_noise[i,j]
    intensity[i,j] = wp.clamp(intensity[i,j], wp.float32(0.0), wp.float32(1.0))

    result[i,j] = wp.vec3(r[i,j] * wp.cos(azi[i,j]),
                          r[i,j] * wp.sin(azi[i,j]),
                          intensity[i,j])

class ImagingSonarSensor:

    def __init__(self, prim_path : str, 
                 trans : list[float]= [0.0, 0.0, 0.0], 
                 orients: list[float] = rotations_utils.euler_angles_to_quat(np.array([0,0,0]))
                 ):
        
        # Raw parameters from Oculus M370s\MT370s\MD370s
        self.max_range = 3 # m (max is 200 m in datasheet )
        self.min_range = 0.5 # m (min is 0.2 m in datasheet)
        self.range_res = 0.008 # m (datasheet is 0.008 m)
        self.update_rate = 40 # Hz (max update rate) (NOT USED FOR NOW)!!
        self.hori_fov = 130 # degree (hori_fov is 130 degrees in datasheet)
        self.vert_fov = 20 # degree (vert_fov is 20 degrees in datasheet)
        self.num_beams = 256 # (max number of beams) (NOT USED FOR NOW)!!
        self.angular_res = 1.0 # degree (datasheet is 2 deg)
        self.beam_separation = 0.5 # degree

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

        # We introduce this factor to adjust raycast density
        # (Equivalently, adjust the beam_separation) 
        # Increase this value by 1 will quadraple the total number of raycasts
        self.ray_factor = 20 # below 15 is advised for me

        self.AR = self.hori_fov / self.vert_fov
        self.hori_res = int(self.ray_factor * (self.hori_fov / self.beam_separation))
        self.vert_res = int(self.hori_res / self.AR)
        # By doing this, I am assuming the vertical beam separation
        # is the same as the beam vertical separation. 
        # This is bacause replicator raytracing is specified as resolutions
        # while non-squre pixel is not supported in Isaac sim. See details below.

        print(f'resolution: {self.hori_res} x {self.vert_res}')
        self.camera_prim_path = prim_path + '/Sonar'
        self.camera = Camera(
            prim_path=self.camera_prim_path,
            translation=trans,
            orientation=orients,
            resolution=(self.hori_res, self.vert_res)
            )
        self.camera.set_clipping_range(
            near_distance=self.min_range,
            far_distance=self.max_range
        )
        # This is a bug. Needs to call initialize() before changing aperture
        # https://forums.developer.nvidia.com/t/error-when-setting-a-cameras-vertical-horizontal-aperture/271314
        self.camera.initialize()

        # Assume the default focal length to compute the desired horizontal aperture
        # The reason why we are doing this is because Isaac sim will fix vertical aperture
        # given aspect ratio for mandating square pixles
        # https://forums.developer.nvidia.com/t/how-to-modify-the-cameras-field-of-view/278427/5
        focal_length = self.camera.get_focal_length()
        horizontal_aper = 2 * focal_length * np.tan(np.deg2rad(self.hori_fov) / 2)
        self.camera.set_horizontal_aperture(horizontal_aper)
        # Notice if you would like to observe sonar view from linked viewport.
        # Only horizontal fov is displayed correctly while the vertical fov is
        # followed by your viewport aspect ratio settings.
        
        # Future: maybe able to increase W or H resolution and cut out data points outside of the view
        # But this method requires us to think about math about combinations 
        # of f and A_h will enable us to cut out the least number of points.


    # Initialize the sensor so that annotator is 
    # loaded on cuda and ready to acquire data
    # Data is generated per simulation tick
    def initialize(self, output_dir : str = None):
        self.writing = False
        self.scan_data = {}
        self.id = 0
        self.rp = rep.create.render_product(
            camera=self.camera_prim_path,
            resolution=(self.hori_res, self.vert_res)
            )

        self.pointcloud_annot = rep.AnnotatorRegistry.get_annotator(
            name="pointcloud",
            init_params={"includeUnlabelled": True},
            do_array_copy=True
            )
        
        self.cameraParams_annot = rep.AnnotatorRegistry.get_annotator(
            name="CameraParams",
            do_array_copy=True
            )
        
        self.semanticSeg_annot = rep.AnnotatorRegistry.get_annotator(
            name='semantic_segmentation',
            init_params={"colorize": False},
            do_array_copy=True
        )
        # do_array_copy: If True, retrieve a copy of the data array. 
        # This is recommended for workflows using asynchronous
        # backends to manage the data lifetime. 
        # Can be set to False to gain performance if the data is 
        # expected to be used immediately within the writer. Defaults to True.

        self.pointcloud_annot.attach(self.rp)
        self.cameraParams_annot.attach(self.rp)
        self.semanticSeg_annot.attach(self.rp)
        
        if output_dir is not None:
            self.writing = True
            self.backend = rep.BackendDispatch({"paths": {"out_dir": output_dir}})
        
        
        self.bin_sum.zero_()
        self.bin_count.zero_()
        self.binned_intensity.zero_()
        self.sonar_map.zero_()
        # Enable this feature so that warp will automatically free up GPU mamory 
        # if above certain total percentage usage
        if wp.is_mempool_supported:
            if not wp.is_mempool_enabled:
                wp.set_mempool_enabled()
            wp.set_mempool_release_threshold("cuda:0", 0.6)
        
        print(f'Sonar is initialized. (Writing: {self.writing})')

    def scan(self):
        self.scan_data['pcl'] = self.pointcloud_annot.get_data()['data']
        if self.scan_data['pcl'].shape[0] != 0:
            self.scan_data['normals'] = self.pointcloud_annot.get_data()['info']['pointNormals'][:,:3]
            self.scan_data['semantics'] = self.pointcloud_annot.get_data()['info']['pointSemantic']
            self.scan_data['viewTransform'] = self.cameraParams_annot.get_data()['cameraViewTransform'].reshape(4,4).T
            self.scan_data['idToLabels'] = self.semanticSeg_annot.get_data()['info']['idToLabels']


    def make_sonar_data(self, binning_method: str = "sum", normalizing_method: str = "all"):
        # A utility function helps to convert idToLabels into indexToProp array
        # This manipulation is needed for warp computation framework
        # indexToProp is an 1-dim array where the values associated with the query property 
        # are placed at the index corresponding to the key
        # First two entry are always zero for {'0': {'class': 'BACKGROUND'}, '1': {'class': 'UNLABELLED'}}
        # eg: indexToProp = [0, 0, 0.1, 1 .....] 
        def make_indexToProp_array(idToLabels: dict, query_property: str):
            max_id = max(idToLabels.keys(), default=-1)
            indexToProp_array = np.zeros((int(max_id)+1,))
            for id in idToLabels.keys():
                for property in idToLabels.get(id):
                    if property == query_property:
                        indexToProp_array[int(id)] = idToLabels.get(id).get(property)
            return indexToProp_array

        num_points = self.scan_data['pcl'].shape[0]
        if num_points != 0:
            pcl = wp.array(self.scan_data['pcl'], dtype=wp.vec3)
            normals=wp.array(self.scan_data['normals'], dtype=wp.vec3)
            viewTransform=wp.mat44(self.scan_data['viewTransform'])
            semantics = wp.array(self.scan_data['semantics'], wp.int8)
            indexToRefl = wp.array(make_indexToProp_array(idToLabels=self.scan_data['idToLabels'],
                                                         query_property='reflectivity'),
                                                         dtype=wp.float32)
        else:
            return

        # Intensity parameters
        attenuation = 0.1
        
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

        # Sonar map Noise Parameters
        gau_noise_param = 0.2
        ray_noise_param = 0.05
        intensity_offset = 0.0
        intensity_gain = 1.0
        # Calculate noise
        gau_noise = np.random.normal(loc=0, scale=gau_noise_param, size=self.bin_sum.shape)
        ray_noise = np.random.rayleigh(scale=ray_noise_param, size=self.bin_sum.shape)
        std = self.hori_fov/64
        range_dependent_ray_noise = self.r**2/self.max_range**2*(1 + np.exp(-(self.azi-np.pi/2)**2/std))*ray_noise 


        self.sonar_map.zero_()
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
                    maximum
                ]
            )
            maximum = maximum.numpy()[0]
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
            maximum = wp.zeros(shape=(self.r.shape[0],), dtype=wp.float32)
            wp.launch(
                dim=self.bin_sum.shape,
                kernel=range_max,
                inputs=[
                    self.binned_intensity,
                ],
                outputs=[
                    maximum
                ]
            )
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
        

        
        if self.writing:
            self.backend.schedule(write_np, f"intensity_{self.id}.npy", data=intensity)
            self.backend.schedule(write_np, f'pcl_local_{self.id}.npy', data=pcl_local)
            self.backend.schedule(write_np, f'sonar_data_{self.id}.npy', data=self.sonar_map)

            print(f"[{self.id}] Writing intensity, pcl_local, and sonar map")
        
        self.id += 1
    


    def close(self):
        self.pointcloud_annot.detach(self.rp)
        self.cameraParams_annot.detach(self.rp)

        self.pointcloud_annot = None
        self.cameraParams_annot = None
