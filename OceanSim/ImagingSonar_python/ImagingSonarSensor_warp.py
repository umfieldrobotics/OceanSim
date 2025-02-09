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
                    reflectivity: float,
                    attenuation: float,
                    intensity: wp.array(dtype=wp.float32)
                    ):
    tid = wp.tid()
    R = wp.mat33(viewTransform[0,0], viewTransform[0,1], viewTransform[0,2],
                 viewTransform[1,0], viewTransform[1,1], viewTransform[1,2],
                 viewTransform[2,0], viewTransform[2,1], viewTransform[2,2])
    T = wp.vec3(viewTransform[0,3], viewTransform[1,3], viewTransform[2,3])
    sensor_loc = - (wp.transpose(R) @ T)
    # Will use warp.math.norm_l2() in future release
    incidence = pcl[tid] - sensor_loc
    dist = wp.sqrt(incidence[0]*incidence[0] + incidence[1]*incidence[1] + incidence[2]*incidence[2])
    unit_directs = wp.normalize(pcl[tid] - sensor_loc)
    cos_theta = wp.dot(unit_directs, normals[tid])
    intensity[tid] = reflectivity * wp.abs(cos_theta) * wp.exp(-attenuation * dist)

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
    else:
        avg[i,j] = 0.0



# Simply convert cartesian to polar for easy plotting
@wp.kernel 
def process_sonar_data(r: wp.array(ndim=2, dtype=wp.float32),
                       azi: wp.array(ndim=2, dtype=wp.float32),
                       intensity: wp.array(ndim=2, dtype=wp.float32),
                       max_intensity:wp.float32,
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

# TODO #
# Implement 'pointSemantic': dnarray[...],  #  shape=(<num_points>), dtype=uint8), from PointCloud annotator 
# to take into account of material property
class ImagingSonarSensor:

    def __init__(self, prim_path : str, 
                 trans : list[float]= [0.0, 0.0, 0.0], 
                 orients: list[float] = rotations_utils.euler_angles_to_quat(np.array([0,0,0]))
                 ):
        
        # Raw parameters from Oculus M370s\MT370s\MD370s
        self.max_range = 5 # m (max is 200 m in datasheet )
        self.min_range = 0.2 # m (min is 0.2 m in datasheet)
        self.range_res = 0.01 # m (datasheet is 0.008 m)
        self.update_rate = 40 # Hz (max update rate) (NOT USED FOR NOW)!!
        self.hori_fov = 90 # degree (hori_fov is 130 degrees in datasheet)
        self.vert_fov = 90 # degree (vert_fov is 20 degrees in datasheet)
        self.num_beams = 256 # (max number of beams) (NOT USED FOR NOW)!!
        self.angular_res = 0.1 # degree (datasheet is 2 deg)
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
        self.mean_intensity = wp.zeros(shape=self.r.shape, dtype=wp.float32)


        # We introduce this factor to adjust raycast density
        # (Equivalently, adjust the beam_separation) 
        # Increase this value by 1 will quadraple the total number of raycasts
        self.ray_factor = 10 # below 15 is advised for me

        self.AR = self.hori_fov / self.vert_fov
        self.hori_res = int(self.ray_factor * (self.hori_fov / self.beam_separation))
        self.vert_res = int(self.hori_res / self.AR)
        # By doing this, I am assuming the vertical beam separation
        # is the same as the beam vertical separation. 
        # This is bacause replicator raytracing is specified as resolutions
        # while non-squre pixel is not supported in Isaac sim. See details below.

        print(f'resolution: {self.hori_res} x {self.vert_res}')
        self.camera_prim_path = prim_path + '/Camera'
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
    # For now, data is generated per simulation tick
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
        # do_array_copy: If True, retrieve a copy of the data array. 
        # This is recommended for workflows using asynchronous
        # backends to manage the data lifetime. 
        # Can be set to False to gain performance if the data is 
        # expected to be used immediately within the writer. Defaults to True.

        self.pointcloud_annot.attach(self.rp)
        self.cameraParams_annot.attach(self.rp)

        if output_dir is not None:
            self.writing = True
            self.backend = rep.BackendDispatch({"paths": {"out_dir": output_dir}})
        
        
        self.bin_sum.zero_()
        self.bin_count.zero_()
        self.mean_intensity.zero_()

        # Enable this feature so that warp will automatically free up GPU mamory 
        # if above certain total percentage usage
        if wp.is_mempool_supported:
            if not wp.is_mempool_enabled:
                wp.set_mempool_enabled()
            wp.set_mempool_release_threshold("cuda:0", 0.6)
        

    def scan(self):
        self.scan_data['pcl'] = self.pointcloud_annot.get_data()['data']
        self.scan_data['normals'] = self.pointcloud_annot.get_data()['info']['pointNormals']
        self.scan_data['viewTransform'] = self.cameraParams_annot.get_data()['cameraViewTransform'].reshape(4,4).T



    def make_sonar_data(self):

        reflectivity = 1
        attenuation = 0.3


        if self.scan_data['pcl'].size != 0:
            pcl = wp.array(self.scan_data['pcl'], dtype=wp.vec3)
            normals=wp.array(self.scan_data['normals'][:,:3], dtype=wp.vec3)
            viewTransform=wp.mat44(self.scan_data['viewTransform'])
            num_points = pcl.shape[0]
        else:
            return
        
        intensity = wp.empty(shape=(num_points,), dtype=wp.float32)

        wp.launch(kernel=compute_intensity,
                  dim=num_points,
                  inputs=[
                      pcl,
                      normals,
                      viewTransform,
                      reflectivity,
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
        self.mean_intensity.zero_()

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
        
        wp.launch(
            kernel=average,
            dim=self.bin_sum.shape,
            inputs=[
                self.bin_sum,
                self.bin_count
            ],
            outputs=[
                self.mean_intensity,
            ]
            )
        # wp.max() has bug (in future will avoid moving this back to host)
        self.max_intensity = np.max(self.mean_intensity.numpy())
        self.backend.schedule(write_np, f"intensity_{self.id}.npy", data=intensity)
        self.backend.schedule(write_np, f'pcl_local_{self.id}.npy', data=pcl_local)
        print(f"[{self.id}] Writing data")
        self.id += 1


        self.process_sonar_data()
    
    def process_sonar_data(self):

        gau_noise_param = 0.2
        ray_noise_param = 0.15
        intensity_offset = 0.2
        intensity_gain = 1.0
        
        # Calculate noise
        gau_noise = np.random.normal(loc=0, scale=gau_noise_param, size=self.mean_intensity.shape)
        ray_noise = np.random.rayleigh(scale=ray_noise_param, size=self.mean_intensity.shape)
        range_dependent_ray_noise = self.r**2/self.max_range**2*(1 + np.exp(-np.abs(self.azi)))*ray_noise 

        sonar_map = wp.empty(shape=self.mean_intensity.shape, dtype=wp.vec3)
        wp.launch(kernel=process_sonar_data,
                  dim=sonar_map.shape,
                  inputs=[
                      wp.array(self.r, ndim=2, dtype=wp.float32),
                      wp.array(self.azi, ndim=2, dtype=wp.float32),
                      self.mean_intensity,
                      self.max_intensity,
                      wp.array(gau_noise, ndim=2, dtype=wp.float32),
                      wp.array(range_dependent_ray_noise, ndim=2, dtype=wp.float32),
                      intensity_offset,
                      intensity_gain
                  ],
                  outputs=[
                      sonar_map
                  ]
                  )


        self.backend.schedule(write_np, f'sonar_data_{self.id}.npy', data=sonar_map)




    def close(self):
        self.pointcloud_annot.detach(self.rp)
        self.cameraParams_annot.detach(self.rp)

        self.pointcloud_annot = None
        self.cameraParams_annot = None
