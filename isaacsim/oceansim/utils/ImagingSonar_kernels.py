import warp as wp
from typing import Any

@wp.struct
class sonarGrid:
    x_offset: float
    y_offset: float
    x_res: float
    y_res: float
    x_num: int
    y_num: int


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
def bin_process(pcl: wp.array(dtype=wp.vec3),
                  intensity: wp.array(dtype=wp.float32),
                  semantics: wp.array(dtype=wp.uint32),
                  sonar_grid: sonarGrid,
                  bin_sum: wp.array(ndim=2, dtype=wp.float32),
                  bin_count: wp.array(ndim=2, dtype=wp.int32),
                  pcl_bin_idx: wp.array(dtype=wp.vec2ui),
                  bin_min_zenith: wp.array(ndim=2, dtype=wp.float32)
                  ):
    tid = wp.tid()

    # Get the range, azimuth of the point
    x = pcl[tid][0]
    y = pcl[tid][1]
    # Calculate the bin indices for range and azimuth
    x_bin_idx = wp.uint32((x - sonar_grid.x_offset) / sonar_grid.x_res)
    y_bin_idx = wp.uint32((y - sonar_grid.y_offset) / sonar_grid.y_res)
    wp.atomic_add(bin_sum, x_bin_idx, y_bin_idx, intensity[tid])
    wp.atomic_add(bin_count, x_bin_idx, y_bin_idx, 1)
    # Store the bin idx that corresponding to this pcl
    pcl_bin_idx[tid] = wp.vec2ui(x_bin_idx, y_bin_idx)
    # Store the minimum zenith value recorded for all the pcl 
    # that falls into this bin and is not background or unlabelled
    if semantics[tid] != 0 or 1:
        wp.atomic_min(bin_min_zenith, x_bin_idx, y_bin_idx, pcl[tid][2])




@wp.kernel
def bin_semantics_process(pcl: wp.array(dtype=wp.vec3),
                          semantics: wp.array(dtype=wp.uint32),
                          pcl_bin_idx: wp.array(dtype=wp.vec2ui),
                          bin_min_zenith: wp.array(ndim=2, dtype=wp.float32),
                          bin_semantics: wp.array(ndim=2, dtype=wp.uint32)
                          ):
    tid = wp.tid()

    # Get the zenith of this pcl
    z = pcl[tid][2]

    # Get the index of the bin in which this pcl falls in
    x_bin_idx = pcl_bin_idx[tid][0]
    y_bin_idx = pcl_bin_idx[tid][1]
    # This ensures the semantics of this cell only belongs to the pcl semantics with the smallest zenith value
    if (z < bin_min_zenith[x_bin_idx, y_bin_idx]) or (z == bin_min_zenith[x_bin_idx, y_bin_idx]):
        bin_semantics[x_bin_idx, y_bin_idx] = semantics[tid]




@wp.kernel
def bin_bbox_process(bbox_corners: wp.array(ndim=3, dtype=wp.float32),
                     sonar_grid: sonarGrid,
                     aligned_bbox_min: wp.array(ndim=2, dtype=wp.int32),
                     aligned_bbox_max: wp.array(ndim=2, dtype=wp.int32)
                    ):
    i, j = wp.tid()
    # Convert 8 corners local frame carteisan to local frame spherical
    bbox_corner_spher = cartesian_to_spherical(wp.vec3(bbox_corners[i,j,0],
                                                       bbox_corners[i,j,1],
                                                       bbox_corners[i,j,2]))
    # collapse 8 corners to the sonar grid
    x_bin_idx = wp.int32((bbox_corner_spher[0] - sonar_grid.x_offset) / sonar_grid.x_res)
    y_bin_idx = wp.int32((bbox_corner_spher[1] - sonar_grid.y_offset) / sonar_grid.y_res)

    x_bin_idx = wp.clamp(x_bin_idx, 0, sonar_grid.x_num-1)
    y_bin_idx = wp.clamp(y_bin_idx, 0, sonar_grid.y_num-1)
    # Compute an axis-aligned minimum-area bbox 
    # that contains all 8 corners of the 3d bbox
    # x_min
    wp.atomic_min(aligned_bbox_min, i, 0, x_bin_idx)
    # y_min
    wp.atomic_min(aligned_bbox_min, i, 1, y_bin_idx)
    # x_max
    wp.atomic_max(aligned_bbox_max, i, 0, x_bin_idx)
    # y_max
    wp.atomic_max(aligned_bbox_max, i, 1, y_bin_idx)


@wp.kernel
def draw_bbox(n : int,
              aligned_bbox_min: wp.array(ndim=2, dtype=wp.int32),
              aligned_bbox_max: wp.array(ndim=2, dtype=wp.int32),
              bbox_colors: wp.array(ndim=2, dtype=wp.uint8),
              image: wp.array(ndim=3, dtype=wp.uint8),
              ):
    # loop through the horizontal and vertical length, respectively
    i, j = wp.tid()
    width = image.shape[1]
    
    x_min = aligned_bbox_min[n,0]
    y_min = aligned_bbox_min[n,1]
    x_max = aligned_bbox_max[n,0]
    y_max = aligned_bbox_max[n,1]

    for c in range(4):
        image[x_min + i, width - y_min, c] = bbox_colors[n, c]
        image[x_min + i, width - y_max, c] = bbox_colors[n, c]
        image[x_min, width - (y_min + j), c] = bbox_colors[n, c]
        image[x_max, width - (y_min + j), c] = bbox_colors[n, c]


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
def normal_2d(seed: int,
              mean: float,
              std: float,
              output: wp.array(ndim=2, dtype=wp.float32),

):
    i, j = wp.tid()
    state = wp.rand_init(seed, i * output.shape[1] + j)  
    
    # Generate normal random variable
    output[i,j] = mean + std * wp.randn(state)



@wp.kernel
def range_dependent_rayleigh_2d(seed: int,
                                r: wp.array(ndim=2, dtype=wp.float32),
                                azi: wp.array(ndim=2, dtype=wp.float32),
                                max_range: float,
                                rayleigh_scale: float,
                                central_peak: float,
                                central_std: float,
                                output: wp.array(ndim=2, dtype = wp.float32)
):
    i, j = wp.tid()
    state = wp.rand_init(seed, i * output.shape[1] + j)
    
    # Generate two uniform random numbers
    n1 = wp.randn(state)
    n2 = wp.randn(state)  # Offset for independence
    
    # Transform to Rayleigh distribution
    rayleigh = rayleigh_scale * wp.sqrt(n1*n1 + n2*n2)
    # Apply range dependency
    output[i,j] = wp.pow(r[i,j]/max_range, 2.0) * (1.0 + central_peak * wp.exp(-wp.pow(azi[i,j] - wp.PI/2.0, 2.0) / central_std)) * rayleigh




@wp.kernel 
def make_sonar_map_all(r: wp.array(ndim=2, dtype=wp.float32),
                       azi: wp.array(ndim=2, dtype=wp.float32),
                       intensity: wp.array(ndim=2, dtype=wp.float32),
                       max_intensity: wp.array(ndim=1, dtype=wp.float32),
                       gau_noise: wp.array(ndim=2, dtype=wp.float32),
                       range_ray_noise: wp.array(ndim=2, dtype=wp.float32),
                       offset: wp.float32,
                       gain: wp.float32,
                       result: wp.array(ndim=2, dtype=wp.vec3)):
    i, j = wp.tid()
    intensity[i,j] = intensity[i,j]/max_intensity[0]
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


@wp.kernel
def make_semantics_image(bin_semantics: wp.array(ndim=2, dtype=wp.uint32),
                         semantics_color: wp.array(ndim=2, dtype=wp.uint8),
                         semantics_image: wp.array(ndim=3, dtype=wp.uint8),

                         ):
    i, j = wp.tid()
    width = bin_semantics.shape[1]
    semantics_image[i,width-j,0] = semantics_color[bin_semantics[i,j], 0]
    semantics_image[i,width-j,1] = semantics_color[bin_semantics[i,j], 1]
    semantics_image[i,width-j,2] = semantics_color[bin_semantics[i,j], 2]
    semantics_image[i,width-j,3] = semantics_color[bin_semantics[i,j], 3]

