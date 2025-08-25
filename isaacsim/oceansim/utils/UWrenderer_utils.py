import warp as wp
import matplotlib.pyplot as plt


@wp.func
def vec3_exp(exponent: wp.vec3):
    return wp.vec3(wp.exp(exponent[0]), wp.exp(exponent[1]), wp.exp(exponent[2]), dtype=type(exponent[0]))

@wp.func
def vec3_mul(vec_1: wp.vec3,
            vec_2: wp.vec3):
    return wp.vec3(vec_1[0] * vec_2[0], vec_1[1] * vec_2[1], vec_1[2] * vec_2[2], dtype=type(vec_1[0]))

@wp.kernel
def UW_render(raw_image: wp.array(ndim=3, dtype=wp.uint8),
             depth_image: wp.array(ndim=2, dtype=wp.float32),
             backscatter_value: wp.vec3,
             atten_coeff: wp.vec3,
             backscatter_coeff: wp.vec3,
             uw_image: wp.array(ndim=3, dtype=wp.uint8)):
    """
    Notice: This kernel is deprecated for UW_render_2, which support caustics.
    Render the UW image.
    """
    i,j = wp.tid()
    raw_RGB = wp.vec3(wp.float32(raw_image[i,j,0]), wp.float32(raw_image[i,j,1]), wp.float32(raw_image[i,j,2]), dtype=wp.float32)
    depth = depth_image[i,j]
    exp_atten = vec3_exp(- depth * atten_coeff)
    exp_back = vec3_exp(- depth * backscatter_coeff)
    UW_RGB = vec3_mul(raw_RGB, exp_atten) + vec3_mul(backscatter_value * wp.float32(255), (wp.vec3f(1.0,1.0,1.0) - exp_back) )
    uw_image[i,j,0] = wp.uint8(wp.clamp(UW_RGB[0], wp.float32(0), wp.float32(255)))
    uw_image[i,j,1] = wp.uint8(wp.clamp(UW_RGB[1], wp.float32(0), wp.float32(255)))
    uw_image[i,j,2] = wp.uint8(wp.clamp(UW_RGB[2], wp.float32(0), wp.float32(255)))
    uw_image[i,j,3] = raw_image[i,j,3]


@wp.kernel
def UW_render_2(raw_image: wp.array(ndim=3, dtype=wp.uint8),
             depth_image: wp.array(ndim=2, dtype=wp.float32),
            #  caustics: wp.array(ndim=3, dtype=wp.uint8),
             backscatter_value: wp.vec3,
             atten_coeff: wp.vec3,
             backscatter_coeff: wp.vec3,
             uw_image: wp.array(ndim=3, dtype=wp.uint8)):
    i,j = wp.tid()
    raw_RGB = wp.vec3(wp.float32(raw_image[j,i,0]), wp.float32(raw_image[j,i,1]), wp.float32(raw_image[j,i,2]), dtype=wp.float32)
    # caustics_RGB = wp.vec3(wp.float32(caustics[j,i,0]), wp.float32(caustics[j,i,1]), wp.float32(caustics[j,i,2]), dtype=wp.float32)
    # raw_RGB = raw_RGB + 0.7 * caustics_RGB
    depth = depth_image[j,i]
    exp_atten = vec3_exp(- depth * atten_coeff)
    exp_back = vec3_exp(- depth * backscatter_coeff)
    UW_RGB = vec3_mul(raw_RGB, exp_atten) + vec3_mul(backscatter_value * wp.float32(255), (wp.vec3f(1.0,1.0,1.0) - exp_back) )
    uw_image[j,i,0] = wp.uint8(wp.clamp(UW_RGB[0], wp.float32(0), wp.float32(255)))
    uw_image[j,i,1] = wp.uint8(wp.clamp(UW_RGB[1], wp.float32(0), wp.float32(255)))
    uw_image[j,i,2] = wp.uint8(wp.clamp(UW_RGB[2], wp.float32(0), wp.float32(255)))
    uw_image[j,i,3] = raw_image[j,i,3]

@wp.func
def fract(x: float):
    return x - wp.floor(x)

@wp.func
def clamp01(x: float):
    return wp.min(1.0, wp.max(0.0, x))

@wp.func
def smoothstep(edge0: float, edge1: float, x: float):
    t = clamp01((x - edge0) / (edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)

@wp.func
def length2(v: wp.vec2f):
    return wp.sqrt(v[0]*v[0] + v[1]*v[1])

@wp.func
def normalize2(v: wp.vec2f):
    l = length2(v)
    return wp.vec2f(v[0]/(l+1e-6), v[1]/(l+1e-6))

@wp.func
def randomVal(inVal: float):
    d = inVal * 12.9898 + 2523.2361 * 78.233
    s = wp.sin(d)
    return fract(s * 43758.5453) - 0.5

@wp.func
def randomVec2(inVal: float):
    v = wp.vec2f(randomVal(inVal), randomVal(inVal + 151.523))
    return normalize2(v)

@wp.func
def makeWaves(uv: wp.vec2f, theTime: float, offset: float, timeSpeed: float):
    result = 0.0
    for n in range(16):
        i = float(n) + offset
        randVec = randomVec2(i)
        direction = uv[0] * randVec[0] + uv[1] * randVec[1]
        s = wp.sin(direction * randomVal(i + 1.6516) + theTime * timeSpeed)
        s = smoothstep(0.0, 1.0, s)
        result += randomVal(i + 123.0) * s
    return result

# NOTE: ndim=3 tells Warp this is a 3D array (H, W, 3)
@wp.kernel
def water_caustics(
    out_img: wp.array(ndim=3, dtype=wp.uint8),  # shape: (H, W, 3)
    width: int, height: int,
    time: float, timeSpeed: float,
):
    x, y = wp.tid()  # (W, H)

    # match your GLSL: uv normalized by width (iResolution.x)
    uv = wp.vec2f(float(x)/float(width), float(y)/float(width))
    uv2 = wp.vec2f(uv[0] * 150.0, uv[1] * 150.0)
    uv  = wp.vec2f(uv[0] * 2.0,   uv[1] * 2.0)

    r1 = makeWaves(wp.vec2f(uv2[0] + time*timeSpeed, uv2[1]),
                   time, 0.1, timeSpeed)
    r2 = makeWaves(wp.vec2f(uv2[0] - time*0.8*timeSpeed, uv2[1]),
                   time*0.8 + 0.06, 0.26, timeSpeed)

    r1 = smoothstep(0.4, 1.1, 1.0 - wp.abs(r1))
    r2 = smoothstep(0.4, 1.1, 1.0 - wp.abs(r2))
    val = 2.0 * smoothstep(0.35, 1.8, (r1 + r2) * 0.5)

    col = wp.uint8(wp.clamp(val * 0.7 * 255.0, 0.0, 255.0))

    # write to (H, W, 3) directly (row-major: y, x, channel)
    out_img[y, x, 0] = col
    out_img[y, x, 1] = col
    out_img[y, x, 2] = col
    out_img[y, x, 3] = wp.uint8(255)

@wp.kernel
def blend_caustics_PyTX(
    rgb_img: wp.array(ndim=3, dtype=wp.uint8),       # (H,W,3)
    depth_img: wp.array(ndim=2, dtype=wp.float32),        # (H,W)
    normals_img: wp.array(ndim=3, dtype=wp.float32),   # (H,W,3)
    caustics_img: wp.array(ndim=3, dtype=wp.uint8),     # (H,W) grayscale
    sun_dir: wp.vec3f,                         # normalized
    blend_weight: float,
    min_depth: float,
    max_depth: float,
    out_img: wp.array(ndim=3, dtype=wp.uint8)        # (H,W,3)
):
    x, y = wp.tid()  # 2D thread indices

    rgb = wp.vec3(wp.float32(rgb_img[y, x, 0]), wp.float32(rgb_img[y, x, 1]), wp.float32(rgb_img[y, x, 2]), dtype=wp.float32)
    depth = depth_img[y, x]
    nml = wp.vec3(wp.float32(normals_img[y, x, 0]), wp.float32(normals_img[y, x, 1]), wp.float32(normals_img[y, x, 2]), dtype=wp.float32)
    caustic_val = wp.vec3(wp.float32(caustics_img[y, x, 0]), wp.float32(caustics_img[y, x, 1]), wp.float32(caustics_img[y, x, 2]), dtype=wp.float32)

    # Lambert term
    dot_normals = wp.max(wp.dot(nml, sun_dir), 0.0)

    # Depth weighting
    norm_depth = (depth - min_depth) / (max_depth - min_depth + 1e-8)
    depth_weight = 1.0 - norm_depth

    # Final blend factor
    blend_factor = blend_weight * dot_normals * depth_weight

    blended = rgb * (1.0 - blend_factor) + caustic_val * blend_factor
    out_img[y, x, 0] = wp.uint8(wp.clamp(blended[0], 0.0, 255.0))
    out_img[y, x, 1] = wp.uint8(wp.clamp(blended[1], 0.0, 255.0))
    out_img[y, x, 2] = wp.uint8(wp.clamp(blended[2], 0.0, 255.0))
    out_img[y, x, 3] = wp.uint8(255)

@wp.kernel
def blend_caustics(
    rgb_aov: wp.array(ndim=3, dtype=wp.uint8),       # (H, W, 3 or 4) base color RGBA
    world_pos_aov: wp.array(ndim=3, dtype=wp.float32), # (H, W, 3) world positions
    world_nml_aov: wp.array(ndim=3, dtype=wp.float32), # (H, W, 3) world normals
    caustics_aov: wp.array(ndim=3, dtype=wp.uint8),  # (H, W, 3 or 4) caustics RGBA
    sun_dir: wp.vec3f,                               # light dir (world space)
    blend_weight: float,                             # base blend weight [0..1]
    uv_scale_x: float,                               # tiling scale for caustics U direction
    uv_scale_y: float,                               # tiling scale for caustics V direction
    depth_min: float,                                # min depth for normalization
    depth_max: float,                                # max depth for normalization
    tex_w: int,                                      # caustics width
    tex_h: int,                                      # caustics height
    out_aov: wp.array(ndim=3, dtype=wp.uint8)        # (H, W, 4) output RGBA
):
    x, y = wp.tid()  # 2D launch index
    


    # Read base color (handle both RGB and RGBA)
    base_col = wp.vec3(wp.float32(rgb_aov[y, x, 0]), wp.float32(rgb_aov[y, x, 1]), wp.float32(rgb_aov[y, x, 2]), dtype=wp.float32)

    # Read world pos & normal
    pos = wp.vec3(world_pos_aov[y, x, 0], world_pos_aov[y, x, 1], world_pos_aov[y, x, 2])
    nml = wp.normalize(wp.vec3(world_nml_aov[y, x, 0], world_nml_aov[y, x, 1], world_nml_aov[y, x, 2]))

    # Lambert shading factor (expects sun_dir to point from surface toward light)
    sun = wp.normalize(sun_dir)
    ndotl = wp.max(wp.dot(nml, sun), 0.0)

    # Depth-based weight
    denom = depth_max - depth_min + 1e-8
    norm_depth = wp.clamp((pos.z - depth_min) / denom, 0.0, 1.0)
    depth_weight = 1.0 - norm_depth

    # Blend factor
    blend_factor = wp.clamp(blend_weight * ndotl * depth_weight, 0.0, 1.0)

    # World-space planar UV projection (XZ plane) with separate scaling
    u = pos.x * uv_scale_x
    v = pos.z * uv_scale_y
    u = u - wp.floor(u)
    v = v - wp.floor(v)

    # Map to texture coordinates
    tx = wp.int32(wp.clamp(wp.floor(u * wp.float32(tex_w - 1)), 0.0, wp.float32(tex_w - 1)))
    ty = wp.int32(wp.clamp(wp.floor(v * wp.float32(tex_h - 1)), 0.0, wp.float32(tex_h - 1)))

    # Sample caustics texture (explicit channels)
    tex_r = wp.float32(caustics_aov[ty, tx, 0])
    tex_g = wp.float32(caustics_aov[ty, tx, 1])
    tex_b = wp.float32(caustics_aov[ty, tx, 2])
    tex_a = wp.float32(caustics_aov[ty, tx, 3]) / 255.0
    tex_rgb = wp.vec3(tex_r, tex_g, tex_b, dtype=wp.float32)

    # Scale caustics intensity by blend factor
    caustics_intensity = wp.clamp(tex_a * blend_factor, 0.0, 1.0)
    
    # Add caustics to base color (additive blending)
    out_rgb = base_col + tex_rgb * caustics_intensity

    out_aov[y, x, 0] = wp.uint8(wp.clamp(out_rgb[0], 0.0, 255.0))
    out_aov[y, x, 1] = wp.uint8(wp.clamp(out_rgb[1], 0.0, 255.0))
    out_aov[y, x, 2] = wp.uint8(wp.clamp(out_rgb[2], 0.0, 255.0))
    out_aov[y, x, 3] = wp.uint8(255)


@wp.func
def intrinsics_from_proj(P:wp.mat44f, width:int, height:int):
    fx = P[0,0] * wp.float32(width) / 2.0
    fy = P[1,1] * wp.float32(height) / 2.0
    cx = (1.0 - P[0,2]) * wp.float32(width) / 2.0
    cy = (1.0 + P[1,2]) * wp.float32(height) / 2.0
    return fx, fy, cx, cy


@wp.func
def intrinsics_from_proj(P:wp.mat44f, width:int, height:int):
    fx = P[0,0] * wp.float32(width) / 2.0
    fy = P[1,1] * wp.float32(height) / 2.0
    cx = (1.0 - P[0,2]) * wp.float32(width) / 2.0
    cy = (1.0 + P[1,2]) * wp.float32(height) / 2.0
    return fx, fy, cx, cy


@wp.func
def intrinsics_from_proj(P:wp.mat44f, width:int, height:int):
    fx = P[0,0] * wp.float32(width) / 2.0
    fy = P[1,1] * wp.float32(height) / 2.0
    cx = (1.0 - P[0,2]) * wp.float32(width) / 2.0
    cy = (1.0 + P[1,2]) * wp.float32(height) / 2.0
    return fx, fy, cx, cy

@wp.kernel
def depth_to_world_pos(
    depth: wp.array(ndim=2, dtype=wp.float32),           # (H, W) depth buffer in world units
    proj_matrix: wp.mat44,     # (4, 4) projection matrix
    view_matrix: wp.mat44,     # (4, 4) camera->world matrix
    H: int,
    W: int,
    world_points: wp.array(ndim=3, dtype=wp.float32),

):
    x, y = wp.tid()

    d = depth[y, x]

    # Check validity using mathematical operations (avoid branching)
    is_valid = wp.float32(d > 0.0 and wp.isfinite(d))
    
    # Since depth is in world units, we can use it directly to reconstruct world position
    # Extract camera parameters from projection matrix
    fx, fy, cx, cy = intrinsics_from_proj(proj_matrix, W, H)

    # Pixel → ray direction in camera space
    cam_x = (wp.float32(x) - cx) / fx
    cam_y = -(wp.float32(y) - cy) / fy  # flip Y if needed
    cam_z = -1.0  # forward in OpenGL convention

    ray_vec = wp.normalize(wp.vec3(cam_x, cam_y, cam_z))

    # Scale ray by depth (world units)
    cam_pos = ray_vec * d

    # Camera → world
    world_pos_h = view_matrix @ wp.vec4(cam_pos[0], cam_pos[1], cam_pos[2], 1.0)

    # Store world-space position (zero out if invalid depth)
    world_points[y, x, 0] = world_pos_h[0] * is_valid
    world_points[y, x, 1] = world_pos_h[1] * is_valid
    world_points[y, x, 2] = world_pos_h[2] * is_valid




if __name__ == "__main__":
    wp.init()
    width, height = 1024, 512
    out_img = wp.zeros(shape=(height, width, 3), dtype=wp.uint8)

    # Animation params
    fps = 30
    seconds = 5
    time_speed = 2.0
    num_frames = fps * seconds

    # Setup interactive display
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Water Caustics")
    ax.axis('off')
    im_artist = ax.imshow(out_img.numpy())

    for frame_idx in range(num_frames):
        t = float(frame_idx) / float(fps)
        wp.launch(
            water_caustics,
            dim=(width, height),
            inputs=[out_img, width, height, t, time_speed],
        )
        wp.synchronize()
        im_artist.set_data(out_img.numpy())
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(1.0 / fps)

    plt.ioff()
    plt.show()


    
    
