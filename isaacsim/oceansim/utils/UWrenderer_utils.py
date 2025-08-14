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
             caustics: wp.array(ndim=3, dtype=wp.uint8),
             backscatter_value: wp.vec3,
             atten_coeff: wp.vec3,
             backscatter_coeff: wp.vec3,
             uw_image: wp.array(ndim=3, dtype=wp.uint8)):
    i,j = wp.tid()
    raw_RGB = wp.vec3(wp.float32(raw_image[i,j,0]), wp.float32(raw_image[i,j,1]), wp.float32(raw_image[i,j,2]), dtype=wp.float32)
    caustics_RGB = wp.vec3(wp.float32(caustics[i,j,0]), wp.float32(caustics[i,j,1]), wp.float32(caustics[i,j,2]), dtype=wp.float32)
    raw_RGB = raw_RGB + 0.7 * caustics_RGB
    depth = depth_image[i,j]
    exp_atten = vec3_exp(- depth * atten_coeff)
    exp_back = vec3_exp(- depth * backscatter_coeff)
    UW_RGB = vec3_mul(raw_RGB, exp_atten) + vec3_mul(backscatter_value * wp.float32(255), (wp.vec3f(1.0,1.0,1.0) - exp_back) )
    uw_image[i,j,0] = wp.uint8(wp.clamp(UW_RGB[0], wp.float32(0), wp.float32(255)))
    uw_image[i,j,1] = wp.uint8(wp.clamp(UW_RGB[1], wp.float32(0), wp.float32(255)))
    uw_image[i,j,2] = wp.uint8(wp.clamp(UW_RGB[2], wp.float32(0), wp.float32(255)))
    uw_image[i,j,3] = raw_image[i,j,3]

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


    
    
