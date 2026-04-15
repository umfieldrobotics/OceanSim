import warp as wp


@wp.func
def vec3_exp(exponent: wp.vec3):
    return wp.vec3(
        wp.exp(exponent[0]),
        wp.exp(exponent[1]),
        wp.exp(exponent[2]),
        dtype=type(exponent[0]),
    )


@wp.func
def vec3_mul(vec_1: wp.vec3, vec_2: wp.vec3):
    return wp.vec3(
        vec_1[0] * vec_2[0],
        vec_1[1] * vec_2[1],
        vec_1[2] * vec_2[2],
        dtype=type(vec_1[0]),
    )


@wp.kernel
def UW_render(
    raw_image: wp.array(ndim=3, dtype=wp.uint8),
    depth_image: wp.array(ndim=2, dtype=wp.float32),
    backscatter_value: wp.vec3,
    atten_coeff: wp.vec3,
    backscatter_coeff: wp.vec3,
    uw_image: wp.array(ndim=3, dtype=wp.uint8),
):
    i, j = wp.tid()
    raw_RGB = wp.vec3(
        wp.float32(raw_image[i, j, 0]),
        wp.float32(raw_image[i, j, 1]),
        wp.float32(raw_image[i, j, 2]),
        dtype=wp.float32,
    )
    depth = depth_image[i, j]
    exp_atten = vec3_exp(-depth * atten_coeff)
    exp_back = vec3_exp(-depth * backscatter_coeff)
    UW_RGB = vec3_mul(raw_RGB, exp_atten) + vec3_mul(
        backscatter_value * wp.float32(255), (wp.vec3f(1.0, 1.0, 1.0) - exp_back)
    )
    uw_image[i, j, 0] = wp.uint8(wp.clamp(UW_RGB[0], wp.float32(0), wp.float32(255)))
    uw_image[i, j, 1] = wp.uint8(wp.clamp(UW_RGB[1], wp.float32(0), wp.float32(255)))
    uw_image[i, j, 2] = wp.uint8(wp.clamp(UW_RGB[2], wp.float32(0), wp.float32(255)))
    uw_image[i, j, 3] = raw_image[i, j, 3]


@wp.kernel
def UW_depth_turbidity_attenuator(
    raw_image: wp.array(ndim=3, dtype=wp.uint8),
    depth_image: wp.array(ndim=2, dtype=wp.float32),
    max_range: wp.float32,
    backscatter_value: wp.vec3,
    atten_coeff: wp.vec3,
    backscatter_coeff: wp.vec3,
    sigma: wp.float32,
    seed: int,
    adjusted_depth: wp.array(ndim=2, dtype=wp.float32),
):
    i, j = wp.tid()
    raw_RGB = wp.vec3(
        wp.float32(raw_image[i, j, 0]),
        wp.float32(raw_image[i, j, 1]),
        wp.float32(raw_image[i, j, 2]),
        dtype=wp.float32,
    )
    depth = depth_image[i, j]

    # beer-lambert decay
    exp_atten = vec3_exp(-depth * atten_coeff)
    min_intens = wp.float32(0.25)
    rgb_intens = vec3_mul(raw_RGB, exp_atten)

    # average the RGB intensity per pixel
    av_rgb_intens = (rgb_intens[0] + rgb_intens[1] + rgb_intens[2]) / (
        wp.float32(3.0) * wp.float32(255.0)
    )

    # filter all depth points with too low of an intensity to be observable
    if av_rgb_intens <= min_intens:
        adjusted_depth[i, j] = wp.float32(0.0)
    else:
        state = wp.rand_init(seed, i * depth_image.shape[1] + j)
        sample = wp.randn(state)
        g_noise = sample * sigma
        adjusted_depth[i, j] = wp.max(depth + g_noise, wp.float32(0.0))
