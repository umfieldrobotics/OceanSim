import numpy as np
import warp as wp


#  Internal parameters
PROFILE_EXTENT = 410.0
PROFILE_RES = 8192
PROFILE_WAVENUM = 1000
MIN_WAVE_LENGTH = 0.1
MAX_WAVE_LENGTH = 250.0


#   Helpers
# ------------------------------------------------------------------------------


# fractional part of a (w.r.t. floor(a))
@wp.func
def frac(a: float):
    return a - wp.floor(a)


# square of a
@wp.func
def sqr(a: float):
    return a * a


@wp.func
def alpha_beta_spectrum(omega: float, peak_omega: float, alpha: float, beta: float, gravity: float):
    return (alpha * gravity * gravity / wp.pow(omega, 5.0)) * wp.exp(-beta * wp.pow(peak_omega / omega, 4.0))


@wp.func
def jonswap_peak_sharpening(omega: float, peak_omega: float, gamma: float):
    sigma = float(0.07)
    if omega > peak_omega:
        sigma = float(0.09)
    return wp.pow(gamma, wp.exp(-0.5 * sqr((omega - peak_omega) / (sigma * peak_omega))))


@wp.func
def jonswap_spectrum(omega: float, gravity: float, wind_speed: float, fetch_km: float, gamma: float):
    # https://wikiwaves.org/Ocean-Wave_Spectra#JONSWAP_Spectrum
    fetch = 1000.0 * fetch_km
    alpha = 0.076 * wp.pow(wind_speed * wind_speed / (gravity * fetch), 0.22)
    peak_omega = 22.0 * wp.pow(wp.abs(gravity * gravity / (wind_speed * fetch)), 1.0 / 3.0)
    return jonswap_peak_sharpening(omega, peak_omega, gamma) * alpha_beta_spectrum(
        omega, peak_omega, alpha, 1.25, gravity
    )


@wp.func
def TMA_spectrum(omega: float, gravity: float, wind_speed: float, fetch_km: float, gamma: float, water_depth: float):
    # https://dl.acm.org/doi/10.1145/2791261.2791267
    omegaH = omega * wp.sqrt(water_depth / gravity)
    omegaH = wp.max(0.0, wp.min(2.2, omegaH))
    phi = 0.5 * omegaH * omegaH
    if omegaH > 1.0:
        phi = 1.0 - 0.5 * sqr(2.0 - omegaH)
    return phi * jonswap_spectrum(omega, gravity, wind_speed, fetch_km, gamma)




#   Kernels
# ------------------------------------------------------------------------------

# warp kernel definitions
@wp.kernel
def update_profile(
    profile: wp.array(dtype=wp.vec3),
    profile_res: int,
    profile_data_num: int,
    min_lambda: float,
    max_lambda: float,
    profile_extend: float,
    time: float,
    wind_speed: float,
    water_depth: float,
):
    x = wp.tid()
    randState = wp.rand_init(7)
    # sampling parameters
    omega0 = wp.sqrt(wp.tau * 9.80665 / min_lambda)
    omega1 = wp.sqrt(wp.tau * 9.80665 / max_lambda)
    omega_delta = wp.abs(omega1 - omega0) / float(profile_data_num)
    # we blend three displacements for seamless spatial profile tiling
    space_pos_1 = profile_extend * float(x) / float(profile_res)
    space_pos_2 = space_pos_1 + profile_extend
    space_pos_3 = space_pos_1 - profile_extend
    p1 = wp.vec2(0.0, 0.0)
    p2 = wp.vec2(0.0, 0.0)
    p3 = wp.vec2(0.0, 0.0)
    for i in range(profile_data_num):
        omega = wp.abs(omega0 + (omega1 - omega0) * float(i) / float(profile_data_num))  # linear sampling of omega
        k = omega * omega / 9.80665
        phase = -time * omega + wp.randf(randState) * 2.0 * wp.pi
        amplitude = float(10000.0) * wp.sqrt(
            wp.abs(2.0 * omega_delta * TMA_spectrum(omega, 9.80665, wind_speed, 100.0, 3.3, water_depth))
        )
        p1 = wp.vec2(
            p1[0] + amplitude * wp.sin(phase + space_pos_1 * k), p1[1] - amplitude * wp.cos(phase + space_pos_1 * k)
        )
        p2 = wp.vec2(
            p2[0] + amplitude * wp.sin(phase + space_pos_2 * k), p2[1] - amplitude * wp.cos(phase + space_pos_2 * k)
        )
        p3 = wp.vec2(
            p3[0] + amplitude * wp.sin(phase + space_pos_3 * k), p3[1] - amplitude * wp.cos(phase + space_pos_3 * k)
        )
    # cubic blending coefficients
    s = float(float(x) / float(profile_res))
    c1 = float(2.0 * s * s * s - 3.0 * s * s + 1.0)
    c2 = float(-2.0 * s * s * s + 3.0 * s * s)
    disp_out = wp.vec3(
        (p1[0] + c1 * p2[0] + c2 * p3[0]) / float(profile_data_num),
        (p1[1] + c1 * p2[1] + c2 * p3[1]) / float(profile_data_num),
        0.0,
    )
    profile[x] = disp_out


@wp.kernel
def update_points(
    points: wp.array(dtype=wp.vec3),
    profile: wp.array(dtype=wp.vec3),
    profile_res: int,
    profile_extent: float,
    amplitude: float,
    directionality: float,
    direction: float,
    cam_pos: wp.vec3,
    clipmap_cell_size: float,
    out_points: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    p_crd = wp.vec3(
        points[tid][0] + wp.floor(cam_pos[0] / clipmap_cell_size) * clipmap_cell_size,
        points[tid][1],
        points[tid][2] + wp.floor(cam_pos[2] / clipmap_cell_size) * clipmap_cell_size,
    )

    randState = wp.rand_init(7)
    disp_x = float(0.0)
    disp_y = float(0.0)
    disp_z = float(0.0)
    w_sum = float(0.0)
    direction_count = 128
    for d in range(0, direction_count):
        r = float(d) * wp.tau / float(direction_count) + 0.02
        dir_x = wp.cos(r)
        dir_y = wp.sin(r)
        # directional amplitude
        t = wp.abs(direction - r)
        if t > wp.pi:
            t = wp.tau - t
        t = pow(t, 1.2)
        dir_amp = (2.0 * t * t * t - 3.0 * t * t + 1.0) * 1.0 + (-2.0 * t * t * t + 3.0 * t * t) * (
            1.0 - directionality
        )
        dir_amp = dir_amp / (1.0 + 10.0 * directionality)
        rand_phase = wp.randf(randState)
        x_crd = (p_crd[0] * dir_x + p_crd[2] * dir_y) / profile_extent + rand_phase
        pos_0 = int(wp.floor(x_crd * float(profile_res))) % profile_res
        if x_crd < 0.0:
            pos_0 = pos_0 + profile_res - 1
        pos_1 = int(pos_0 + 1) % profile_res
        p_disp_0 = profile[pos_0]
        p_disp_1 = profile[pos_1]
        w = frac(x_crd * float(profile_res))
        prof_height_x = dir_amp * float((1.0 - w) * p_disp_0[0] + w * p_disp_1[0])
        prof_height_y = dir_amp * float((1.0 - w) * p_disp_0[1] + w * p_disp_1[1])
        disp_x = disp_x + dir_x * prof_height_x
        disp_y = disp_y + prof_height_y
        disp_z = disp_z + dir_y * prof_height_x
        w_sum = w_sum + 1.0

    # write output vertex position
    out_points[tid] = wp.vec3(
        p_crd[0] + amplitude * disp_x / w_sum,
        p_crd[1] + amplitude * disp_y / w_sum,
        p_crd[2] + amplitude * disp_z / w_sum,
    )


def ocean_deform_launch_kernel(
        in_points : np.ndarray,
        time : float,
        amplitude : float = 1.0,
        cameraPos : np.ndarray = np.array([0.0, 0.0, 0.0]),
        clipmapCellSize : float = 1.0,
        direction : float = 0.0,
        directionality : float = 0.0,
        scale : float = 1.0,
        waterDepth : float = 50.0,
        windSpeed : float = 10.0         
        ):
    """
    Deforms a set of input points to simulate ocean surface displacement based on wave parameters.
    Args:
        in_points (np.ndarray): Input array of 3D points (shape: [N, 3]) to be deformed.
        time (float): Simulation time, used to animate the ocean surface.
        amplitude (float, optional): Wave amplitude scaling factor. Default is 1.0.
        cameraPos (np.ndarray, optional): 3D position of the camera, used for relative calculations. Default is np.array([0.0, 0.0, 0.0]).
        clipmapCellSize (float, optional): Size of the clipmap cell for spatial scaling. Default is 1.0.
        direction (float, optional): Main wave direction in radians. Default is 0.0.
        directionality (float, optional): Degree of wave directionality (0.0 to 1.0). Default is 0.0.
        scale (float, optional): Spatial scale of the ocean surface. Default is 1.0.
        waterDepth (float, optional): Depth of the water in meters. Default is 50.0.
        windSpeed (float, optional): Wind speed in meters per second, affecting wave generation. Default is 10.0.
    Returns:
        np.ndarray: Deformed array of 3D points representing the ocean surface at the given time.
    """
    

    profile = wp.zeros(PROFILE_RES, dtype=wp.vec3, device='cuda:0')
    in_points = wp.from_numpy(in_points, dtype=wp.vec3f, device='cuda:0')
    

    amplitude = max(0.0001, min(1000.0, amplitude))
    direction = direction % 6.28318530718
    directionality = max(0.0, min(1.0, 0.02 * directionality))
    wind_speed = max(0.0, min(30.0, windSpeed))
    water_depth = max(1.0, min(1000.0, waterDepth))
    scale = min(10000.0, max(0.001, scale))
    
    # create 1D profile buffer for this timestep using wave parameters

    wp.launch(
        kernel=update_profile,
        dim=(PROFILE_RES,),
        inputs=(
            profile,
            PROFILE_RES,
            PROFILE_WAVENUM,
            MIN_WAVE_LENGTH,
            MAX_WAVE_LENGTH,
            PROFILE_EXTENT,
            time,
            wind_speed,
            water_depth,
        ),
    )

    # Copy the in_points to create a out_points results holder.

    out_points = in_points

    # Update point positions using the profile buffer created above
    wp.launch(
        kernel=update_points,
        dim=len(in_points),
        inputs=(
            in_points,
            profile,
            PROFILE_RES,
            PROFILE_EXTENT * scale,
            amplitude,
            directionality,
            direction,
            cameraPos,
            clipmapCellSize,
        ),
        outputs=(out_points,),
    )

    return out_points