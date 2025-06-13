import argparse
import json
import os

import yaml
from isaacsim import SimulationApp
import carb


# Default config dict, can be updated/replaced using json/yaml config files ('--config' cli argument)
config = {
    "launch_config": {
        "renderer": "RaytracedLighting",
        "headless": True,
    },
    "env_url": "/home/haoyu/Desktop/onur_SDG/usd_Scene/ship.usd",
    "rt_subframes": 4,
    "num_frames": 100,
    "num_cameras": 1, # TODO  multi render product
    "simulation_duration_between_captures": 0.05,
    "resolution": (1920, 1080),
    "camera_properties_kwargs": {
        "focalLength": 24.0,
        "focusDistance": 400,
        "fStop": 0.0,
        "clippingRange": (0.01, 10000),
    },
    "writer_type": "BasicWriter",
    "writer_kwargs": {
        "output_dir": "/home/haoyu/Desktop/viz/",
        "rgb":True,
        'distance_to_image_plane': True,
        "camera_params": True,
        'colorize_depth': True,
        'normals': True
    },
}


# Check if there are any config files (yaml or json) are passed as arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=False, help="Include specific config parameters (json or yaml))")
args, unknown = parser.parse_known_args()
args_config = {}
if args.config and os.path.isfile(args.config):
    with open(args.config, "r") as f:
        if args.config.endswith(".json"):
            args_config = json.load(f)
        elif args.config.endswith(".yaml"):
            args_config = yaml.safe_load(f)
        else:
            carb.log_warn(f"File {args.config} is not json or yaml, will use default config")
else:
    carb.log_warn(f"File {args.config} does not exist, will use default config")

# Update the default config dict with the external one
config.update(args_config)

print(f"[SDG] Using config:\n{config}")

launch_config = config.get("launch_config", {})
simulation_app = SimulationApp(launch_config=launch_config)



import time
from itertools import chain
import carb.settings
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.storage.native import get_assets_root_path
import isaacsim.core.utils.prims as prims_utils
from pxr import PhysxSchema, Sdf, UsdGeom, UsdPhysics, Gf
import numpy as np


# Isaac nucleus assets root path
assets_root_path = get_assets_root_path()

# ENVIRONMENT
# Create an empty or load a custom stage (clearing any previous semantics)
env_url = config.get("env_url", "")
if env_url:
    omni.usd.get_context().open_stage(env_url)
    stage = omni.usd.get_context().get_stage()
else:
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    # Add a distant light to the empty stage
    distant_light = stage.DefinePrim("/World/Lights/DistantLight", "DistantLight")
    distant_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(400.0)
    if not distant_light.HasAttribute("xformOp:rotateXYZ"):
        UsdGeom.Xformable(distant_light).AddRotateXYZOp()
    distant_light.GetAttribute("xformOp:rotateXYZ").Set((0, 60, 0))

# Create a physics scene to modify custom physics settings
physics_scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/PhysicsScene"))
physx_scene.GetTimeStepsPerSecondAttr().Set(60)

# REPLICATOR
# Disable capturing every frame (capture will be triggered manually using the step function)
rep.orchestrator.set_capture_on_play(False)

# Create the camera prims and their properties
cam = rep.create.camera()
cam_prim = prims_utils.get_prim_at_path('/Replicator/Camera_Xform')
camera_properties_kwargs = config.get("camera_properties_kwargs", {})
for key, value in camera_properties_kwargs.items():
    if cam_prim.HasAttribute(key):
        cam_prim.GetAttribute(key).Set(value)
    else:
        print(f"Unknown camera attribute with {key}:{value}")



ship_prim = prims_utils.get_prim_at_path('/World/ship')
ship_loc = ship_prim.GetAttribute("xformOp:translate").Get()


# Wait an app update to ensure the prim changes are applied
simulation_app.update()

# Create render products using the cameras
resolution = config.get("resolution", (640, 480))
rp = rep.create.render_product(cam_prim.GetPath(), resolution)
# Create the writer and attach the render products
writer_type = config.get("writer_type", "BasicWriter")
writer_kwargs = config.get("writer_kwargs", {})
# If not an absolute path, set it relative to the current working directory
if out_dir := writer_kwargs.get("output_dir"):
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(os.getcwd(), out_dir)
        writer_kwargs["output_dir"] = out_dir
    print(f"[SDG] Writing data to: {out_dir}")

if writer_type is not None:
    writer = rep.writers.get(writer_type)
    writer.initialize(**writer_kwargs)
    writer.attach(rp)




def randomize_camera_poses(origin, radius):
    
    
    def get_random_pose_on_sphere(origin, radius, camera_forward_axis=(-1, 0, 0)):
        origin = Gf.Vec3f(origin)
        camera_forward_axis = Gf.Vec3f(camera_forward_axis)

        # Generate random angles for spherical coordinates
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.arcsin(np.random.uniform(-0.75, -0.25))

        # Spherical to Cartesian conversion
        x = radius * np.cos(theta) * np.cos(phi)
        y = radius * np.sin(phi)
        z = radius * np.sin(theta) * np.cos(phi)

        location = origin + Gf.Vec3f(x, y, z)

        # Calculate direction vector from camera to look_at point
        direction = origin - location
        direction_normalized = direction.GetNormalized()

        # Calculate rotation from forward direction (rotateFrom) to direction vector (rotateTo)
        rotation = Gf.Rotation(Gf.Vec3d(camera_forward_axis), Gf.Vec3d(direction_normalized))
        orientation = Gf.Quatf(rotation.GetQuat())

        return location, orientation
    offset = Gf.Vec3d(0 , 0, 0.5)
    origin = origin + offset
    new_loc, _ = get_random_pose_on_sphere(origin, radius)
    with cam:
        rep.modify.pose(
            position=new_loc,
            look_at = [*origin]
        )


def capture_pathtracing(duration=0.0, spp=128):

    # Set the render mode to PathTracing
    prev_render_mode = carb.settings.get_settings().get("/rtx/rendermode")
    carb.settings.get_settings().set("/rtx/rendermode", "PathTracing")
    carb.settings.get_settings().set("/rtx/pathtracing/spp", spp)
    carb.settings.get_settings().set("/rtx/pathtracing/totalSpp", spp)
    carb.settings.get_settings().set("/rtx/pathtracing/optixDenoiser/enabled", 0)

    # Make sure the timeline is playing
    if not timeline.is_playing():
        timeline.play()

    # Capture the frame by advancing the simulation for the given duration and combining the sub samples
    rep.orchestrator.step(delta_time=duration, pause_timeline=False)

    # Restore the previous render and motion blur  settings
    print(f"[SDG] Restoring render mode from 'PathTracing' to '{prev_render_mode}'")
    carb.settings.get_settings().set("/rtx/rendermode", prev_render_mode)


# Update the app until a given simulation duration has passed (simulate the world between captures)
def run_simulation_loop(duration):
    timeline = omni.timeline.get_timeline_interface()
    elapsed_time = 0.0
    previous_time = timeline.get_current_time()
    if not timeline.is_playing():
        timeline.play()
    app_updates_counter = 0
    while elapsed_time <= duration:
        simulation_app.update()
        elapsed_time += timeline.get_current_time() - previous_time
        previous_time = timeline.get_current_time()
        app_updates_counter += 1
        print(
            f"\t Simulation loop at {timeline.get_current_time():.2f}, current elapsed time: {elapsed_time:.2f}, counter: {app_updates_counter}"
        )
    print(
        f"[SDG] Simulation loop finished in {elapsed_time:.2f} seconds at {timeline.get_current_time():.2f} with {app_updates_counter} app updates."
    )


# SDG
# Number of frames to capture
num_frames = config.get("num_frames", 10)

# Increase subframes if materials are not loaded on time, or ghosting artifacts appear on moving objects,
# see: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/subframes_examples.html
rt_subframes = config.get("rt_subframes", -1) 
# Because we are only using path tracing mode, otherwise we need to call 
# rep.orchestrator.step(delta_time=duration, rt_subframes=st_subframes, pause_timeline=False) 
# to suppress ghosting


# Amount of simulation time to wait between captures
sim_duration_between_captures = config.get("simulation_duration_between_captures", 0.0)

for _ in range(5):
    simulation_app.update()

# Set the timeline parameters (start, end, no looping) and start the timeline
timeline = omni.timeline.get_timeline_interface()
timeline.set_start_time(0)
timeline.set_end_time(1000000)
timeline.set_looping(False)
# If no custom physx scene is created, a default one will be created by the physics engine once the timeline starts
timeline.play()
timeline.commit()
simulation_app.update()

# Store the wall start time for stats
wall_time_start = time.perf_counter()


# Run the simulation and capture data triggering randomizations and actions at custom frame intervals
for i in range(num_frames):
    # Cameras will be moved to a random position and look at a randomly selected labeled asset
    print(f"\t Randomizing camera poses")
    randomize_camera_poses(ship_loc, 8)
        

    # Capture the current frame
    print(f"[SDG] Capturing frame {i}/{num_frames}, at simulation time: {timeline.get_current_time():.2f}")
    capture_pathtracing()
    
    # Run the simulation for a given duration between frame captures
    if sim_duration_between_captures > 0:
        run_simulation_loop(duration=sim_duration_between_captures)
    else:
        simulation_app.update()

# Wait for the data to be written (default writer backends are asynchronous)
rep.orchestrator.wait_until_complete()

# Get the stats
wall_duration = time.perf_counter() - wall_time_start
sim_duration = timeline.get_current_time()
avg_frame_fps = num_frames / wall_duration

print(
    f"[SDG] Captured {num_frames} frames in {wall_duration:.2f} seconds.\n"
    f"\t Simulation duration: {sim_duration:.2f}\n"
    f"\t Simulation duration between captures: {sim_duration_between_captures:.2f}\n"
    f"\t Average frame FPS: {avg_frame_fps:.2f}\n"
)

# Unsubscribe the physics overlap checks and stop the timeline
simulation_app.update()
timeline.stop()

simulation_app.close()
