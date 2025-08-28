# Default config dict, can be updated/replaced using json/yaml config files ('--config' cli argument)
config = {
    "launch_config": {
        "renderer": "RealTimePathTracing",
        "headless": True,
        "extra_args": [
            # "--/persistent/renderer/rtpt/enabled=True",             # This enables RTX 2.0 for Isaac 4.5
            "--/persistent/rtx/modes/rt2/enabled=True",              # This enables RTX 2.0 for Isaac 5.0
            "--/persistent/rtx/modes/pt/enabled=True",              # This enables Path Tracing for Isaac 5.0
            "--/persistent/rtx/modes/rt/enabled=True",              # This enables Ray Tracing for Isaac 5.0
            "--/log/level=error",                                    # These will shut isaac sim the fuck up 
            "--/log/fileLogLevel=error", 
            "--/log/outputStreamLevel=error"
            ]
    },
    "total_captures" : 4000,
    "camera_collider_radius": 0.1,
    # "env_url": "/frog-drive/ocean-sim/sim2real/sceneAssets/duluth/Collected_pebble_floor/padded_pebble_floor_water.usd",
    "env_url": "C:/Users/mahaoyu/Desktop/pebble_floor/padded_pebble_floor_water.usd",
    # "objects_url": "/frog-drive/ocean-sim/sim2real/ObjectAssets_simready",
    "objects_url": "C:/Users/mahaoyu/Desktop/ObjectAssets_simready",
    "rt_subframes": 16,
    "resolution": [1024, 1024],
    "masked_objects_ratio": 0.96,
    "camera_properties_kwargs": {
        "focalLength": 50.0,
        "focusDistance": 400,
        "fStop": 0.0,
        "clippingRange": [0.01, 100],
    },
    "writers": [
        {
            # Type of the writer to use (e.g. PoseWriter, BasicWriter, etc.) and the kwargs to pass to the writer init
            "type": "UWCam_KittiWriter",
            "kwargs": {
                "output_dir": "C:/Users/mahaoyu/Desktop/SDG/",
                # "output_dir": "/home/haoyu/Desktop/viz/",
                "colorize_instance_segmentation": False,
                "veiling_visibility_threshold": 12,
                "use_tight_bbox": True,
                # "UW_param": "/frog-drive/ocean-sim/sim2real/sceneAssets/duluth/duluth.yaml",
                "debug_mode": False,
                "enable_caustics": True,
            },
        }
    ],
    "obj_workspace": [-1.5, -2.3, 1.15, 1.5, 3.7, 1.65],
    "cam_workspace" : [-1.5, -2.3, 1.25, 1.5, 3.7, 1.5],
    "disable_render_products": True,
    "debug_mode": False,
    "path_tracing": False,
}



###################################################
#### Here goes arg parser and Isaac Sim config ####
###################################################
import argparse
import json
import os
import sys
import yaml

# Check if there are any config files (yaml or json) are passed as arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=False, help="Include specific config parameters (json or yaml))")
parser.add_argument("--close_on_completion", action="store_true", help="Close the app on completion")
args, unknown = parser.parse_known_args()
args_config = {}
if args.config and os.path.isfile(args.config):
    with open(args.config, "r") as f:
        if args.config.endswith(".json"):
            args_config = json.load(f)
        elif args.config.endswith(".yaml"):
            args_config = yaml.safe_load(f)
        else:
            print(f"[SDG] File {args.config} is not json or yaml, will use default config")
else:
    print(f"[SDG] File {args.config} does not exist, will use default config")

# Update the default config dict with the external one
config.update(args_config)

print(f"[SDG] Using config:\n{config}")

launch_config = config.get("launch_config", {})
debug_mode = config.get("debug_mode", False)
if debug_mode:
    launch_config["headless"] = False

from isaacsim.simulation_app import SimulationApp

simulation_app = SimulationApp(launch_config=launch_config)

# load up OceanSim
import isaacsim.core.utils.extensions as extensions_utils
value = extensions_utils.enable_extension(extension_name='isaacsim.oceansim')
if value:
    print("[SDG] OceanSim loaded successfully")
else:
    simulation_app.update()
    simulation_app.close()
    sys.exit("[SDG] OceanSim loaded failed. SDG Stopped...")

# Load an environment extension that some usd scenes will rely on

extensions_utils.enable_extension(extension_name="omni.kit.actions.core")
extensions_utils.enable_extension(extension_name="omni.kit.window.preferences")
extensions_utils.enable_extension(extension_name="omni.kit.widget.sliderbar")
extensions_utils.enable_extension(extension_name="omni.kit.viewport.utility")
extensions_utils.enable_extension(extension_name="omni.kit.usd.layers")
extensions_utils.enable_extension(extension_name="omni.rtx.window.settings")
extensions_utils.enable_extension(extension_name="omni.kit.notification_manager")
extensions_utils.enable_extension(extension_name="omni.kit.window.filepicker")
extensions_utils.enable_extension(extension_name="omni.kit.environment.core")
extensions_utils.enable_extension(extension_name="omni.kit.property.environment")
extensions_utils.enable_extension(extension_name="omni.kit.window.environment")


import omni.replicator.core as rep
from pxr import PhysxSchema, Sdf, UsdGeom, UsdPhysics, Gf
import numpy as np
import random
from isaacsim.oceansim.utils.UWCam_sdg_utils import *


#########################################################
#### Here goes scene construction and randomnization ####
#########################################################


def run_sdg(config: dict=config):
    

    # Increase maximum assets loading time in case assets are too many
    carb.settings.get_settings().set('/exts/omni.replicator.core/maxAssetLoadingTime', 1000)
    # Disable UJITSO cooking ([Warning] [omni.ujitso] UJITSO : Build storage validation failed)
    carb.settings.get_settings().set("/physics/cooking/ujitsoCollisionCooking", False)
    # Disable capture on play
    rep.orchestrator.set_capture_on_play(False)




    env_url = config.get("env_url", "")
    objects_url = config.get("objects_url", "")
    total_captures = config.get("total_captures", 10)
    rt_subframes = config.get("rt_subframes", 4) 
    obj_ws = config.get("obj_workspace")
    cam_ws = config.get("cam_workspace")
    camera_properties_kwargs = config.get("camera_properties_kwargs", {})
    masked_objects_ratio = config.get("masked_objects_ratio", 0.5)
    path_tracing = config.get("path_tracing", False)
    num_cameras = config.get("num_cameras", 1)
    resolution = tuple(config.get("resolution", [640, 480]))

    disable_render_products = config.get("disable_render_products", False)

    # ENVIRONMENT
    # Create an empty or load a custom stage (clearing any previous semantics)
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




    # Create the cameras
    cameras = []
    stage.DefinePrim("/Cameras", "Scope")
    for i in range(num_cameras):
        cam_prim = stage.DefinePrim(f"/Cameras/cam_{i}", "Camera")
        for key, value in camera_properties_kwargs.items():
            if cam_prim.HasAttribute(key):
                if key == "clippingRange":
                    cam_prim.GetAttribute(key).Set(tuple(value))
                else:
                    cam_prim.GetAttribute(key).Set(value)
            else:
                print(f"Unknown camera attribute with {key}:{value}")
        cameras.append(cam_prim)
    print(f"[SDG] Created {len(cameras)} cameras")

    # Create the render products for the cameras
    render_products = []
    disable_render_products = config.get("disable_render_products", False)
    for cam in cameras:
        rp = rep.create.render_product(cam.GetPath(), resolution, name=f"rp_{cam.GetName()}")
        if disable_render_products:
            rp.hydra_texture.set_updates_enabled(False)
        render_products.append(rp)
    print(f"[SDG] Created {len(render_products)} render products")

    # Add collision spheres (disabled by default) to cameras to avoid objects overlaping with the camera view
    camera_colliders = []
    camera_collider_radius = config.get("camera_collider_radius", 0)
    if camera_collider_radius > 0:
        for i, cam in enumerate(cameras):
            cam_collider = stage.DefinePrim(f"/Cameras/cam_{i}/CollisionSphere_{i}", "Sphere")
            cam_collider.GetAttribute("radius").Set(camera_collider_radius)
            add_colliders(cam_collider)
            UsdGeom.Imageable(cam_collider).MakeInvisible()
            camera_colliders.append(cam_collider)
    
        print(f"[SDG] Created camera colliders with radius {camera_collider_radius}")
    
    # Add COU objects
    objects, kitti_labels = add_COU_objects(objects_folder_path=objects_url, physics=True)
    print(f"[SDG] {len(objects)} numbers of COU objects being added to the scene")

    # Resolve any centimeter-meter scale issues of the assets
    resolve_scale_issues_with_metrics_assembler()
    
    
    # Only create the writers if there are render products to attach to
    writers = []
    if render_products:
        for writer_config in config.get("writers", []):
            writer_type = writer_config.get("type", None)
            writer = None
            if writer_type is None:
                print("[SDG] No writer type specified. No writer will be used.")

            try:
                writer = rep.writers.get(writer_type)
            except Exception as e:
                print(f"[SDG] Writer type '{writer_type}' not found. No writer will be used. Error: {e}")

            writer_kwargs = writer_config.get("kwargs", {})
            if out_dir := writer_kwargs.get("output_dir"):
                # If not an absolute path, make path relative to the current working directory
                if not os.path.isabs(out_dir):
                    out_dir = os.path.join(os.getcwd(), out_dir)
                    writer_kwargs["output_dir"] = out_dir
            
            
            if writer:
                writer.initialize(**writer_kwargs, mapping_dict=kitti_labels)
                writer.attach(render_products)
                writers.append(writer)
                print(f"\t {writer_config['type']}'s out dir: {writer_config.get('kwargs', {}).get('output_dir', '')}")

    print(f"[SDG] Created {len(writers)} writers")





    capture_counter = 0
    while capture_counter < total_captures:

        enable_global_volumetric_effects(enable=True, 
                                        density_mult=random.uniform(1.75, 1.95), 
                                        anisotropy_factor=-0.999, 
                                        transmittance_distance=random.uniform(3000, 10000),
                                        )

        # Randomize the poses of the objects
        randomize_poses(objects, location_range=obj_ws, rotation_range=(0, 360), scale_range=(0.75, 1.25))
        
        # Run simulation a bit for objects to fall
        run_simulation(num_frames=random.randint(3, 30), render=False)

        # Mask a random number of objects
        if masked_objects_ratio == 1:
            for obj in objects:
                obj.GetAttribute("visibility").Set("invisible")
            
            randomize_camera_poses_rel_to_ws(cameras, objects, cam_ws, look_at_offset=(-0.0, 0.0))
        else:
            masked_objects = mask_random_objects(objects, ratio=masked_objects_ratio)
            visible_objects = [obj for obj in objects if obj not in masked_objects]
            
            
            # Randomize the poses of the cameras
            randomize_camera_poses_rel_to_ws(cameras, visible_objects, cam_ws, look_at_offset=(-0.0, 0.0))
        
        # Run simulation a bit for collider to settle
        run_simulation(num_frames=2, render=False)

        # Check if the render products need to be enabled for the capture
        if disable_render_products:
            for rp in render_products:
                rp.hydra_texture.set_updates_enabled(True)
    
        if path_tracing:
            capture_pathtracing(delta_time=0.0, spp=512, pause_timeline=True)
        else:
            rep.orchestrator.step(rt_subframes=rt_subframes, delta_time=0.0)

        
        
        # NOTE: Temporary code to save the object info as metadata
        if not os.path.exists(os.path.join(out_dir, "metadata")):
            os.makedirs(os.path.join(out_dir, "metadata"), exist_ok=True)
        save_object_info(objects, cameras, os.path.join(out_dir, "metadata", f"object_info_{capture_counter}.json"))
        
        
        
        # Check if the render products need to be disabled until the next capture
        if disable_render_products:
            for rp in render_products:
                rp.hydra_texture.set_updates_enabled(False)


        unmask_objects(objects)

        capture_counter += 1
        print(f"[SDG] Captured {capture_counter}/{total_captures} frames")


    # Wait until the data is written to the disk
    rep.orchestrator.wait_until_complete()

    # Detach the writers
    print(f"[SDG] Detaching writers")
    for writer in writers:
        writer.detach()

    # Destroy render products
    print(f"[SDG] Destroying render products")
    for rp in render_products:
        rp.destroy()

    print(f"[SDG] SDG Finished, captured {capture_counter * num_cameras} frames..")



register_UWCam_KittiWriter()

if debug_mode:
    np.random.seed(10)
    random.seed(10)
    rep.set_global_seed(10)


# Start the SDG pipeline
print(f"[SDG] Starting the SDG pipeline.")
run_sdg(config)
print(f"[SDG] SDG pipeline finished.")

# Make sure the app closes on completion even if in debug mode
if args.close_on_completion:
    simulation_app.close()

# In debug mode, keep the app running until manually closed
if debug_mode:
    while simulation_app.is_running():
        simulation_app.update()

simulation_app.close()