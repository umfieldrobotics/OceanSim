# Default config dict, can be updated/replaced using json/yaml config files ('--config' cli argument)
config = {
    "launch_config": {
        "renderer": "RealTimePathTracing",
        "headless": True,
        "extra_args": [
            "--/persistent/renderer/rtpt/enabled=True",              # This enables RTX realtime preview renderer
            "--/log/level=error",                                    # These will shut isaac sim the fuck up 
            "--/log/fileLogLevel=error", 
            "--/log/outputStreamLevel=error"
            ]
    },
    "total_captures" : 10,
    "rt_subframes": 4,
    # NOTE: Because each sonar writer requires a camera with different FOV settings, the total camera generated in the scene is num_cameras * num_writers
    # NOTE: For this reason, the total number of captures is num_cameras * total_captures * num_writers
    "num_cameras": 1, 
    "camera_collider_radius": 0.2,
    "env_url": "/frog-drive/ocean-sim/sim2real/sceneAssets/duluth/Collected_pebble_floor/padded_pebble_floor.usd",
    # "env_url": "D:/haoyu/Assets/sceneAssets/duluth/Collected_pebble_floor/padded_pebble_floor_water.usd",
    "objects_url": "/frog-drive/ocean-sim/sim2real/ObjectAssets_simready",
    # "objects_url": "D:/haoyu/Assets/ObjectAssets_simready",
    "masked_objects_ratio": 0.92,
    "writers": [
        {
            # Type of the writer to use (e.g. PoseWriter, BasicWriter, etc.) and the kwargs to pass to the writer init
            "type": "FLS_KittiWriter",
            "kwargs": {
                # "output_dir": "D:/haoyu/SDG/",
                "output_dir": "/home/haoyu/Desktop/viz/",
                "sonar_param": {
                    "max_range": 3, 
                    "min_range": 0.2,
                    "hori_fov": 130, # Notice: on camera end, hori_fov and vert_fov is required to 
                    "vert_fov": 20,  # compute camera AR and vert_res given arbitrary hori_res
                    "range_res": 0.005, 
                    "angular_res": 0.25,
                    "normalizing_method": "range",
                    "query_prop": "reflectivity", 
                    "attenuation": 0.1,
                    "gau_noise_param": 0.2,
                    "ray_noise_param": 0.05,
                    "intensity_offset": 0.0,
                    "intensity_gain": 1.0,
                    "central_peak": 2,
                    "central_std": 0.001,
                    "hori_res": 5000
                    },
                "reflectivity_mapping" : {
                        "UNLABELLED": 0,
                        "BACKGROUND": 0,
                        "scissors": 7.0,
                        "plastic_cup": 7.0,
                        "metal_rod": 7.0,
                        "fork": 7.0,
                        "bottle": 7.0,
                        "soda_can": 7.0,
                        "case": 7.0,
                        "plastic_bag": 7.0,
                        "cup": 7.0,
                        "goggles": 7.0,
                        "flipper": 7.0,
                        "loco": 7.0,
                        "aqua": 7.0,
                        "pipe": 7.0,
                        "snorkel": 7.0,
                        "spoon": 7.0,
                        "lure": 7.0,
                        "screwdriver": 7.0,
                        "car": 7.0,
                        "tripod": 7.0,
                        "rov": 7.0,
                        "knife": 7.0,
                        "dive_weight": 7.0,
                        "ground": 1.0,
                },
                "debug_mode": True,
            },
        }
    ],
    "obj_workspace": [-1.5, -2.3, 1.15, 1.5, 3.7, 1.65],
    "cam_workspace" : [-1.5, -2.3, 1.25, 1.5, 3.7, 1.5],
    "disable_render_products": False,
    "debug_mode": True,
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
from isaacsim.oceansim.utils.FLS_sdg_utils import *


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
    rt_subframes = config.get("rt_subframes", 16)
    obj_ws = config.get("obj_workspace")
    cam_ws = config.get("cam_workspace")
    masked_objects_ratio = config.get("masked_objects_ratio", 0.5)
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





    # Add COU objects
    objects, kitti_labels = add_COU_objects(objects_folder_path=objects_url, physics=True, reflectivity=3.0)
    print(f"[SDG] {len(objects)} numbers of COU objects being added to the scene")

    # Resolve any centimeter-meter scale issues of the assets
    resolve_scale_issues_with_metrics_assembler()


    writer_configs = config.get("writers", [{}])
    num_cameras = config.get("num_cameras", 0)
    cameras = []
    render_products = []
    writers = []
    camera_colliders = []
    camera_collider_radius = config.get("camera_collider_radius", 0)
    
    if writer_configs[0]:
        for i, writer_config in enumerate(writer_configs):
            sonar_param = writer_config.get("kwargs", {}).get("sonar_param", {})
            tmp_rp_list = []
            for k in range(num_cameras):
                cam_prim, resolution = create_sonar_compatible_camera(sonar_param, f"/Sonars/sonar_{i}_{k}")

                rp = rep.create.render_product(cam_prim.GetPath(), resolution, name=f"rp_sonar_{i}_{k}")
                tmp_rp_list.append(rp)
                

                if camera_collider_radius > 0:
                    cam_collider = stage.DefinePrim(f"/Sonars/sonar_{i}_{k}/CollisionSphere_{i}_{k}", "Sphere")
                    cam_collider.GetAttribute("radius").Set(camera_collider_radius)
                    add_colliders(cam_collider)
                    UsdGeom.Imageable(cam_collider).MakeInvisible()

                cameras.append(cam_prim)
                render_products.append(rp)
                camera_colliders.append(cam_collider)

            writer_type = writer_config.get("type", None)
            writer = rep.writers.get(writer_type)
            writer_kwargs = writer_config.get("kwargs", {})
            if out_dir := writer_kwargs.get("output_dir"):
                # If not an absolute path, make path relative to the current working directory
                if not os.path.isabs(out_dir):
                    out_dir = os.path.join(os.getcwd(), out_dir)
                    writer_kwargs["output_dir"] = out_dir
            if writer:
                writer.initialize(**writer_kwargs, mapping_dict=kitti_labels)
                writer.attach(tmp_rp_list)
            writers.append(writer)

        print(f"[SDG] {len(writers)} writers initialized with {num_cameras} cameras, colliders: {camera_collider_radius > 0}")

        




    capture_counter = 0
    while capture_counter < total_captures:


        run_simulation(num_frames=1, render=False)

        # Randomize the poses of the objects
        randomize_poses(objects, location_range=obj_ws, rotation_range=(0, 360), scale_range=(0.75, 1.25))
        
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
        run_simulation(num_frames=3, render=False)


        # Check if the render products need to be enabled for the capture
        if disable_render_products:
            for rp in render_products:
                rp.hydra_texture.set_updates_enabled(True)
    
        # This captures the sonar data and trigger the writers
        rep.orchestrator.step(rt_subframes=rt_subframes, delta_time=0.0, pause_timeline=True)

        
        
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



register_FLS_KittiWriter()

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