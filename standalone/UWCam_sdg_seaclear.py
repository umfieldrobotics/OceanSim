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
            "--/log/outputStreamLevel=error",
            "--/renderer/multiGpu/enabled=false"            # Nvidia another freaking bug?
            ]
    },
    "total_captures" : 15,
    "camera_collider_radius": 0.1,
    "env_url": "/frog-drive/projects/OceanSim/sim2real/SDG_assets/sceneAssets/Collected_rocky_2x2/scene_2x2.usd",
    "objects_url": "/frog-drive/projects/OceanSim/sim2real/SDG_assets/ObjectAssets/ObjectAssets_detect_sea_urchin_seaclear/",
    "rt_subframes": 16,
    "resolution": [1024, 1024],
    "camera_properties_kwargs": {
        "focalLength": 24.0,
        "focusDistance": 400,
        "fStop": 0.0,
        "clippingRange": [0.001, 100],
    },
    "writers": [
        # random haze writer
        {
            "type": "UWCam_KittiWriter",
            "kwargs": {
                "output_dir": "/home/nsieh/Desktop/test_SDG/random_haze",
                "colorize_instance_segmentation": False,
                "veiling_visibility_threshold": 12, # This is not used now (writer code, can not easily remove)
                "use_tight_bbox": True,
                "debug_mode": False,
            },
        },
        # No haze writer
        {
            "type": "UWCam_KittiWriter",
            "kwargs": {
                "output_dir": "/home/nsieh/Desktop/test_SDG/no_haze",
                "colorize_instance_segmentation": False,
                "veiling_visibility_threshold": 12, # This is not used now (writer code, can not easily remove)
                "use_tight_bbox": True,
                "debug_mode": False,
                "UW_param": {
                    "scale_range": (1.0, 1.0),
                    "veiling": {
                        "no_haze": (0.0, 0.0, 0.0)
                    },
                    "backscatter": {
                        "no_haze": (0.0, 0.0, 0.0)
                    }
                }
            },
        },
        # Single haze writer
        {
            "type": "UWCam_KittiWriter",
            "kwargs": {
                "output_dir": "/home/nsieh/Desktop/test_SDG/single_haze",
                "colorize_instance_segmentation": False,
                "veiling_visibility_threshold": 12, # This is not used now (writer code, can not easily remove)
                "use_tight_bbox": True,
                "debug_mode": False,
                "UW_param": {   
                    "scale_range": (1.0, 1.0),
                    "veiling": {
                        "seaclear_sea_urchin": (0.08, 0.42, 0.52)                    },
                    "backscatter": {
                        "seaclear_sea_urchin": (1.0, 1.0, 1.0)
                    }
                }
            },
        }
    ],
    "add_distractors": False,
    "cam_workspace" : [-1.0, -1.0, 0.3, 1.0, 1.0, 1.3], # [minX, minY, minZ, maxX, maxY, maxZ] in the world frame
    "obj_workspace" : [-1.5, -1.5, -0.1, 1.5, 1.5, 1.0], # [minX, minY, minZ, maxX, maxY, maxZ] in the world frame
    "disable_render_products": False,
    "debug_mode": False,
    "seed": 984,
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
seed = config.get("seed", None)


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
    
    # Add objects
    objects, kitti_labels = add_objects(objects_folder_path=objects_url, 
                                        override_semantic_mapping=None, 
                                        physics=True,
                                        count=40,
                                        )
    print(f"[SDG] {len(objects)} numbers of COU objects being added to the scene")

    distractors = []
    if config.get("add_distractors", False):
        objects, kitti_labels = add_distractor(mapping=kitti_labels,
                                                root_path="SDG_distractors",
                                                name_prefix="distractor_",
                                                physics=True,
                                                num=10,
                                                count=1,
                                                )
        distractors.extend(objects)
        print(f"[SDG] {len(distractors)} numbers of distractor objects being added to the scene")

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

    terrian_mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/plane_0_collider"))
    points = terrian_mesh.GetPointsAttr().Get()
    points = [
        point
        for point in points
        if obj_ws[0] <= point[0] <= obj_ws[3]
        and obj_ws[1] <= point[1] <= obj_ws[4]
        and obj_ws[2] <= point[2] <= obj_ws[5]
    ]

    capture_counter = 0
    while capture_counter < total_captures:

        enable_global_volumetric_effects(enable=True, 
                                        density_mult=random.uniform(0.75, 1.25), 
                                        anisotropy_factor=-1.0, 
                                        transmittance_distance=10,
                                        )

        sample_objects_on_points(points, objects, offset=(0, 0, 0.05))

        if distractors:
            sample_objects_on_points(points, distractors, offset=(0, 0, 0.25))

        randomize_camera_poses_rel_to_ws(cameras, objects, cam_ws, look_at_offset=(-0.0, 0.0))

        perturb_object_poses(objects, translation_range=(-0.1, 0.1),scale_range=(0.5, 1.5))
        # Run simulation a bit for collider to settle
        run_simulation(num_frames=3, render=False)

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

if seed:
    np.random.seed(seed)
    random.seed(seed)
    rep.set_global_seed(seed)


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