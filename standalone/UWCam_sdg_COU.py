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
            "--/renderer/multiGpu/enabled=false"            # Nvidia another freaking bug? Will crash on multi-gpu
            ]
    },
    "total_captures" : 30,
    "camera_collider_radius": 0.1,
    "env_url": "/mnt/frog-users/projects/OceanSim/sim2real/SDG_assets/sceneAssets/terrains_3x3",
    "objects_url": "/mnt/frog-users/projects/OceanSim/sim2real/SDG_assets/ObjectAssets/ObjectAssets_COU_replica",
    "distractors_folder": "/mnt/frog-users/projects/OceanSim/sim2real/SDG_assets/ObjectAssets/OceanRealm_assets/",
    "rt_subframes": 16,
    "resolution": [1920, 1080],
    "camera_properties_kwargs": {
        "focalLength": 32,
        "focusDistance": 400,
        "fStop": 0.0,
        "clippingRange": [0.001, 100],
    },
    "writers": [
        {
            "type": "UWCam_KittiWriter",
            "kwargs": {
                "output_dir": "/mnt/frog-users/projects/OceanSim/sim2real/training_data/temp",
                "colorize_instance_segmentation": False,
                "truncation_dropout_threshold": 0.8,
                "bbox2d_partly_occluded_threshold": 0.7,
                "use_tight_bbox": True,
                "debug_mode": False,
                "UW_param": {
                    "scale_range": (0.5, 0.5), # 0.25 for now is on an ablalation study that gives the greatest mAP 
                    "veiling": {
                            # "deep_sea": (0.0, 0.0, 0.28),
                            # "shallow_water": (0.05, 0.11, 0.7),
                            "akdeniz": (0.14, 0.3, 0.5),
                            "river": (0.294, 0.4, 0.263),
                            "mud": (0.259, 0.259, 0.024),
                            "mhl": (0.0, 0.3021, 0.239),
                            "murky": (0.275, 0.212, 0.071),
                            # "seaclear_sea_urchin": (0.08, 0.42, 0.52),
                        },
                    "backscatter": {
                            "Type I": (0.905, 0.961, 0.982),
                            "Type IA": (0.804, 0.954, 0.975),
                            "Type IB": (0.830, 0.940, 0.968),
                            "Type II": (0.800, 0.925, 0.940),
                            "Type III": (0.750, 0.885, 0.890),
                            "Type 1": (0.750, 0.885, 0.875),
                            "Type 3": (0.710, 0.820, 0.800),
                            "Type 5": (0.670, 0.730, 0.670),
                            "Type 7": (0.620, 0.610, 0.590),
                            "Type 9": (0.550, 0.460, 0.290),

                    },
                    "attenuation": {
                            "Type I": (1.3575, 1.4415, 1.473),
                            "Type IA": (1.206, 1.431, 1.4625),
                            "Type IB": (1.245, 1.41, 1.452),
                            "Type II": (1.2, 1.3875, 1.41),
                            # "Type III": (0.750*1.5, 0.885*1.5, 0.890*1.5),
                            # "Type 1": (0.750*1.5, 0.885*1.5, 0.875*1.5),
                            # "Type 3": (0.710*1.5, 0.820*1.5, 0.800*1.5),
                            # "Type 5": (0.670*1.5, 0.730*1.5, 0.670*1.5),
                            # "Type 7": (0.620*1.5, 0.610*1.5, 0.590*1.5),
                            # "Type 9": (0.550*1.5, 0.460*1.5, 0.290*1.5),
                    }
                }
            },
        }

    ],
    # If specified, this dict will override the default mapping from object names to kitti labels. 
    # Format: {"object_name_keyword": kitti_label_id, ...}
    "override_semantic_mapping": {
        "Scissors" : 0,
        "Fork" : 1,
        "Cup" : 2,
        "Spoon" : 3,
        "Screwdriver" : 4,
        "Car" : 5,
        "ROV" : 6,
        "Knife" : 7,
    }, 
    "add_distractors": True,
    "cam_workspace" : [-0.25, -0.25, 0.75, 0.25, 0.25, 1.25], # [minX, minY, minZ, maxX, maxY, maxZ] in the world frame
    "cam_lookat_workspace" : [-1.5, -1.5, -1.0, 1.5, 1.5, 1.0], # [minX, minY, minZ, maxX, maxY, maxZ] in the world frame
    "obj_workspace" : [-2.0, -2.0, -1.0, 2.0, 2.0, 1.0], # [minX, minY, minZ, maxX, maxY, maxZ] in the world frame
    "dist_workspace" : [-2.5, -2.5, -1.0, 2.5, 2.5, 1.0], # [minX, minY, minZ, maxX, maxY, maxZ] in the world frame
    "randomize_object_color": True,
    "color_bias_range": (-0.25, 0.25),
    "color_scale_range": (0.8, 1.2),
    "disable_render_products": False,
    "debug_mode": False,
    "seed": 114514,
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
    dist_ws = config.get("dist_workspace")
    cam_ws = config.get("cam_workspace")
    randomize_object_color = config.get("randomize_object_color", False)
    color_bias_range = config.get("color_bias_range", ())
    color_scale_range = config.get("color_scale_range", ())
    lookat_ws = config.get("cam_lookat_workspace")
    camera_properties_kwargs = config.get("camera_properties_kwargs", {})
    path_tracing = config.get("path_tracing", False)
    num_cameras = config.get("num_cameras", 1)
    domelight_intensity = config.get("domelight_intensity", 1500.0)
    resolution = tuple(config.get("resolution", [640, 480]))
    override_semantic_mapping = config.get("override_semantic_mapping", None)
    disable_render_products = config.get("disable_render_products", False)

    # ENVIRONMENT
    parsed_envs = parse_env_folder(env_url)

    # This is an another freaking bug that Isaac Sim has to open a stage with MDL displacement on first,
    # so that the stages loaded after can have displacement effective
    omni.usd.get_context().open_stage(list(parsed_envs.values())[0])
    run_simulation(num_frames=100, render=True)
    print('heated up the renderer')
    omni.usd.get_context().new_stage()

    num_envs = len(parsed_envs)
    envs_iter = iter(parsed_envs.values())
    add_reference_to_stage(usd_path=next(envs_iter), prim_path='/terrain')

    stage = omni.usd.get_context().get_stage()
    domelight = create_dome_ligth(stage, "/Environment", intensity=domelight_intensity)




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
                                        override_semantic_mapping=override_semantic_mapping, 
                                        physics=True,
                                        count=2,
                                        )
    print(f"[SDG] {len(objects)} numbers of detection objects being added to the scene")

    distractors = []
    distract_folder = config.get('distractors_folder', None)
    if config.get("add_distractors", False):
        # update the kitti labels mapping dict with distractor objects
        ds, kitti_labels = add_distractor_from_UE(mapping=kitti_labels,
                                                UE_asset_folder=distract_folder,
                                                root_path="SDG_distractors",
                                                name_prefix="distractor_",
                                                physics=True,
                                                num=30,
                                                count=2,
                                                )
        distractors.extend(ds)

        print(f"[SDG] {len(distractors)} numbers of distractor objects being added to the scene")

    # Resolve any centimeter-meter scale issues of the assets
    resolve_scale_issues_with_metrics_assembler()
    
    # Get objects UVtexture shader handles
    objects_uv_texture_shaders = []
    if (color_bias_range or color_scale_range) and randomize_object_color and objects:
        objects_material_prims = get_material_prims(stage.GetPrimAtPath("/SDG_objects"))
        objects_uv_texture_shaders = list(chain.from_iterable([get_UsdUVTexture_shaders(prim) for prim in objects_material_prims]))


    # Get distractors UVtexture shader handles
    distractors_uv_texture_shaders = []
    if (color_bias_range or color_scale_range) and distractors:
        distractors_material_prims = get_material_prims(stage.GetPrimAtPath("/SDG_distractors"))
        distractors_uv_texture_shaders = list(chain.from_iterable([get_UsdUVTexture_shaders(prim) for prim in distractors_material_prims]))
    
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

    obj_ws_points = extract_points_from_mesh(stage.GetPrimAtPath("/terrain/collider"), obj_ws)
    dist_ws_points = extract_points_from_mesh(stage.GetPrimAtPath("/terrain/collider"), dist_ws)

    capture_counter = 0

    env_switch_interval = max(1, total_captures // num_envs)
    env_index = 0
    print(f"[SDG] env_switch_interval: {env_switch_interval}, num_envs: {num_envs}")


    # Enable FFT bloom (This makes the underwater scene look a bit more blurry, a bit more realistic)
    enable_FFT_bloom(enable=False, energyConstrainingBlend=True)
    enable_global_volumetric_effects(enable=True, 
                                    density_mult=1.0, 
                                    anisotropy_factor=-1.0, 
                                    transmittance_distance=40,
                                    )
    # Set the background type to Black
    # set_background_type(background_type="Color", color=(0.0, 0.0, 0.0))
    # carb.settings.get_settings().set("/rtx/background/source/color", (0, 0, 0))
    # carb.settings.get_settings().set("/rtx/background/source/type", "Color")
    # carb.settings.get_settings().set("/rtx/background/source/color", (0, 0, 0))
    while capture_counter < total_captures:

        if capture_counter % env_switch_interval == 0 and capture_counter > 0:
            try:
                next_env = next(envs_iter)
                stage.RemovePrim("/terrain")
                add_reference_to_stage(usd_path=next_env, prim_path='/terrain')
                print(f"[SDG] Switching environment from [{env_index}] {list(parsed_envs.keys())[env_index]} to [{(env_index + 1)}] {list(parsed_envs.keys())[env_index + 1]}")
                env_index += 1
            except:
                print(f"[SDG] Environment exhausted, reuse the last environment.")


            # Recompute the sampled points on the new terrain 
            obj_ws_points = extract_points_from_mesh(stage.GetPrimAtPath("/terrain/collider"), obj_ws)
            dist_ws_points = extract_points_from_mesh(stage.GetPrimAtPath("/terrain/collider"), dist_ws)
            for _ in range(100):
                simulation_app.update()

        # we put objects a bit higher than the terrain
        sample_objects_on_points(obj_ws_points, objects, offset=(0, 0, 0.2))
        if distractors:
            sample_objects_on_points(dist_ws_points, distractors)
        if objects_uv_texture_shaders:
            randomize_UVTexture_scale_bias(objects_uv_texture_shaders, 
                                        scale_range=color_scale_range,
                                        bias_range=color_bias_range)
        if distractors_uv_texture_shaders:
            randomize_UVTexture_scale_bias(distractors_uv_texture_shaders, 
                                       scale_range=color_scale_range,
                                       bias_range=color_bias_range)

        randomize_camera_poses_rel_to_objs(cameras, objects, lookat_ws, cam_ws, look_at_offset=(-0.1, 0.1))

        perturb_object_poses(objects, scale_range=(0.75, 1.25))

        # Run simulation a bit for collider to settle
        run_simulation(num_frames=10, render=False)

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
        # save_object_info(objects, cameras, os.path.join(out_dir, "metadata", f"object_info_{capture_counter}.json"))
        
        
        
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