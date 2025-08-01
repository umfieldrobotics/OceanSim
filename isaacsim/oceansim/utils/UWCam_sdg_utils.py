
# The warehouse distractors which will be added to the scene and randomized
# THese mesh are defualt and used to test the code
DEFAULT_OBJECTS = 3 * [
    "/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/S_WetFloorSign.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_03.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_B_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_B_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_B_03.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_C_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticA_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticB_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticA_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticA_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticD_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticE_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BucketPlastic_B.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_1262.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_1268.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_1482.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_1683.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_291.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_01_1454.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_01_1513.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_A_04.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_B_03.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_B_05.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_C_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_E_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_PushcartA_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_RackPile_04.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_RackPile_03.usd",
]
DEFAULT_LABELS = {
    "UNLABELLED": (0, 0, 0, 0),
    "BACKGROUND": (0, 0, 0, 0),
    "wall": (0, 0, 0, 0),
    "pillar": (0, 0, 0, 0),
    "floor": (0, 0, 0, 0),
    "floor_decal": (0, 0, 0, 0),
    "klt_bin": (111, 74, 0, 255),
    "palletjack": (81, 0, 81, 255),
    "cone": (128, 64, 128, 255),
    "sign": (244, 35, 232, 255),
    "barel": (250, 170, 160, 255),
    "bottle": (230, 150, 140, 255),
    "box": (70, 70, 70, 255),
    "crate": (102, 102, 156, 255),
    "cart": (190, 153, 153, 255),
    "rack": (180, 165, 180, 255),
    "sign": (150, 100, 100, 255),
    "bucket": (150, 120, 90, 255),
    "wire": (153, 153, 153, 255),
    "pallet": (153, 153, 153, 255),
    "fire_extinguisher": (250, 170, 30, 255),
    "fuse_box": (220, 220, 0, 255),

}


# Predefined colors for better visual distinction
# Using distinct colors that are visually different from each other
COLOR_PALETTE = [
        (255, 0, 0, 255),      # Red
        (0, 255, 0, 255),      # Green
        (0, 0, 255, 255),      # Blue
        (255, 255, 0, 255),    # Yellow
        (255, 0, 255, 255),    # Magenta
        (0, 255, 255, 255),    # Cyan
        (128, 0, 0, 255),      # Dark Red
        (0, 128, 0, 255),      # Dark Green
        (0, 0, 128, 255),      # Dark Blue
        (128, 128, 0, 255),    # Olive
        (128, 0, 128, 255),    # Purple
        (0, 128, 128, 255),    # Teal
        (255, 128, 0, 255),    # Orange
        (128, 255, 0, 255),    # Lime
        (0, 255, 128, 255),    # Spring Green
        (128, 0, 255, 255),    # Violet
        (255, 0, 128, 255),    # Pink
        (0, 128, 255, 255),    # Sky Blue
        (255, 128, 128, 255),  # Light Red
        (128, 255, 128, 255),  # Light Green
        (128, 128, 255, 255),  # Light Blue
        (255, 255, 128, 255),  # Light Yellow
        (255, 128, 255, 255),  # Light Magenta
        (128, 255, 255, 255),  # Light Cyan
        (192, 192, 192, 255),  # Silver
        (128, 64, 0, 255),     # Brown
        (64, 128, 0, 255),     # Dark Green
        (0, 64, 128, 255),     # Dark Blue
        (128, 0, 64, 255),     # Dark Purple
        (64, 0, 128, 255),     # Dark Violet
        (0, 128, 64, 255),     # Dark Teal
        (128, 64, 128, 255),   # Medium Purple
        (64, 128, 128, 255),   # Medium Teal
        (128, 128, 64, 255),   # Medium Olive
    ]

COU_OBJECTS_FOLDER_PATH = "/frog-drive/ocean-sim/sim2real/ObjectAssets_simready/"

import math
import os
import random
import re
from itertools import chain
from collections import defaultdict
import json

import omni.kit.app
import omni.kit.commands
import omni.physx
import omni.replicator.core as rep
import omni.timeline
import omni.usd
import carb.settings
from isaacsim.core.utils.semantics import add_update_semantics, remove_all_semantics
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.transformations import get_relative_transform

def find_usd_files_recursively(root_folder):
    """
    Recursively find all files with .usd extension in the given folder and return their absolute paths.

    Args:
        root_folder (str): The root directory to search.

    Returns:
        List[str]: List of absolute paths to .usd files found.
    """
    usd_files = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith('.usd'):
                abs_path = os.path.abspath(os.path.join(dirpath, filename))
                usd_files.append(abs_path)
    return usd_files

def parse_object_folder(objects_folder_path):
    """
    Parses the object folder and returns a dictionary of subfolders and their corresponding paths.
    """
    category_dict = {}

    categories = os.listdir(objects_folder_path)
    print(f"[SDG] Found {len(categories)} categories: {categories} in {objects_folder_path}")
    for category in categories:
        category_folder_path = os.path.join(objects_folder_path, category)
        usd_files = find_usd_files_recursively(category_folder_path)
        category_dict[category] = usd_files
    return category_dict



def add_COU_objects(
                objects_folder_path=COU_OBJECTS_FOLDER_PATH, 
                root_path="SDG_objects",
                name_prefix="", 
                collider=False,
                ) -> tuple[list[Usd.Prim], dict[str, tuple[int, int, int, int]]]:
    stage = omni.usd.get_context().get_stage()
    stage.DefinePrim(f"/{root_path}", "Scope")
    categories = parse_object_folder(objects_folder_path)
    assets = []
    for category, usd_files in categories.items():
        for usd_file in usd_files:
            
            prim_path = omni.usd.get_stage_next_free_path(stage, f"/{root_path}/{name_prefix}{category}", False)

            prim = add_reference_to_stage(usd_path=usd_file, prim_path=prim_path)
            if collider:
                add_colliders(prim, approximation_type="boundingCube")
            remove_all_semantics(prim, recursive=True)
            add_update_semantics(prim, category)
            assets.append(prim)
    return assets, generate_kitti_labels(categories)
    

# needed for loading textures correctly
def set_transform_attributes(
    prim: Usd.Prim,
    location: Gf.Vec3d | None = None,
    orientation: Gf.Quatf | None = None,
    rotation: Gf.Vec3f | None = None,
    scale: Gf.Vec3f | None = None,
) -> None:
    """Set transformation attributes (location, orientation, rotation, scale) on a prim."""
    if location is not None:
        if not prim.HasAttribute("xformOp:translate"):
            UsdGeom.Xformable(prim).AddTranslateOp()
        prim.GetAttribute("xformOp:translate").Set(location)
    if orientation is not None:
        if not prim.HasAttribute("xformOp:orient"):
            UsdGeom.Xformable(prim).AddOrientOp()
        prim.GetAttribute("xformOp:orient").Set(orientation)
    if rotation is not None:
        if not prim.HasAttribute("xformOp:rotateXYZ"):
            UsdGeom.Xformable(prim).AddRotateXYZOp()
        prim.GetAttribute("xformOp:rotateXYZ").Set(rotation)
    if scale is not None:
        if not prim.HasAttribute("xformOp:scale"):
            UsdGeom.Xformable(prim).AddScaleOp()
        prim.GetAttribute("xformOp:scale").Set(scale)


def add_colliders(root_prim: Usd.Prim, approximation_type: str = "convexHull") -> None:
    """Add collision attributes to mesh and geometry primitives under the root prim."""
    for desc_prim in Usd.PrimRange(root_prim):
        if desc_prim.IsA(UsdGeom.Gprim):
            if not desc_prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI.Apply(desc_prim)
            else:
                collision_api = UsdPhysics.CollisionAPI(desc_prim)
            collision_api.CreateCollisionEnabledAttr(True)

        if desc_prim.IsA(UsdGeom.Mesh):
            if not desc_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(desc_prim)
            else:
                mesh_collision_api = UsdPhysics.MeshCollisionAPI(desc_prim)
            mesh_collision_api.CreateApproximationAttr().Set(approximation_type)


def has_colliders(root_prim: Usd.Prim) -> bool:
    """Check if any descendant prims under the root prim have collision attributes."""
    for desc_prim in Usd.PrimRange(root_prim):
        if desc_prim.HasAPI(UsdPhysics.CollisionAPI):
            return True
    return False


def add_rigid_body_dynamics(prim: Usd.Prim, disable_gravity: bool = False) -> None:
    """Add rigid body dynamics properties to a prim if it has colliders, with optional gravity setting."""
    if has_colliders(prim):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        else:
            rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
        rigid_body_api.CreateRigidBodyEnabledAttr(True)

        # Apply PhysX rigid body dynamics
        if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        else:
            physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI(prim)
        physx_rigid_body_api.GetDisableGravityAttr().Set(disable_gravity)
    else:
        print(
            f"[SDG-Infinigen] Prim '{prim.GetPath()}' has no colliders. Skipping adding rigid body dynamics properties."
        )


def add_colliders_and_rigid_body_dynamics(prim: Usd.Prim, disable_gravity: bool = False) -> None:
    """Add colliders and rigid body dynamics properties to a prim, with optional gravity setting."""
    add_colliders(prim)
    add_rigid_body_dynamics(prim, disable_gravity)

# Createa  collision box area wrapping the given working area with origin in (0, 0, 0) with thickness towards outside
def create_collision_box_walls(stage, path, width, depth, height, thickness=0.5, visible=False):
    # Define the walls (name, location, size) with thickness towards outside of the working area
    walls = [
        ("floor", (0, 0, (height + thickness) / -2.0), (width, depth, thickness)),
        ("ceiling", (0, 0, (height + thickness) / 2.0), (width, depth, thickness)),
        ("left_wall", ((width + thickness) / -2.0, 0, 0), (thickness, depth, height)),
        ("right_wall", ((width + thickness) / 2.0, 0, 0), (thickness, depth, height)),
        ("front_wall", (0, (depth + thickness) / 2.0, 0), (width, thickness, height)),
        ("back_wall", (0, (depth + thickness) / -2.0, 0), (width, thickness, height)),
    ]
    for name, location, size in walls:
        prim = stage.DefinePrim(f"{path}/{name}", "Cube")
        scale = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
        set_transform_attributes(prim, location=location, scale=scale)
        add_colliders(prim)
        if not visible:
            UsdGeom.Imageable(prim).MakeInvisible()

def find_matching_prims(
    match_strings: list[str], root_path: str | None = None, prim_type: str | None = None, first_match_only: bool = False
) -> Usd.Prim | list[Usd.Prim] | None:
    """Find prims matching specified strings, with optional type filtering and single match return."""
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPseudoRoot() if root_path is None else stage.GetPrimAtPath(root_path)

    matching_prims = []
    for prim in Usd.PrimRange(root_prim):
        if any(match in str(prim.GetPath()) for match in match_strings):
            if prim_type is None or prim.GetTypeName() == prim_type:
                if first_match_only:
                    return prim
                matching_prims.append(prim)

    return matching_prims if not first_match_only else None


def hide_matching_prims(match_strings: list[str], root_path: str | None = None, prim_type: str | None = None) -> None:
    """Set visibility of prims matching specified strings to 'invisible' within the root path."""
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPseudoRoot() if root_path is None else stage.GetPrimAtPath(root_path)

    for prim in Usd.PrimRange(root_prim):
        if prim_type is None or prim.GetTypeName() == prim_type:
            if any(match in str(prim.GetPath()) for match in match_strings):
                prim.GetAttribute("visibility").Set("invisible")



def get_random_pose_on_sphere(
    origin: tuple[float, float, float],
    radius_range: tuple[float, float],
    polar_angle_range: tuple[float, float],
    camera_forward_axis: tuple[float, float, float] = (0, 0, -1),
) -> tuple[Gf.Vec3d, Gf.Quatf]:
    """Generate a random pose on a sphere looking at the origin, with specified radius and polar angle ranges."""
    # https://docs.omniverse.nvidia.com/isaacsim/latest/reference_conventions.html
    # Convert degrees to radians for polar angles (theta)
    polar_angle_min_rad = math.radians(polar_angle_range[0])
    polar_angle_max_rad = math.radians(polar_angle_range[1])

    # Generate random spherical coordinates
    radius = random.uniform(radius_range[0], radius_range[1])
    polar_angle = random.uniform(polar_angle_min_rad, polar_angle_max_rad)
    azimuthal_angle = random.uniform(0, 2 * math.pi)

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * math.sin(polar_angle) * math.cos(azimuthal_angle)
    y = radius * math.sin(polar_angle) * math.sin(azimuthal_angle)
    z = radius * math.cos(polar_angle)

    # Calculate the location in 3D space
    location = Gf.Vec3d(origin[0] + x, origin[1] + y, origin[2] + z)

    # Calculate direction vector from camera to look_at point
    direction = Gf.Vec3d(origin) - location
    direction_normalized = direction.GetNormalized()

    # Calculate rotation from forward direction (rotateFrom) to direction vector (rotateTo)
    rotation = Gf.Rotation(Gf.Vec3d(camera_forward_axis), direction_normalized)
    orientation = Gf.Quatf(rotation.GetQuat())

    return location, orientation

def randomize_poses(
    prims: list[Usd.Prim],
    location_range: tuple[float, float, float, float, float, float],
    rotation_range: tuple[float, float],
    scale_range: tuple[float, float],
) -> None:
    """Randomize the location, rotation, and scale of a list of prims within specified ranges."""
    for prim in prims:
        rand_loc = (
            random.uniform(location_range[0], location_range[3]),
            random.uniform(location_range[1], location_range[4]),
            random.uniform(location_range[2], location_range[5]),
        )
        rand_rot = (
            random.uniform(rotation_range[0], rotation_range[1]),
            random.uniform(rotation_range[0], rotation_range[1]),
            random.uniform(rotation_range[0], rotation_range[1]),
        )
        rand_scale = random.uniform(scale_range[0], scale_range[1])
        set_transform_attributes(prim, location=rand_loc, rotation=rand_rot, scale=(rand_scale, rand_scale, rand_scale))

def randomize_camera_poses_rel_to_targets(
    cameras: list[Usd.Prim],
    targets: list[Usd.Prim],
    distance_range: tuple[float, float],
    polar_angle_range: tuple[float, float] = (0, 180),
    look_at_offset: tuple[float, float] = (-0.1, 0.1),
) -> None:
    """Randomize the poses of cameras to look at random targets with adjustable distance and offset."""
    for cam in cameras:
        # Get a random target asset to look at
        target_asset = random.choice(targets)

        # Add a look_at offset so the target is not always in the center of the camera view
        target_loc = target_asset.GetAttribute("xformOp:translate").Get()
        target_loc = (
            target_loc[0] + random.uniform(look_at_offset[0], look_at_offset[1]),
            target_loc[1] + random.uniform(look_at_offset[0], look_at_offset[1]),
            target_loc[2] + random.uniform(look_at_offset[0], look_at_offset[1]),
        )

        # Generate random camera pose
        loc, quat = get_random_pose_on_sphere(target_loc, distance_range, polar_angle_range)

        # Set the camera's transform attributes to the generated location and orientation
        set_transform_attributes(cam, location=loc, orientation=quat)

def randomize_camera_poses_rel_to_ws(
    cameras: list[Usd.Prim],
    targets: list[Usd.Prim],
    cam_ws: tuple[float, float, float, float, float, float],
    look_at_offset: tuple[float, float] = (-0.1, 0.1),
) -> None:
    """Randomize the poses of cameras to look at random targets with adjustable distance and offset."""
    for cam in cameras:
        # Get a random target asset to look at
        target_asset = random.choice(targets)

        # Add a look_at offset so the target is not always in the center of the camera view
        target_loc = target_asset.GetAttribute("xformOp:translate").Get()
        target_loc = (
            target_loc[0] + random.uniform(look_at_offset[0], look_at_offset[1]),
            target_loc[1] + random.uniform(look_at_offset[0], look_at_offset[1]),
            target_loc[2] + random.uniform(look_at_offset[0], look_at_offset[1]),
        )

        # Generate random camera pose
        camera_loc = (
            random.uniform(cam_ws[0], cam_ws[3]),
            random.uniform(cam_ws[1], cam_ws[4]),
            random.uniform(cam_ws[2], cam_ws[5]),
        )
        set_camera_view(eye=camera_loc, target=target_loc, camera_prim_path=cam.GetPath().pathString)


def mask_random_objects(objects: list[Usd.Prim], ratio: float = 0.5) -> list[Usd.Prim]:
    """Mask a random number of objects in the list."""
    num_objects = int(len(objects) * ratio)
    masked_objects = random.sample(objects, num_objects)
    for obj in masked_objects:
        obj.GetAttribute("visibility").Set("invisible")
    return masked_objects

def unmask_objects(objects: list[Usd.Prim]) -> None:
    """Unmask a list of objects."""
    for obj in objects:
        obj.GetAttribute("visibility").Set("inherited")

# def add_default_objects(physics=False):
#     full_objs_list = []

#     for obj in DEFAULT_OBJECTS:
#         full_objs_list.append(prefix_with_isaac_asset_server(obj))

#     assets = []
#     for obj in full_objs_list:
#         asset = rep.create.from_usd(obj)
#         prim = asset.get_output_prims()["prims"][0]

#         if physics:
#             add_colliders(prim, approximation_shape="boundingCube")

#         assets.append(asset)

#     return rep.create.group(assets)



def capture_pathtracing(delta_time=0.0, spp=128, pause_timeline=True):
    # timeline = omni.timeline.get_timeline_interface()

    # Set the render mode to PathTracing
    prev_render_mode = carb.settings.get_settings().get("/rtx/rendermode")
    carb.settings.get_settings().set("/rtx/pathtracing/clampSpp", 0)
    carb.settings.get_settings().set("/rtx/rendermode", "PathTracing")
    carb.settings.get_settings().set("/rtx/pathtracing/spp", spp)
    carb.settings.get_settings().set("/rtx/pathtracing/totalSpp", spp)
    carb.settings.get_settings().set("/rtx/pathtracing/optixDenoiser/enabled", 0)

    # # Make sure the timeline is playing, this is used to get the motion blur
    # if not timeline.is_playing():
    #     timeline.play()

    # Capture the frame by advancing the simulation for the given duration and combining the sub samples
    rep.orchestrator.step(delta_time=delta_time, pause_timeline=pause_timeline)

    # Restore the previous render and motion blur settings
    carb.settings.get_settings().set("/rtx/rendermode", prev_render_mode)


def capture_raytracing2(rt_subframes=-1, pause_timeline=True, delta_time=0.0,):
    if carb.settings.get_settings().get("/rtx/rendermode") != "RealTimePathTracing":
        carb.settings.get_settings().set("/rtx/rendermode", "RealTimePathTracing")    
    # Capture the frame by advancing the simulation for the given duration and combining the sub samples
    rep.orchestrator.step(rt_subframes=rt_subframes, delta_time=delta_time, pause_timeline=pause_timeline)# Update the app until a given simulation duration has passed (simulate the world between captures)



def run_simulation(num_frames: int, render: bool = True) -> None:
    """Run a simulation for a specified number of frames, optionally without rendering."""
    if render:
        # Start the timeline and advance the app, this will render the physics simulation results every frame
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_start_time(0)
        timeline.set_end_time(1000000)
        timeline.set_looping(False)
        timeline.play()
        for _ in range(num_frames):
            omni.kit.app.get_app().update()
        timeline.pause()
    else:
        # Run the physics simulation steps without advancing the app
        stage = omni.usd.get_context().get_stage()
        physx_scene = None

        # Search for or create a physics scene
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                physx_scene = PhysxSchema.PhysxSceneAPI.Apply(prim)
                break

        if physx_scene is None:
            physics_scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/PhysicsScene"))

        # Get simulation parameters
        physx_dt = 1 / physx_scene.GetTimeStepsPerSecondAttr().Get()
        physx_sim_interface = omni.physx.get_physx_simulation_interface()

        # Run physics simulation for each frame
        for _ in range(num_frames):
            physx_sim_interface.simulate(physx_dt, 0)
            physx_sim_interface.fetch_results()



def generate_kitti_labels(categories):
    """
    Automatically generates KITTI labels based on subfolder names.
    
    Returns:
        dict: A dictionary with category names as keys and RGBA color tuples as values.
    """
    
    # Create KITTI labels dictionary
    kitti_labels = {
        "UNLABELLED": (0, 0, 0, 0),
        "BACKGROUND": (0, 0, 0, 0),
    }
    
    # Add each category with a unique color
    categories_list = sorted(categories.keys())
    for i, category in enumerate(categories_list):
        if i < len(COLOR_PALETTE):
            kitti_labels[category] = COLOR_PALETTE[i]
        else:
            # Generate a color if we run out of predefined colors
            # Use a hash-based approach for consistent colors
            import hashlib
            hash_obj = hashlib.md5(category.encode())
            hash_bytes = hash_obj.digest()
            r = hash_bytes[0]
            g = hash_bytes[1]
            b = hash_bytes[2]
            kitti_labels[category] = (r, g, b, 255)
    
    
    return kitti_labels




def register_UWCam_KittiWriter() -> None:
    from isaacsim.oceansim.writers.UWCam_KittiWriter import UWCam_KittiWriter
    rep.WriterRegistry.register(UWCam_KittiWriter)






def resolve_scale_issues_with_metrics_assembler() -> None:
    """Enable and execute metrics assembler to resolve scale issues in the stage."""
    import omni.kit.app

    ext_manager = omni.kit.app.get_app().get_extension_manager()
    if not ext_manager.is_extension_enabled("omni.usd.metrics.assembler"):
        ext_manager.set_extension_enabled_immediate("omni.usd.metrics.assembler", True)
    from omni.metrics.assembler.core import get_metrics_assembler_interface

    stage_id = omni.usd.get_context().get_stage_id()
    get_metrics_assembler_interface().resolve_stage(stage_id)


def save_object_info(objects: list[Usd.Prim], cameras: list[Usd.Prim], output_path: str):
    info = defaultdict(dict)
    for cam in cameras:
        cam_name = cam.GetPath().pathString.split("/")[-1]
        objs_info = info.setdefault(cam_name, {})
        for obj in objects:
            obj_name = obj.GetPath().pathString.split("/")[-1]
            obj_info = objs_info.setdefault(obj_name, {})
            cam_to_obj_tf_gf = Gf.Transform()
            # NOTE: get_relative_transform returns column major matrix, so we need to transpose it before using it in Gf
            cam_to_obj_tf_gf.SetMatrix(Gf.Matrix4d(get_relative_transform(obj, cam).T.tolist()))
            rotation = cam_to_obj_tf_gf.GetRotation().GetQuat()
            translation = cam_to_obj_tf_gf.GetTranslation()
            scale = cam_to_obj_tf_gf.GetScale()
            obj_info["location"] = list(translation)
            obj_info["rotation"] = [rotation.GetReal()] + list(rotation.GetImaginary())
            obj_info["scale"] = list(scale)
            obj_info["visibility"] = True if obj.GetAttribute("visibility").Get() == "inherited" else False
    with open(output_path, "w") as f:
        json.dump(info, f)