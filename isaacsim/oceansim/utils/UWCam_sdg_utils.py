
# The warehouse distractors which will be added to the scene and randomized
# THese mesh are defualt and used to test the code
DEFAULT_OBJECTS = 3 * [
    "/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/S_WetFloorSign.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_01.usd",
    # "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_02.usd",
    # "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_03.usd",
    # "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_B_01.usd",
    # "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_B_01.usd",
    # "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_B_03.usd",
    # "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_C_02.usd",
    # "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticA_02.usd",
    # "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticB_01.usd",
    # "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticA_02.usd",
    # "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticA_02.usd",
    # "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticD_01.usd",
    # "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticE_01.usd",
    # "/Isaac/Environments/Simple_Warehouse/Props/SM_BucketPlastic_B.usd",
    # "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_1262.usd",
    # "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_1268.usd",
    # "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_1482.usd",
    # "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_1683.usd",
    # "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_291.usd",
    # "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_01_1454.usd",
    # "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_01_1513.usd",
    # "/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_A_04.usd",
    # "/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_B_03.usd",
    # "/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_B_05.usd",
    # "/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_C_02.usd",
    # "/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_E_02.usd",
    # "/Isaac/Environments/Simple_Warehouse/Props/SM_PushcartA_02.usd",
    # "/Isaac/Environments/Simple_Warehouse/Props/SM_RackPile_04.usd",
    # "/Isaac/Environments/Simple_Warehouse/Props/SM_RackPile_03.usd",
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

COU_objects_folder_path = "/frog-drive/ocean-sim/sim2real/ObjectAssets_simready/"

import carb.settings
import omni.replicator.core as rep
import omni.timeline
from isaacsim.storage.native import get_assets_root_path
import random
import numpy as np
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics
from isaacsim.core.utils.stage import get_current_stage
from omni.kit.viewport.utility import get_active_viewport
import os

def parse_object_folder(objects_folder_path):
    """
    Parses the object folder and returns a dictionary of subfolders and their corresponding paths.
    """
    subfolders = {}
    try:
        # Get all items in the directory
        items = os.listdir(objects_folder_path)
        print(f"Found {len(items)} categories: {items} in {objects_folder_path}")
        # Filter for directories only
        for item in items:
            item_path = os.path.join(objects_folder_path, item)
            if os.path.isdir(item_path):
                subfolders[item] = item_path
        return subfolders
                
    except PermissionError:
        print(f"Error: Permission denied accessing {objects_folder_path}")
        return {}
    except Exception as e:
        print(f"Error accessing {objects_folder_path}: {e}")
        return {}



def add_COU_objects(objects_folder_path=COU_objects_folder_path, physics=False):
    """
    Returns all subfolder names and their corresponding paths within the COU_objects_folder_path.
    
    Returns:
        dict: A dictionary where keys are subfolder names and values are their full paths.
              Returns empty dict if the folder doesn't exist or has no subfolders.
    """
    if categories := parse_object_folder(objects_folder_path):

        assets_paths = []
        for category, folder_path in categories.items():
            node = rep.create.from_dir(folder_path, recursive=True, semantics=[("class", category)])
            if physics:
                with node:
                    rep.physics.collider("boundingSphere")


            for prim in node.get_output_prims()["prims"]:
                print(f"{category} : {prim.GetPath().pathString} added.")
                assets_paths.append(prim.GetPath())

        assets_group = rep.create.group(assets_paths)
        return assets_group, generate_kitti_labels(categories)
    
    else:
        print(f"Adding default objects")
        return add_default_objects(), DEFAULT_LABELS

# needed for loading textures correctly
def prefix_with_isaac_asset_server(relative_path):
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        raise Exception(
            "Nucleus server not found, could not access Isaac Sim assets folder"
        )
    return assets_root_path + relative_path




# Add transformation properties to the prim (if not already present)
def set_transform_attributes(prim, location=None, orientation=None, rotation=None, scale=None):
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


# Enables collisions with the asset (without rigid body dynamics the asset will be static)
def add_colliders(root_prim):
    # Iterate descendant prims (including root) and add colliders to mesh or primitive types
    for desc_prim in Usd.PrimRange(root_prim):
        if desc_prim.IsA(UsdGeom.Mesh) or desc_prim.IsA(UsdGeom.Gprim):
            # Physics
            if not desc_prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI.Apply(desc_prim)
            else:
                collision_api = UsdPhysics.CollisionAPI(desc_prim)
            collision_api.CreateCollisionEnabledAttr(True)
            # PhysX
            if not desc_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(desc_prim)
            else:
                physx_collision_api = PhysxSchema.PhysxCollisionAPI(desc_prim)
            # Set PhysX specific properties
            physx_collision_api.CreateContactOffsetAttr(0.001)
            physx_collision_api.CreateRestOffsetAttr(0.0)

        # Add mesh specific collision properties only to mesh types
        if desc_prim.IsA(UsdGeom.Mesh):
            # Add mesh collision properties to the mesh (e.g. collider aproximation type)
            if not desc_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(desc_prim)
            else:
                mesh_collision_api = UsdPhysics.MeshCollisionAPI(desc_prim)
            mesh_collision_api.CreateApproximationAttr().Set("convexHull")


# Check if prim (or its descendants) has colliders
def has_colliders(root_prim):
    for desc_prim in Usd.PrimRange(root_prim):
        if desc_prim.HasAPI(UsdPhysics.CollisionAPI):
            return True
    return False


# Enables rigid body dynamics (physics simulation) on the prim
def add_rigid_body_dynamics(prim, disable_gravity=False, angular_damping=None):
    if has_colliders(prim):
        # Physics
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        else:
            rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
        rigid_body_api.CreateRigidBodyEnabledAttr(True)
        # PhysX
        if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        else:
            physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI(prim)
        physx_rigid_body_api.GetDisableGravityAttr().Set(disable_gravity)
        if angular_damping is not None:
            physx_rigid_body_api.CreateAngularDampingAttr().Set(angular_damping)
    else:
        print(f"Prim '{prim.GetPath()}' has no colliders. Skipping rigid body dynamics properties.")


# Add dynamics properties to the prim (if mesh or primitive) (rigid body to root + colliders to the meshes)
# https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/rigid-bodies.html#rigid-body-simulation
def add_colliders_and_rigid_body_dynamics(prim, disable_gravity=False):
    # Add colliders to mesh or primitive types of the descendants of the prim (including root)
    add_colliders(prim)
    # Add rigid body dynamics properties (to the root only) only if it has colliders
    add_rigid_body_dynamics(prim, disable_gravity=disable_gravity)


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


# Enable or disable the render products and viewport rendering
def set_render_products_updates(render_products, enabled, include_viewport=False):
    for rp in render_products:
        rp.hydra_texture.set_updates_enabled(enabled)
    if include_viewport:
        get_active_viewport().updates_enabled = enabled

# Generate a random pose on a sphere looking at the origin
# https://docs.omniverse.nvidia.com/isaacsim/latest/reference_conventions.html
def get_random_pose_on_sphere(origin, radius, camera_forward_axis=(0, 0, -1)):
    origin = Gf.Vec3f(origin)
    camera_forward_axis = Gf.Vec3f(camera_forward_axis)

    # Generate random angles for spherical coordinates
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.arcsin(np.random.uniform(-1, 1))

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




def add_default_objects(physics=False):
    full_objs_list = []

    for obj in DEFAULT_OBJECTS:
        full_objs_list.append(prefix_with_isaac_asset_server(obj))

    assets = []
    for obj in full_objs_list:
        asset = rep.create.from_usd(obj)
        # prim = asset.get_output_prims()["prims"][0]

        if physics:
            with asset:
                rep.physics.collider("boundingSphere")

        assets.append(asset)

    return rep.create.group(assets)



def capture_pathtracing(duration=0.0, spp=128):
    timeline = omni.timeline.get_timeline_interface()

    # Set the render mode to PathTracing
    prev_render_mode = carb.settings.get_settings().get("/rtx/rendermode")
    carb.settings.get_settings().set("/rtx/pathtracing/clampSpp", 0)
    carb.settings.get_settings().set("/rtx/rendermode", "PathTracing")
    carb.settings.get_settings().set("/rtx/pathtracing/spp", spp)
    carb.settings.get_settings().set("/rtx/pathtracing/totalSpp", spp)
    carb.settings.get_settings().set("/rtx/pathtracing/optixDenoiser/enabled", 0)

    # Make sure the timeline is playing
    if not timeline.is_playing():
        timeline.play()

    # Capture the frame by advancing the simulation for the given duration and combining the sub samples
    rep.orchestrator.step(delta_time=duration, pause_timeline=False)

    # Restore the previous render and motion blur settings
    print(f"[SDG] Restoring render mode from 'PathTracing' to '{prev_render_mode}'")
    carb.settings.get_settings().set("/rtx/rendermode", prev_render_mode)


def capture_raytracing2(rt_subframes=-1, pause_timeline=True, duration=0.0,):
    if carb.settings.get_settings().get("/rtx/rendermode") != "RealTimePathTracing":
        carb.settings.get_settings().set("/rtx/rendermode", "RealTimePathTracing")    
    # Capture the frame by advancing the simulation for the given duration and combining the sub samples
    rep.orchestrator.step(rt_subframes=rt_subframes, delta_time=duration, pause_timeline=pause_timeline)# Update the app until a given simulation duration has passed (simulate the world between captures)



def run_simulation_loop(duration, simulation_app):
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




def generate_kitti_labels(categories):
    """
    Automatically generates KITTI labels based on subfolder names in the given path.
    
    Args:
        folder_path (str): Path to the folder containing subfolders for object categories.
    
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








