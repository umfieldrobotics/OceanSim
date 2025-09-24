from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
import omni.replicator.core as rep
import omni.usd


##################################################################################
# A script to set the opacity threshold of all the shaders to 0.5
stage = omni.usd.get_context().get_stage()

# Traverse through all the prims in the stage and query the referenced asset path to get the object type
for prim in stage.Traverse():
    # Check if the prim is a UsdShade.Shader
    if UsdShade.Shader(prim):
        shader = UsdShade.Shader(prim)

        # Check if the shader's `info:id` is 'UsdPreviewSurface'
        shader_id_attr = shader.GetIdAttr()
        if shader_id_attr and shader_id_attr.Get() == "UsdPreviewSurface":
            print(f"Found UsdPreviewSurface shader at: {prim.GetPath()}")

            # Get the 'opacityThreshold' input
            opacity_threshold_input = shader.GetInput("opacityThreshold")

            # If the input exists, set its value
            if opacity_threshold_input:
                opacity_threshold_input.Set(0.5)
                print(f"  > Set 'opacityThreshold' to {0.5}")
            else:
                # If the input doesn't exist, create it and set the value
                shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.5)
                print(f"  > Created and set 'opacityThreshold' to {0.5}")
##################################################################################

##################################################################################
# A script to create a plane mesh with specified dimensions and resolution displacement scale according to a height map
# This is used to define the coarse collision for the imported terrain
from PIL import Image
import numpy as np
from pxr import Sdf, UsdGeom
from usdrt import Vt
import warp as wp
import omni.kit.commands


def create_plane_mesh(stage, target_path, plane_resolution=100, plane_width=100):

    _, tmp_path = omni.kit.commands.execute(
        "CreateMeshPrimWithDefaultXform",
        prim_type="Plane",
        u_patches=plane_resolution,
        v_patches=plane_resolution,
        u_verts_scale=1,
        v_verts_scale=1,
        half_scale=0.5 * plane_width,
        select_new_prim=False,
    )
    omni.kit.commands.execute("MovePrim", path_from=tmp_path, path_to=Sdf.Path(target_path))
    omni.usd.get_context().get_selection().set_selected_prim_paths([], False)
    return UsdGeom.Mesh.Get(stage, target_path)


@wp.kernel
def deform_points(
    points: wp.array(dtype=wp.vec3),
    height_map: wp.array2d(dtype=wp.float32),
    displacement_scale: float,
    mesh_size: float,
    tile_x: float,
    tile_y: float,
    deformed_points: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    
    # Get original point
    point = points[tid]
    
    # Map the point coordinates to height map indices with tiling
    # Normalize coordinates to [0, 1] range
    u = (point[0] + mesh_size * 0.5) / mesh_size  # X coordinate normalized
    v = (point[1] + mesh_size * 0.5) / mesh_size  # Y coordinate normalized
    
    # Clamp to valid range
    u = wp.clamp(u, 0.0, 1.0)
    v = wp.clamp(v, 0.0, 1.0)
    
    # Apply tiling - multiply by tile count and use modulo for repetition
    u_tiled = u * tile_x
    v_tiled = v * tile_y
    
    # Get fractional part for texture sampling
    u_frac = wp.frac(u_tiled)
    v_frac = wp.frac(v_tiled)
    
    # Convert to height map pixel coordinates (within single tile)
    x_idx = int(u_frac * float(height_map.shape[1] - 1))
    y_idx = int(height_map.shape[0] - 1) - int(v_frac * float(height_map.shape[0] - 1))
    # y_idx = int(v_frac * float(height_map.shape[0] - 1))
    
    # Ensure indices are within bounds
    x_idx = wp.clamp(x_idx, 0, height_map.shape[1] - 1)
    y_idx = wp.clamp(y_idx, 0, height_map.shape[0] - 1)
    
    # Create new point with height displacement
    deformed_points[tid] = wp.vec3(point[0], point[1], height_map[y_idx, x_idx] * displacement_scale)



height_map_path = "/home/haoyu-ma/Downloads/rocks_ground_02_8k/textures/rocks_ground_02_height_8k.png"
displacement_scale = 0.1 # This fucking value is the maximum displacement window in global scale
size = 2.0   # length of the sqaure plane in meters (stage units)
resolution = 100 # number of vertices + 1 per side
tile_x = 1.0  # number of tiles in the x direction
tile_y = 1.0  # number of tiles in the y direction
tarrain_path = "/World/Terrain"

stage = omni.usd.get_context().get_stage()

# Create the rendering plane mesh (with subdivision)
mesh = create_plane_mesh(stage, f"{tarrain_path}/plane", resolution, size * 100)
mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.catmullClark)
mesh.GetPrim().CreateAttribute("refinementEnableOverride", Sdf.ValueTypeNames.Bool).Set(True)
mesh.GetPrim().CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int).Set(4)

# TODO: Bind the material to the mesh

# Load the height map
img = Image.open(height_map_path)

# Convert to grayscale luminance if needed (e.g., RGB/RGBA/P/CMYK/YCbCr)
if img.mode not in ("L", "I;16", "I", "F"):
    img = img.convert("L")

# Get numpy array and normalize to [0, 1] based on dtype/bit depth
raw_array = np.array(img, copy=False)
dtype = raw_array.dtype
print('dtype', dtype)
if dtype == np.uint8:
    img_array = raw_array.astype(np.float32) / 255.0
elif dtype == np.uint16:
    img_array = raw_array.astype(np.float32) / 65535.0
elif dtype == np.uint32:
    img_array = raw_array.astype(np.float32) / 4294967295.0
else:
    raise ValueError(f"Unsupported dtype: {dtype}")


mesh_prim = create_plane_mesh(stage, f"{tarrain_path}/collider", resolution, size * 100)

# Get the original points from the plane mesh
points = mesh_prim.GetPointsAttr().Get()

# Convert points to Warp array
points_wp = wp.array(points, dtype=wp.vec3, device="cuda")
deformed_points_wp = wp.empty_like(points_wp, device="cuda")
# Convert height map to 2D Warp array
height_map_wp = wp.from_numpy(img_array, dtype=wp.float32, device="cuda")


wp.launch(
    kernel=deform_points,
    dim=len(points_wp),
    inputs=[points_wp, height_map_wp, displacement_scale, size, tile_x, tile_y],
    outputs=[deformed_points_wp],
    device="cuda"
)

# Apply the deformed points to the mesh
mesh_prim.GetPointsAttr().Set(Vt.Vec3fArray(deformed_points_wp.numpy()))


