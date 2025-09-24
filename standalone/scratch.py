# from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
# import omni.replicator.core as rep
# import omni.usd


# ##################################################################################
# # A script to set the opacity threshold of all the shaders to 0.5
# stage = omni.usd.get_context().get_stage()

# # Traverse through all the prims in the stage and query the referenced asset path to get the object type
# for prim in stage.Traverse():
#     # Check if the prim is a UsdShade.Shader
#     if UsdShade.Shader(prim):
#         shader = UsdShade.Shader(prim)

#         # Check if the shader's `info:id` is 'UsdPreviewSurface'
#         shader_id_attr = shader.GetIdAttr()
#         if shader_id_attr and shader_id_attr.Get() == "UsdPreviewSurface":
#             print(f"Found UsdPreviewSurface shader at: {prim.GetPath()}")

#             # Get the 'opacityThreshold' input
#             opacity_threshold_input = shader.GetInput("opacityThreshold")

#             # If the input exists, set its value
#             if opacity_threshold_input:
#                 opacity_threshold_input.Set(0.5)
#                 print(f"  > Set 'opacityThreshold' to {0.5}")
#             else:
#                 # If the input doesn't exist, create it and set the value
#                 shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.5)
#                 print(f"  > Created and set 'opacityThreshold' to {0.5}")
# ##################################################################################

# ##################################################################################
# # A script to create a plane mesh with specified dimensions and resolution displacement scale according to a height map
# # This is used to define the coarse collision for the imported terrain
from PIL import Image
import numpy as np
from pxr import Sdf, UsdGeom
from usdrt import Vt
import warp as wp
import omni.usd
from isaacsim.oceansim.utils.mesh_utils import *
import carb.settings

# # # Enable MDL displacement globally
# carb.settings.get_settings().set("/rtx/material/enableMDLDisplacement", True)
# import isaacsim.core.utils.stage as stage_utils
# stage_utils.update_stage()



height_map_path = "/frog-drive/projects/OceanSim/sim2real/SDG_assets/sceneAssets/Collected_rocky_2x2/textures/rocks_ground_02_height_8k.png"
displacement_scale = 0.1 # This fucking value is the maximum displacement window in global scale
size = 4.0   # length of the sqaure plane in meters (stage units)
resolution = 100 # number of vertices + 1 per side
tile_x = 1.0  # number of tiles in the x direction
tile_y = 1.0  # number of tiles in the y direction
terrain_path = "/World"
background_size = size * 10  # Size of the background plane


stage = omni.usd.get_context().get_stage()

# # Create the rendering plane mesh (with subdivision)
mesh = create_plane_mesh(stage, f"{terrain_path}/plain", resolution, size * 100)
mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.catmullClark)
mesh.GetPrim().CreateAttribute("refinementEnableOverride", Sdf.ValueTypeNames.Bool).Set(True)
mesh.GetPrim().CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int).Set(4)

# TODO: Bind the material to the mesh
# When coding PBR node, make sure texture has the right reading format (like sRGB or raw)
# !!! If displacement texture is sRGB, the displacement will be wrong !!!


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


mesh_prim = create_plane_mesh(stage, f"{terrain_path}/collider", resolution, size * 100)

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


# This function creates the backgroun plane with 16 vertices
# UV are mapped corner to corner, with a hole in the middle
##################
#     #    #     #
#     #    #     #
##################
#     #    #     #
#     #    #     #
##################
#     #    #     #   
#     #    #     #
##################
background_mesh = create_plane_with_hole(
    stage,
    f"{terrain_path}/background",
    plane_width=background_size,
    hole_size=size,
)

background_mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.bilinear)
background_mesh.GetPrim().CreateAttribute("refinementEnableOverride", Sdf.ValueTypeNames.Bool).Set(True)
background_mesh.GetPrim().CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int).Set(8)
