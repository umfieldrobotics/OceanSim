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
from pxr import Sdf, UsdGeom, UsdPhysics
import usdrt
import warp as wp
import omni.usd
from isaacsim.oceansim.utils.mesh_utils import *
import carb.settings
import os


carb.settings.get_settings().set("/rtx/material/enableMDLDisplacement", True)
texture_path = "/home/nsieh/Desktop/test/rocky_trail_8k/"
# displacement = 0.1 # This fucking value is the maximum displacement window in global scale
gt_size = 4.0 # meters (NOTE: only square texture is supported). This should comes with any texture assets online that indicates physical scan area
dp_factor = 0.05
patch_factor = 5
bg_factor = 5
resolution = 10 # number of vertices + 1 per side



displacement = dp_factor * gt_size
size = patch_factor * gt_size   # length of the sqaure plane in meters (stage units)
terrain_path = "/World"
tile_x = bg_factor * patch_factor # number of tiles in the x direction
tile_y = bg_factor * patch_factor # number of tiles in the y direction
background_size = size * bg_factor  # Size of the background plane


stage = omni.usd.get_context().get_stage()



def parse_textures(url) -> dict:
    """Parse a single texture folder and return a mapping of texture semantic -> file path.

    The function accepts a single texture folder path (the folder that directly contains texture files).
    Do NOT pass a parent folder that contains many texture folders — a separate caller should iterate those and call
    this function for each individual texture folder.

    It looks for common name tokens (case-insensitive) and maps files to these keys:
    'displacement', 'albedo', 'normal', 'roughness', 'AO', 'ORM'.

    If a type isn't found the key will be absent (caller can use .get(key, "")).
    """
    textures = {}

    # Expect the provided path to be the folder that directly contains texture files.
    if not os.path.isdir(url):
        raise ValueError(f"Provided path is not a directory: {url}")


    # Patterns in priority order for each semantic channel
    PATTERNS = {
        "displacement": ["height", "disp", "displacement"],
        "albedo": ["col", "diff", "albedo", "color", "basecolor"],
        "normal": ["nor", "normal"],
        "roughness": ["roughness", "rough"],
        "AO": ["ao", "ambientocclusion"],
        "ORM": ["arm", "orm", "occlusionroughnessmetallic", "ormap"],
    }

    # allowed texture file extensions
    ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".exr", ".tif", ".tiff"}

    # Track chosen priority index per key so higher-priority matches override lower-priority ones
    chosen_priority = {}

    # --- try to read meta file (meta.txt or meta.json) inside the texture folder ---
    meta = {}
    meta_txt = os.path.join(url, "meta.txt")
    # read meta.txt if present; store values as strings (don't assume they map to channels)

    with open(meta_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip().lower()] = v.strip()
    # try to convert meta values to float if possible
    for mk, mv in list(meta.items()):
        if isinstance(mv, str):
            try:
                # allow comma as decimal separator sometimes
                v = mv.replace(",", ".")
                meta[mk] = float(v)
            except Exception:
                # keep original string
                meta[mk] = mv

    textures["meta"] = meta

    # iterate all files in the provided texture folder
    texture_folder_url = os.path.join(url, "textures")
    if not os.path.isdir(texture_folder_url):
        raise ValueError(f"Texture folder does not exist: {texture_folder_url}")
    
    for fname in os.listdir(texture_folder_url):
        fpath = os.path.join(texture_folder_url, fname)

        if not os.path.isfile(fpath):
            continue
        name_lower = fname.lower()
        _, ext = os.path.splitext(name_lower)
        if ext not in ALLOWED_EXTS:
            continue

        for key, patterns in PATTERNS.items():
            for idx, pat in enumerate(patterns):
                if pat in name_lower:
                    # If not chosen yet, or this match has higher priority (smaller idx), pick it
                    if (key not in textures) or (idx < chosen_priority.get(key, 999)):
                        textures[key] = fpath
                        chosen_priority[key] = idx
                    # stop checking lower-priority patterns for this key
                    break
    
    return textures


texPath = parse_textures(texture_path)
bgMesh_info, plainMesh_info = create_plane_with_cutout(stage, resolution, background_size, size)

plainMesh_info['mesh'].GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.catmullClark)
plainMesh_info['mesh'].GetPrim().CreateAttribute("refinementEnableOverride", Sdf.ValueTypeNames.Bool).Set(True)
plainMesh_info['mesh'].GetPrim().CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int).Set(7)

bgMesh_info['mesh'].GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.bilinear)
bgMesh_info['mesh'].GetPrim().CreateAttribute("refinementEnableOverride", Sdf.ValueTypeNames.Bool).Set(True)
bgMesh_info['mesh'].GetPrim().CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int).Set(5)


plane_uvs = plainMesh_info["uvs"]
# uv00, uv10, uv11, uv01, point_x_min, point_x_max, point_y_min, point_y_max = get_plane_corners_uv(plainMesh_info['points'], plainMesh_info['uvs'])
colliderMesh_info = create_plane_with_uvs(
    stage, 
    1000, 
    plainMesh_info['px_max'] - plainMesh_info["px_min"], 
    plainMesh_info['uv00'], 
    plainMesh_info['uv10'], 
    plainMesh_info['uv11'], 
    plainMesh_info['uv01']
)



# Convert points to Warp array
colliderMeshPoints = wp.array(colliderMesh_info['points'], dtype=wp.vec3, device="cuda")
colliderMeshUVs = wp.array(colliderMesh_info['uvs'], dtype=wp.vec2, device='cuda')
deformed_points = wp.empty_like(colliderMeshPoints, device="cuda")
# Convert height map to 2D Warp array
img_array = read_displacement_map(texPath['displacement'])
height_map_wp = wp.from_numpy(img_array, dtype=wp.float32, device="cuda")


wp.launch(
    kernel=deform_points_uv_tiling,
    dim=len(colliderMeshPoints),
    inputs=[colliderMeshPoints, height_map_wp, displacement, colliderMeshUVs, tile_x, tile_y],
    outputs=[deformed_points],
    device="cuda"
)

colliderMesh_info['mesh'].GetPrim().GetAttribute('points').Set(Vt.Vec3fArray(deformed_points.numpy()))
colliderPrim = colliderMesh_info['mesh'].GetPrim()
colliderPrim.GetAttribute("visibility").Set("invisible")
UsdPhysics.CollisionAPI.Apply(colliderPrim)
meshCollisionAPI = UsdPhysics.MeshCollisionAPI.Apply(colliderPrim)
meshCollisionAPI.GetApproximationAttr().Set("none") # This defaults to triangle mesh colldier


# create the material and shader
scopePath = Sdf.Path(f"{terrain_path}/Looks")
stage.DefinePrim(scopePath, "Scope")
materialPath = scopePath.AppendChild("PlaneMaterial")
materialPrim = stage.DefinePrim(materialPath, "Material")
material = UsdShade.Material.Get(stage, materialPath)

omniPBRShaderPath = materialPath.AppendChild("OmniPBR")
omniPBRShaderPrim = stage.DefinePrim(omniPBRShaderPath, "Shader")
omniPBRShader= UsdShade.Shader.Get(stage, omniPBRShaderPath)


tilingConstShader = create_shader_node(
    "float2_const", "nvidia/support_definitions.mdl", "float2_const", materialPath
)
tilingConstShaderValOutput = tilingConstShader.CreateOutput("out", Sdf.ValueTypeNames.Float2)
tilingConstShaderValInput = tilingConstShader.CreateInput("f2", Sdf.ValueTypeNames.Float2)
tilingConstShaderValInput.Set(Gf.Vec2f(tile_x, tile_y))

omniPBRShaderOut = omniPBRShader.CreateOutput("out", Sdf.ValueTypeNames.Token)
material.CreateSurfaceOutput("mdl").ConnectToSource(omniPBRShaderOut)
material.CreateVolumeOutput("mdl").ConnectToSource(omniPBRShaderOut)
material.CreateDisplacementOutput("mdl").ConnectToSource(omniPBRShaderOut)
omniPBRShader.GetImplementationSourceAttr().Set(UsdShade.Tokens.sourceAsset)
omniPBRShader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
omniPBRShader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")

omniPBRShaderAlbedoInput = omniPBRShader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset)
omniPBRShaderRoughnessInput = omniPBRShader.CreateInput("reflectionroughness_texture", Sdf.ValueTypeNames.Asset)
omniOBRShaderEnableORMInput = omniPBRShader.CreateInput("enable_ORM_texture", Sdf.ValueTypeNames.Bool)
omniPBRShaderORMInput = omniPBRShader.CreateInput("ORM_texture", Sdf.ValueTypeNames.Asset)
omniPBRShaderAOInput = omniPBRShader.CreateInput("ao_texture", Sdf.ValueTypeNames.Asset)
omniPBRShaderNormalInput = omniPBRShader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset)
omniPBRShaderTilingInput = omniPBRShader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2)

omniPBRShaderAlbedoInput.Set(texPath.get("albedo", ""))
omniPBRShaderRoughnessInput.Set(texPath.get("roughness", ""))
omniOBRShaderEnableORMInput.Set(True)
omniPBRShaderORMInput.Set(texPath.get("ORM", ""))
omniPBRShaderAOInput.Set(texPath.get("AO", ""))
omniPBRShaderNormalInput.Set(texPath.get("normal", ""))
omniPBRShaderTilingInput.ConnectToSource(tilingConstShaderValOutput)



dispTexShader = create_shader_node(
    "file_texture", "nvidia/core_definitions.mdl", "file_texture", materialPath
)
dispTexShaderTexInput = dispTexShader.CreateInput("texture", Sdf.ValueTypeNames.Asset)
dispTexShaderTexInput.Set(texPath.get("displacement", ""))
dispTexShaderTilingInput = dispTexShader.CreateInput("scaling", Sdf.ValueTypeNames.Float2)
dispTexShaderMonoOutput = dispTexShader.CreateOutput("mono", Sdf.ValueTypeNames.Float)
dispTexShaderTilingInput.ConnectToSource(tilingConstShaderValOutput)
addDispShader = create_shader_node(
    "add_displacement", "nvidia/core_definitions.mdl", "add_displacement", materialPath
)
addDispShaderDisplacementInput = addDispShader.CreateInput("displacement", Sdf.ValueTypeNames.Float)
addDispShaderDisplacementInput.ConnectToSource(dispTexShaderMonoOutput)
addDispShaderBaseMaterialInput = addDispShader.CreateInput("base", Sdf.ValueTypeNames.Token)
addDispShaderBaseMaterialInput.ConnectToSource(omniPBRShaderOut)
addDispShaderScaleInput = addDispShader.CreateInput("displacement_scale", Sdf.ValueTypeNames.Float)
addDispShaderScaleInput.Set(displacement)

addDispShaderOut = addDispShader.CreateOutput("out", Sdf.ValueTypeNames.Token)
material.CreateSurfaceOutput("mdl").ConnectToSource(addDispShaderOut)
material.CreateVolumeOutput("mdl").ConnectToSource(addDispShaderOut)
material.CreateDisplacementOutput("mdl").ConnectToSource(addDispShaderOut)
material.CreateSurfaceOutput().ConnectToSource(addDispShaderOut)
material.CreateVolumeOutput().ConnectToSource(addDispShaderOut)
material.CreateDisplacementOutput().ConnectToSource(addDispShaderOut)


# # duplicate the material for a bigger tiling for the background plane
# materialPath2 = scopePath.AppendChild("BackgroundMaterial")
# omni.usd.duplicate_prim(stage, str(materialPath), str(materialPath2))
# # set new tiling value for the background material
# materialPrim2 = stage.GetPrimAtPath(materialPath2)
# material2 = UsdShade.Material.Get(stage, materialPath2)
# material2TilingConstShader = UsdShade.Shader.Get(stage, materialPath2.AppendChild("float2_const"))
# material2TilingConstShader.GetInput("f2").Set(Gf.Vec2f(bg_factor * tile_x, bg_factor * tile_y))


# bind the material to the central plane
UsdShade.MaterialBindingAPI.Apply(bgMesh_info['mesh'].GetPrim()).Bind(material)
UsdShade.MaterialBindingAPI.Apply(plainMesh_info['mesh'].GetPrim()).Bind(material)
