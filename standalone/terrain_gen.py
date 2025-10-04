# DISPLACEMENT_FACTOR = 0.05 # This fucking value is the maximum displacement window in global scale (meters)
# PATCH_FACTOR : int = 3
# RES = 100 # number of vertices + 1 per side for the central plane (visual)
# REFINE_LEVEL = 5 # subdivision level for the central plane (catmullClark)
# COL_RES = 1000 # number of vertices + 1 per side for the collider 
# BG_FACTOR = 10 # background plane size factor relative to the central terrain plane


import argparse
import os
import sys

# Check if there are any config files (yaml or json) are passed as arguments
parser = argparse.ArgumentParser()
parser.add_argument("--textures", required=True, help="Path to the texture folder")
parser.add_argument("--headless", action="store_false", help="headless mode")
args, unknown = parser.parse_known_args()

from isaacsim.simulation_app import SimulationApp

config = {
    "renderer": "RealTimePathTracing",
    "headless": args.headless,
    "extra_args": [
        # "--/persistent/renderer/rtpt/enabled=True",             # This enables RTX 2.0 for Isaac 4.5
        "--/persistent/rtx/modes/rt2/enabled=True",              # This enables RTX 2.0 for Isaac 5.0
        "--/persistent/rtx/modes/pt/enabled=True",              # This enables Path Tracing for Isaac 5.0
        "--/persistent/rtx/modes/rt/enabled=True",              # This enables Ray Tracing for Isaac 5.0
        "--/log/level=error",                                    # These will shut isaac sim the fuck up 
        "--/log/fileLogLevel=error", 
        "--/log/outputStreamLevel=error",
        "--/renderer/multiGpu/enabled=false"  
    ]
}
simulation_app = SimulationApp(config)
# load up OceanSim
import isaacsim.core.utils.extensions as extensions_utils

value = extensions_utils.enable_extension(extension_name='isaacsim.oceansim')
if value:
    print("[Terrain Gen] OceanSim loaded successfully")
else:
    simulation_app.update()
    simulation_app.close()
    sys.exit("[Terrain Gen] OceanSim loaded failed. Stopped...")


from pxr import Sdf, UsdShade, Usd, UsdGeom, Gf, UsdPhysics, UsdLux
import os
from isaacsim.oceansim.utils.mesh_utils import *
import carb
import omni.usd

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



def terrain_gen(texture_folder_url : str) -> None:
    dp_factor = 0.05
    patch_factor = 2
    bg_factor = 5
    resolution = 100 # number of vertices + 1 per side
    collider_res = 1000

    texture_name = os.path.basename(texture_folder_url)
    texPath = parse_textures(texture_folder_url)
    for i, key in enumerate(texPath.keys()):
        print(f"Imported [{i}] {key} for {texture_name}")


    # variables related to mesh dimension
    terrain_path = "/terrain"
    # This should comes with any texture assets online that indicates physical scan area 
    gt_texture_size = texPath["meta"].get('wide', 2.0) # meters (NOTE: only square texture is supported). 
    displacement = gt_texture_size * dp_factor # max displacement in meters 

    # NOTE: Since we are using trigangle mesh collider, the mesh resolution determines the collider resolution
    
    size = gt_texture_size * patch_factor   # length of the sqaure plane in meters (stage units) 
    tile_x =  bg_factor * patch_factor # number of tiles in the x direction
    tile_y = bg_factor * patch_factor # number of tiles in the y direction

    background_size = size * bg_factor  # Size of the background plane

    # Create a new stage
    omni.usd.get_context().new_stage()
    # Get the current stage
    stage = omni.usd.get_context().get_stage()

    # NOTE: need to enable MDL displacement for physical displacement to work (maybe?)
    carb.settings.get_settings().set("/rtx/material/enableMDLDisplacement", True)

    defaultPrim = stage.DefinePrim(terrain_path, "Xform")
    stage.SetDefaultPrim(defaultPrim)
    # This is the central plane (high res) for visual 
    bgMesh_info, plainMesh_info = create_plane_with_cutout(stage, terrain_path, resolution, background_size, size)

    plainMesh_info['mesh'].GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.catmullClark)
    plainMesh_info['mesh'].GetPrim().CreateAttribute("refinementEnableOverride", Sdf.ValueTypeNames.Bool).Set(True)
    plainMesh_info['mesh'].GetPrim().CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int).Set(7)

    bgMesh_info['mesh'].GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.bilinear)
    bgMesh_info['mesh'].GetPrim().CreateAttribute("refinementEnableOverride", Sdf.ValueTypeNames.Bool).Set(True)
    bgMesh_info['mesh'].GetPrim().CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int).Set(5)


    # uv00, uv10, uv11, uv01, point_x_min, point_x_max, point_y_min, point_y_max = get_plane_corners_uv(plainMesh_info['points'], plainMesh_info['uvs'])
    colliderMesh_info = create_plane_with_uvs(
        stage, 
        terrain_path + '/collider', 
        collider_res, 
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
    materialPath = scopePath.AppendChild(f"{texture_name}")
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

    # Apply the material
    UsdShade.MaterialBindingAPI.Apply(bgMesh_info['mesh'].GetPrim()).Bind(material)
    UsdShade.MaterialBindingAPI.Apply(plainMesh_info['mesh'].GetPrim()).Bind(material)


    # Export the matieral separately for asset reuse
    material_folder_url = os.path.join(texture_folder_url, "materials")
    # This is the material (lesss tiling) for the central plane
    export_prim_to_layer(materialPrim).Export(material_folder_url + f"/{texture_name}_mat.usd")
    print(f"[Terrain Gen] Exported material to: {material_folder_url}/")
    
    # Export the entire stage
    save_path = f"{texture_folder_url}/{texture_name}.usd"
    # save_stage in stage_utils is does not save carb settings
    omni.usd.get_context().save_as_stage(save_path)
    print(f"[Terrain Gen] Exported stage USD to: {texture_folder_url}/{texture_name}.usd")


def main():
    texture_folder_url = args.textures
    num_textures = len(os.listdir(texture_folder_url))
    print(f"[Terrain Gen] Found {num_textures} texture sets in {texture_folder_url}")
    print("[Terrain Gen] Start terrain generation...")
    for i, texture in enumerate(os.listdir(texture_folder_url)):
        texture_url = os.path.join(texture_folder_url, texture)
        if not os.path.isdir(texture_url):
            continue
        print(f'Processing texture {i+1}/{num_textures}: {texture}')
        terrain_gen(texture_url)
    
    print("[Terrain Gen] Terrain generation completed.")


main()
simulation_app.close()

