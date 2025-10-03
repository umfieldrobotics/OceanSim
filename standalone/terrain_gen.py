DISPLACEMENT_FACTOR = 0.05 # This fucking value is the maximum displacement window in global scale (meters)
PATCH_FACTOR : int = 2
RES = 100 # number of vertices + 1 per side for the central plane (visual)
REFINE_LEVEL = 5 # subdivision level for the central plane (catmullClark)
COL_RES = 1000 # number of vertices + 1 per side for the collider 
BG_FACTOR = 10 # background plane size factor relative to the central terrain plane


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


def export_prim_to_layer(prim: Usd.Prim, flatten=True,
                         include_session_layer=True) -> Sdf.Layer | None:
    """
    Creates a temporary layer with only the given prim in it

    Parameters
    ----------
    prim: Usd.Prim
        The usd object we wish to export by itself into its own layer
    flatten: bool
        If True, then the returned layer will have all composition arcs
        flattened. If False, then the returned layer will contain a reference
        to the original prim.
    include_session_layer: bool
        If True, then include changes from the session layer. If this is
        False, and the prim is ONLY defined in the session layer, then None
        will be returned.

    Note that no parents are included, and xforms are not "flattened" (even if
    `flatten` is True - `flatten` refers to USD composition arcs, not parent
    hierarchy or xforms).
    """
    orig_stage = prim.GetStage()
    orig_root_layer = orig_stage.GetRootLayer()
    orig_session_layer = orig_stage.GetSessionLayer()
    if orig_session_layer and not include_session_layer:
        # Make sure that the prim still exists if we exclude the session layer
        stage_no_session = Usd.Stage.Open(orig_root_layer, sessionLayer=None)
        if not stage_no_session.GetPrimAtPath(prim.GetPrimPath()):
            return None

        orig_session_layer = None

    # If there is a session layer, to get an EXACT copy including possible
    # modifications by the session layer, we need to create a new "copy" layer,
    # with a layer stack composed of the orig_stage's session layer
    # and root layer
    if not orig_session_layer:
        copy_layer = orig_root_layer
    else:
        copy_layer = Sdf.Layer.CreateAnonymous()
        copy_layer.subLayerPaths.append(orig_session_layer.identifier)
        copy_layer.subLayerPaths.append(orig_root_layer.identifier)

    # Now create a "solo" stage, with only our prim (from the copy layer)
    # referenced in
    solo_stage = Usd.Stage.CreateInMemory()
    solo_prim_path = Sdf.Path(f"/{prim.GetName()}")
    solo_prim = solo_stage.DefinePrim(solo_prim_path)
    solo_prim.GetReferences().AddReference(copy_layer.identifier,
                                           prim.GetPrimPath())
    solo_stage.SetDefaultPrim(solo_prim)

    if flatten:
        return solo_stage.Flatten()
    else:
        return solo_stage.GetRootLayer()

def create_shader_node(name, source, function, shader_path):
    # add construct_color node
    new_node = UsdShade.Shader.Define(omni.usd.get_context().get_stage(), shader_path.AppendChild(name))
    # Set attributes
    api_schemas = Sdf.TokenListOp()
    api_schemas.explicitItems = ["NodeGraphNodeAPI"]
    new_node.GetPrim().SetMetadata("apiSchemas", api_schemas)  # Add NodeGraphNodeAPI
    new_node.CreateIdAttr("sourceAsset")  # Set implementation source
    new_node.GetPrim().CreateAttribute("info:implementationSource", Sdf.ValueTypeNames.Token).Set("sourceAsset")
    new_node.GetPrim().CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(source)
    new_node.GetPrim().CreateAttribute("info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token).Set(function)
    new_node.GetPrim().CreateAttribute("ui:nodegraph:node:expansionState", Sdf.ValueTypeNames.Token).Set("open")

    return new_node




def terrain_gen(texture_folder_url : str) -> None:


    texture_name = os.path.basename(texture_folder_url)
    texPath = parse_textures(texture_folder_url)
    for i, key in enumerate(texPath.keys()):
        print(f"Imported [{i}] {key} for {texture_name}")


    # variables related to mesh dimension
    terrain_path = "/terrain"
    # This should comes with any texture assets online that indicates physical scan area 
    gt_texture_size = texPath["meta"].get('wide', 2.0) # meters (NOTE: only square texture is supported). 
    displacement = gt_texture_size * DISPLACEMENT_FACTOR # max displacement in meters 

    # NOTE: Since we are using trigangle mesh collider, the mesh resolution determines the collider resolution
    
    size = gt_texture_size * PATCH_FACTOR   # length of the sqaure plane in meters (stage units) 
    tile_x =  size / gt_texture_size # number of tiles in the x direction
    tile_y = size / gt_texture_size # number of tiles in the y direction

    background_size = size * BG_FACTOR  # Size of the background plane

    # Create a new stage
    omni.usd.get_context().new_stage()
    # Get the current stage
    stage = omni.usd.get_context().get_stage()

    # NOTE: need to enable MDL displacement for physical displacement to work (maybe?)
    carb.settings.get_settings().set("/rtx/material/enableMDLDisplacement", True)

    defaultPrim = stage.DefinePrim(terrain_path, "Xform")
    stage.SetDefaultPrim(defaultPrim)
    # This is the central plane (high res) for visual 
    mesh = create_plane_mesh(stage, f"{terrain_path}/plain", RES, size * 100)

    # Create subdivision schema and assign subdivion level
    mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.catmullClark)
    mesh.GetPrim().CreateAttribute("refinementEnableOverride", Sdf.ValueTypeNames.Bool).Set(True)
    mesh.GetPrim().CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int).Set(REFINE_LEVEL) 
    meshPrim = stage.GetPrimAtPath(f"{terrain_path}/plain")


    # This is the invisible collider (low res) for physics
    meshCollider = create_displaced_plane_mesh(
        stage,
        f"{terrain_path}/collider",
        height_map_path=texPath.get("displacement", ""),
        plane_resolution=COL_RES,
        plane_width=size,
        displacement_scale=displacement,
        tile_x=tile_x,
        tile_y=tile_y,
    )
    meshColliderPrim = stage.GetPrimAtPath(f"{terrain_path}/collider")
    meshColliderPrim.GetAttribute("visibility").Set("invisible")

    # Create background mesh for visual (NOTE: no collider for the background)
    bgMesh = create_plane_with_hole(
        stage,
        f"{terrain_path}/background",
        plane_width=background_size,
        hole_size=size,
    )
    # NOTE: not using catmullClark because it creates smoothed hole edge in the center
    bgMesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.bilinear)
    bgMesh.GetPrim().CreateAttribute("refinementEnableOverride", Sdf.ValueTypeNames.Bool).Set(True)
    bgMesh.GetPrim().CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int).Set(8) # Maximize the subdivision for best visual
    bgMeshPrim = stage.GetPrimAtPath(f"{terrain_path}/background")

    # Assign physics coollder to this mesh
    UsdPhysics.CollisionAPI.Apply(meshColliderPrim)
    meshCollisionAPI = UsdPhysics.MeshCollisionAPI.Apply(meshColliderPrim)
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


    # duplicate the material for a bigger tiling for the background plane
    materialPath2 = scopePath.AppendChild("BackgroundMaterial")
    omni.usd.duplicate_prim(stage, str(materialPath), str(materialPath2))
    # set new tiling value for the background material
    materialPrim2 = stage.GetPrimAtPath(materialPath2)
    material2 = UsdShade.Material.Get(stage, materialPath2)
    material2TilingConstShader = UsdShade.Shader.Get(stage, materialPath2.AppendChild("float2_const"))
    material2TilingConstShader.GetInput("f2").Set(Gf.Vec2f(BG_FACTOR * tile_x, BG_FACTOR * tile_y))


    # bind the material to the central plane
    binding_api = UsdShade.MaterialBindingAPI.Apply(meshPrim)
    binding_api.Bind(material)
    # bind the material to the background plane
    binding_api2 = UsdShade.MaterialBindingAPI.Apply(bgMeshPrim)
    binding_api2.Bind(material2)


    # Export the matieral separately for asset reuse
    material_folder_url = os.path.join(texture_folder_url, "materials")
    # This is the material (lesss tiling) for the central plane
    export_prim_to_layer(materialPrim).Export(material_folder_url + f"/{texture_name}_main.usd")
    # This is the material (more tiling) for the background plane
    export_prim_to_layer(materialPrim2).Export(material_folder_url + f"/{texture_name}_bg.usd")
    print(f"[Terrain Gen] Exported materials to: {material_folder_url}/")
    
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

