from pxr import Sdf, UsdShade, Usd, UsdGeom, Gf
import omni.usd
import random
import os

def parse_textures(url) -> dict:
    textures = {}
    basePath = "/frog-drive/projects/OceanSim/sim2real/SDG_assets/sceneAssets/Collected_rocky_2x2/textures/"
    textures["displacement"] = basePath + "rocks_ground_02_height_8k.png"
    textures['albedo'] = basePath + "rocks_ground_02_col_8k.jpg"
    textures['normal'] = basePath + "rocks_ground_02_nor_gl_8k.exr"
    textures['roughness'] = basePath + "rocks_ground_02_rough_8k.exr"
    textures['AO'] = basePath + "rocks_ground_02_ao_8k.jpg"
    textures['ORM'] = basePath + "rocks_ground_02_arm_8k.jpg"
    return textures

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
    new_node.GetPrim().CreateAttribute("ui:nodegraph:node:pos", Sdf.ValueTypeNames.Float2).Set(
        Gf.Vec2f(random.uniform(-800.0, -600.0), random.uniform(100.0, 200.0))
    )
    return new_node


stage = omni.usd.get_context().get_stage()

xformPath = '/hello'
xformPrim = UsdGeom.Xform.Define(stage, xformPath)
spherePath = '/hello/world'
spherePrim = UsdGeom.Sphere.Define(stage, spherePath)
sphere = stage.GetPrimAtPath('/hello/world')
# generic_spherePrim = stage.DefinePrim('/hello/world_generic', 'Sphere')
# create the material and shader
scopePath = Sdf.Path("/Looks")
stage.DefinePrim(scopePath, "Scope")
materialPath = scopePath.AppendChild("RedMaterial")
materialPrim = stage.DefinePrim(materialPath, "Material")
material = UsdShade.Material.Get(stage, materialPath)

omniPBRShaderPath = materialPath.AppendChild("OmniPBR")
omniPBRShaderPrim = stage.DefinePrim(omniPBRShaderPath, "Shader")
omniPBRShader= UsdShade.Shader.Get(stage, omniPBRShaderPath)

texPath = parse_textures("dummy_url")

tilingConstShader = create_shader_node(
    "float2_const", "nvidia/support_definitions.mdl", "float2_const", materialPath
)
tilingConstShaderValOutput = tilingConstShader.CreateOutput("out", Sdf.ValueTypeNames.Float2)


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

addDispShaderOut = addDispShader.CreateOutput("out", Sdf.ValueTypeNames.Token)
material.CreateSurfaceOutput("mdl").ConnectToSource(addDispShaderOut)
material.CreateVolumeOutput("mdl").ConnectToSource(addDispShaderOut)
material.CreateDisplacementOutput("mdl").ConnectToSource(addDispShaderOut)
material.CreateSurfaceOutput().ConnectToSource(addDispShaderOut)
material.CreateVolumeOutput().ConnectToSource(addDispShaderOut)
material.CreateDisplacementOutput().ConnectToSource(addDispShaderOut)




#bind the material

binding_api = UsdShade.MaterialBindingAPI.Apply(sphere)
binding_api.Bind(material)
