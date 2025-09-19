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
# A script to create a plane mesh with specified dimensions and resolution

from pxr import Usd, UsdGeom, Gf, Sdf

def create_plane_mesh(stage, path, width=1.0, height=1.0, resolution_x=1, resolution_y=1):
    """
    Creates a plane mesh at the specified path in the given USD stage.

    Parameters:
        stage (Usd.Stage): The USD stage to add the mesh to.
        path (str): The SdfPath (e.g., '/World/Plane') to define the mesh at.
        width (float): Width of the plane along X.
        height (float): Height of the plane along Y.
        resolution_x (int): Number of subdivisions along X axis (horizontal).
        resolution_y (int): Number of subdivisions along Y axis (vertical).
    """
    mesh = UsdGeom.Mesh.Define(stage, path)

    # Create grid of points, normals, and UVs
    verts = []
    normals = []
    uvs = []
    
    for y in range(resolution_y + 1):
        for x in range(resolution_x + 1):
            # Vertices
            px = (x / resolution_x - 0.5) * width
            py = (y / resolution_y - 0.5) * height
            verts.append(Gf.Vec3f(px, py, 0.0))  # Z is up in USD by default
            
            # Normals (pointing up along Z)
            normals.append(Gf.Vec3f(0.0, 0.0, 1.0))
            
            # UV coordinates (normalized from 0 to 1)
            u = x / resolution_x
            v = y / resolution_y
            uvs.append(Gf.Vec2f(u, v))

    mesh.GetPointsAttr().Set(verts)
    
    # Set extent attribute for proper bounding box
    mesh.CreateExtentAttr().Set([Gf.Vec3f(-width/2, -height/2, 0), Gf.Vec3f(width/2, height/2, 0)])
    
    mesh.GetNormalsAttr().Set(normals)
    
    # Set UV coordinates as primvar:st
    mesh.GetPrim().CreateAttribute("primvars:st", Sdf.ValueTypeNames.TexCoord2fArray, False).Set(uvs)


    # Create face vertex indices and counts
    faceVertexIndices = []
    faceVertexCounts = []

    for y in range(resolution_y):
        for x in range(resolution_x):
            i = y * (resolution_x + 1) + x
            # Counter-clockwise winding order for proper face orientation
            faceVertexIndices += [
                i,
                i + 1,
                i + resolution_x + 2,
                i + resolution_x + 1
            ]
            faceVertexCounts.append(4)

    mesh.GetFaceVertexIndicesAttr().Set(faceVertexIndices)
    mesh.GetFaceVertexCountsAttr().Set(faceVertexCounts)
    
    # Set normal interpolation
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.varying)

    return mesh
    
    
from pxr import Sdf, UsdGeom
import omni.usd
import omni.kit.commands
def create_plane_mesh(stage, target_path):
        plane_resolution = 100
        plane_width = 100
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

stage = omni.usd.get_context().get_stage()
mesh = create_plane_mesh(stage, "/World/Plane")
# mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.catmullClark)
# mesh.GetPrim().CreateAttribute("refinementEnableOverride", Sdf.ValueTypeNames.Bool).Set(True)
# mesh.GetPrim().CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int).Set(4)





