import warp as wp
import omni.kit.commands
from pxr import Sdf, UsdGeom, Usd, Gf



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


def create_plane_with_hole(stage, target_path, plane_width=10, hole_size=2):
    """
    Create a plane mesh with a square hole in the middle.
    
    Args:
        stage: USD stage
        target_path: Path where to create the mesh
        plane_width: Width of the plane (default: 10)
        hole_size: Size of the square hole (default: 2)
    """
    # Create a mesh for the plane with hole
    plane = UsdGeom.Mesh.Define(stage, target_path)
    
    # Calculate vertices (scaled by plane_width/2)
    scale = plane_width / 2
    hole_scale = hole_size / 2
    
    vertices = [
        # Outer vertices (clockwise from bottom-left)
        Gf.Vec3f(-scale, -scale, 0),    # 0
        Gf.Vec3f(-hole_scale, -scale, 0),# 1
        Gf.Vec3f(hole_scale, -scale, 0), # 2
        Gf.Vec3f(scale, -scale, 0),      # 3
        
        Gf.Vec3f(-scale, -hole_scale, 0),# 4
        Gf.Vec3f(-hole_scale, -hole_scale, 0), # 5
        Gf.Vec3f(hole_scale, -hole_scale, 0),  # 6
        Gf.Vec3f(scale, -hole_scale, 0),       # 7
        
        Gf.Vec3f(-scale, hole_scale, 0),       # 8
        Gf.Vec3f(-hole_scale, hole_scale, 0),  # 9
        Gf.Vec3f(hole_scale, hole_scale, 0),   # 10
        Gf.Vec3f(scale, hole_scale, 0),        # 11
        
        Gf.Vec3f(-scale, scale, 0),     # 12
        Gf.Vec3f(-hole_scale, scale, 0),# 13
        Gf.Vec3f(hole_scale, scale, 0), # 14
        Gf.Vec3f(scale, scale, 0),      # 15
    ]
    
    # Define faces (each face is a quad defined by 4 vertices)
    face_vertex_counts = [4] * 8  # 8 quads
    face_vertex_indices = [
        0, 1, 5, 4,     # bottom-left segment
        1, 2, 6, 5,     # bottom-middle segment
        2, 3, 7, 6,     # bottom-right segment
        4, 5, 9, 8,     # middle-left segment
        6, 7, 11, 10,   # middle-right segment
        8, 9, 13, 12,   # top-left segment
        9, 10, 14, 13,  # top-middle segment
        10, 11, 15, 14, # top-right segment
    ]
    
    # Set the mesh attributes
    plane.GetPointsAttr().Set(vertices)
    plane.GetFaceVertexCountsAttr().Set(face_vertex_counts)
    plane.GetFaceVertexIndicesAttr().Set(face_vertex_indices)
    
    # Create UV mapping
    a = (hole_size / plane_width) / 2
    uv_mapping = {
        0: Gf.Vec2f(0.0, 0.0),
        1: Gf.Vec2f(0.5 - a, 0.0),
        2: Gf.Vec2f(0.5 + a, 0.0),
        3: Gf.Vec2f(1.0, 0.0),
        4: Gf.Vec2f(0.0, 0.5 - a),
        5: Gf.Vec2f(0.5 -a , 0.5 -a),
        6: Gf.Vec2f(0.5 + a, 0.5 - a),
        7: Gf.Vec2f(1.0, 0.5 -a),
        8: Gf.Vec2f(0.0, 0.5 + a),
        9: Gf.Vec2f(0.5 - a, 0.5 + a),
        10: Gf.Vec2f(0.5 + a, 0.5 + a),
        11: Gf.Vec2f(1.0, 0.5 + a),
        12: Gf.Vec2f(0.0, 1.0),
        13: Gf.Vec2f(0.5 - a, 1.0),
        14: Gf.Vec2f(0.5 + a, 1.0),
        15: Gf.Vec2f(1.0, 1.0),
    }
    
    # Create UV list matching face_vertex_indices order
    uvs = [uv_mapping[idx] for idx in face_vertex_indices]
    
    # Set the UV coordinates with faceVarying interpolation
    sts_primvar = UsdGeom.PrimvarsAPI(plane).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray)
    sts_primvar.SetInterpolation("faceVarying")
    sts_primvar.Set(uvs)    
    return plane



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
