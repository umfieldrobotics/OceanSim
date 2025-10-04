import warp as wp
import omni.kit.commands
from pxr import Sdf, UsdGeom, Usd, Gf, UsdShade
from PIL import Image
import numpy as np
from usdrt import Vt
import math
from typing import Optional
# Prevent DecompressionBombError for very large images
Image.MAX_IMAGE_PIXELS = None

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

def read_displacement_map(height_map_path):
    if not height_map_path:
        print("No height map path provided, using default flat map.")
        return np.zeros((1024, 1024), dtype=np.float32)
    
    # Load the height map
    img = Image.open(height_map_path)

    # Convert to grayscale luminance if needed (e.g., RGB/RGBA/P/CMYK/YCbCr)
    if img.mode not in ("L", "I;16", "I", "F"):
        img = img.convert("L")

    # Get numpy array and normalize to [0, 1] based on dtype/bit depth
    raw_array = np.array(img, copy=False)
    dtype = raw_array.dtype
    if dtype == np.uint8:
        img_array = raw_array.astype(np.float32) / 255.0
    elif dtype == np.uint16:
        img_array = raw_array.astype(np.float32) / 65535.0
    elif dtype == np.uint32:
        img_array = raw_array.astype(np.float32) / 4294967295.0
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    return img_array



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

def create_plane_with_uvs(stage, prim_path, resolution, side_length, uv00, uv10, uv11, uv01):
    """
    Create a subdivided plane (XY square) with bilinearly interpolated UVs.

    Args:
        resolution (int): number of quads along one side
        side_length (float): size of the plane
        uv00 (tuple): bottom-left corner UV (x= -L/2, y= -L/2)
        uv10 (tuple): bottom-right corner UV (x= +L/2, y= -L/2)
        uv11 (tuple): top-right corner UV (x= +L/2, y= +L/2)
        uv01 (tuple): top-left corner UV (x= -L/2, y= +L/2)

    Returns:
        dict with:
            - points (list[Gf.Vec3f])
            - uvs_per_vertex (list[Gf.Vec2f])
            - quad_uvs (list[list[Gf.Vec2f]])
            - face_indices (list[int])
            - vertex_counts (list[int])
    """
    n = resolution
    L = side_length

    xs = np.linspace(-L/2, L/2, n+1)
    ys = np.linspace(-L/2, L/2, n+1)

    # Bilinear interpolation function
    def bilerp(u, v):
        # u,v in [0,1]
        return (
            (1-u)*(1-v)*np.array(uv00) +
            u*(1-v)*np.array(uv10) +
            u*v*np.array(uv11) +
            (1-u)*v*np.array(uv01)
        )

    points, uvs_per_vertex = [], []
    index_map = {}

    # Deduplicated grid points
    for j, y in enumerate(ys):
        v = j / n
        for i, x in enumerate(xs):
            u = i / n
            idx = j * (n+1) + i
            points.append(Gf.Vec3f(x, y, 0.0))
            uv = bilerp(u, v)
            uvs_per_vertex.append(Gf.Vec2f(*(uv.tolist())))
            index_map[(i, j)] = idx

    face_indices, vertex_counts, quad_uvs = [], [], []

    # Build quads
    for j in range(n):
        for i in range(n):
            v0 = index_map[(i, j)]
            v1 = index_map[(i+1, j)]
            v2 = index_map[(i+1, j+1)]
            v3 = index_map[(i, j+1)]

            face_indices.extend([v0, v1, v2, v3])
            vertex_counts.append(4)

            # Per-quad UVs (group of 4)
            quad_uvs.extend([
                uvs_per_vertex[v0],
                uvs_per_vertex[v1],
                uvs_per_vertex[v2],
                uvs_per_vertex[v3],
            ])
    mesh = create_custom_mesh(stage, prim_path, face_indices, vertex_counts, points, quad_uvs)
    return {
        "mesh": mesh,
        "points": points,
        "uvs": uvs_per_vertex,
        "quad_uvs": quad_uvs,
        "face_indices": face_indices,
        "vertex_counts": vertex_counts,
    }


@wp.kernel
def deform_points_uv_tiling(
    points: wp.array(dtype=wp.vec3),
    height_map: wp.array2d(dtype=wp.float32),
    displacement_scale: float,
    uvs: wp.array(dtype=wp.vec2),
    tile_x: float,
    tile_y: float,
    deformed_points: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    
    # Get original point
    point = points[tid]
    uv = uvs[tid]
    
    # Apply tiling - multiply by tile count and use modulo for repetition
    u_tiled = uv[0] * tile_x
    v_tiled = uv[1] * tile_y
    
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

def create_custom_mesh(stage, scene_path, face_vertex_indices=None, face_vertex_counts=None, points=None, sts=None):
    mesh = UsdGeom.Mesh.Define(stage, scene_path)
    mesh.CreateSubdivisionSchemeAttr("none")
    if face_vertex_indices is not None:
        mesh.GetFaceVertexIndicesAttr().Set(face_vertex_indices)
    if face_vertex_counts is not None:
        mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts)
    if points is not None:
        mesh.GetPointsAttr().Set(points)
    if sts is not None:
        sts_primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray)
        sts_primvar.SetInterpolation("faceVarying")
        sts_primvar.Set(sts)
    return mesh


def create_plane_mesh(stage, prim_path, resolution, side_length):
    """
    Generate a plane mesh centered at the origin in the X-Y plane.
    
    Args:
        resolution (int): Number of subdivisions along one axis (grid will be resolution x resolution).
        side_length (float): Physical length of the side of the plane.
    
    Returns:
        points (np.ndarray): (N, 3) array of vertex positions.
        face_indices (list[list[int]]): List of quads (each face is 4 indices).
        vertex_counts (list[int]): Number of vertices per face (always 4).
        uvs (np.ndarray): (N, 2) array of UV coordinates.
    """
    
    n = resolution
    L = side_length
    
    # generate grid of points (N+1 x N+1 vertices)
    xs = np.linspace(-L/2, L/2, n+1)
    ys = np.linspace(-L/2, L/2, n+1)
    points = [Gf.Vec3f(x, y, 0.0) for y in ys for x in xs]
    
    uvs = []
    # generate faces (each quad = 4 vertices)
    face_indices = []
    vertex_counts = []
    for j in range(n):
        for i in range(n):
            v0 = j * (n+1) + i
            v1 = v0 + 1
            v2 = v0 + (n+1) + 1
            v3 = v0 + (n+1)
            face_indices.extend([v0, v1, v2, v3])
            vertex_counts.append(4)
            
            
            # Normalized UVs (tile [0,1])
            u0, v0_uv = i/n, j/n
            u1, v1_uv = (i+1)/n, (j+1)/n
            uvs.extend([
                Gf.Vec2f(u0, v0_uv),
                Gf.Vec2f(u1, v0_uv),
                Gf.Vec2f(u1, v1_uv),
                Gf.Vec2f(u0, v1_uv)
            ])

    mesh = create_custom_mesh(stage, 
                              prim_path, 
                              face_indices,
                              vertex_counts,
                              points,
                              uvs)

    return mesh


def create_plane_with_cutout(stage, prim_path, resolution, side_length, hole_size):
    n = resolution
    L = side_length

    hole_size = min(hole_size, L)

    xs = np.linspace(-L/2, L/2, n+1)
    ys = np.linspace(-L/2, L/2, n+1)

    half_hole = hole_size / 2.0

    # Mesh containers
    face_indices_outer, vertex_counts_outer, quad_uvs_outer = [], [], []
    face_indices_cut,   vertex_counts_cut,   quad_uvs_cut   = [], [], []

    # Dedup containers
    points_outer, uvs_outer = [], []
    points_cut,   uvs_cut   = [], []

    cut_min_u, cut_min_v = math.inf, math.inf
    cut_max_u, cut_max_v = -math.inf, -math.inf
    cut_point_min_x, cut_point_min_y = math.inf, math.inf
    cut_point_max_x, cut_point_max_y = -math.inf, -math.inf
    # Maps (point -> index) for deduplication
    outer_map, cut_map = {}, {}

    def get_or_add_point(x, y, container_points, container_uvs, point_map):
        """Deduplicate points and UVs."""
        p = (x, y, 0.0)
        uv = ((x + L/2) / L, (y + L/2) / L)  # normalize [0,1]
        if p not in point_map:
            idx = len(container_points)
            container_points.append(Gf.Vec3f(*p))
            container_uvs.append(Gf.Vec2f(*uv))
            point_map[p] = idx
        return point_map[p]
    


    for j in range(n):
        for i in range(n):
            # Grid corners (in XY plane)
            corners = [
                (xs[i],   ys[j]),
                (xs[i+1], ys[j]),
                (xs[i+1], ys[j+1]),
                (xs[i],   ys[j+1]),
            ]

            # Quad center
            cx = (xs[i] + xs[i+1]) * 0.5
            cy = (ys[j] + ys[j+1]) * 0.5

            # Quad UVs (for per-face usage)
            quad_uvs = [
                Gf.Vec2f(i/n,     j/n),
                Gf.Vec2f((i+1)/n, j/n),
                Gf.Vec2f((i+1)/n, (j+1)/n),
                Gf.Vec2f(i/n,     (j+1)/n),
            ]

            if -half_hole <= cx <= half_hole and -half_hole <= cy <= half_hole:
                # --- belongs to cut-out mesh ---
                indices = []
                for (x, y) in corners:
                    idx = get_or_add_point(x, y, points_cut, uvs_cut, cut_map)
                    indices.append(idx)
                    p = (x, y, 0.0)
                    uv = ((x + L/2) / L, (y + L/2) / L)  # normalize [0,1]
                    cut_max_u = max(cut_max_u, uv[0])
                    cut_min_u = min(cut_min_u, uv[0])
                    cut_max_v = max(cut_max_v, uv[1])
                    cut_min_v = min(cut_min_v, uv[1])
                    cut_point_max_x = max(cut_point_max_x, p[0])
                    cut_point_min_x = min(cut_point_min_x, p[0])
                    cut_point_max_y = max(cut_point_max_y, p[1])
                    cut_point_min_y = max(cut_point_min_y, p[1])
                
                face_indices_cut.extend(indices)
                vertex_counts_cut.append(4)
                quad_uvs_cut.extend(quad_uvs)
            else:
                # --- belongs to outer mesh ---
                indices = []
                for (x, y) in corners:
                    idx = get_or_add_point(x, y, points_outer, uvs_outer, outer_map)
                    indices.append(idx)
                face_indices_outer.extend(indices)
                vertex_counts_outer.append(4)
                quad_uvs_outer.extend(quad_uvs)

    # Build both meshes
    outer_mesh = create_custom_mesh(
        stage,
        prim_path + '/background',
        face_indices_outer,
        vertex_counts_outer,
        points_outer,
        quad_uvs_outer  # still provide per-quad uvs
    )

    cutout_mesh = create_custom_mesh(
        stage,
        prim_path + '/plane',
        face_indices_cut,
        vertex_counts_cut,
        points_cut,
        quad_uvs_cut
    )

    # Provide extra per-vertex UV buffers for later use
    return {
        "mesh": outer_mesh,
        "face_indices": face_indices_outer,
        "vertex_counts": vertex_counts_outer,
        "points": points_outer,
        "quad_uvs": quad_uvs_outer,
        "uvs": uvs_outer
            },{
        "mesh": cutout_mesh,
        "face_indices": face_indices_cut,
        "vertex_counts": vertex_counts_cut,
        "points": points_cut,
        "quad_uvs": quad_uvs_cut,
        "uvs": uvs_cut,
        "uv00": Gf.Vec2f(cut_min_u, cut_min_v),
        "uv10": Gf.Vec2f(cut_max_u, cut_min_v),
        "uv11": Gf.Vec2f(cut_max_u, cut_max_v),
        "uv01": Gf.Vec2f(cut_min_u, cut_max_v),
        "px_max": cut_point_max_x,
        "px_min": cut_point_min_x,
        "py_max": cut_point_max_y,
        "py_min": cut_point_min_y,
            }
    

def export_prim_to_layer(prim: Usd.Prim, flatten=True,
                         include_session_layer=True) -> Optional[Sdf.Layer]:
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


def export_prim_to_string(prim: Usd.Prim, flatten=True,
                          include_session_layer=True) -> Optional[str]:
    """
    Return a string representation of the given prim

    Parameters
    ----------
    prim: Usd.Prim
        The usd object we wish to convert to a string
    flatten: bool
        If True, then return a representation with all USD composition arcs
        flattened. If False, then return the prim definition from the
        strongest composition arc that contributes opinions to this prim.
    include_session_layer: bool
        If True, then include changes from the session layer. If this is
        False, and the prim is ONLY defined in the session layer, then None
        will be returned.

    Note that no parents are included, and xforms are not "flattened" (even if
    `flatten` is True - `flatten` refers to USD composition arcs, not parent
    hierarchy or xforms).
    """
    if not flatten:
        prim_stack = prim.GetPrimStack()
        if not include_session_layer:
            non_session_layers = \
                prim.GetStage().GetLayerStack(includeSessionLayers=False)
            prim_stack = \
                [x for x in prim_stack if x.layer in non_session_layers]
            if not prim_stack:
                return None
            return prim.GetPrimStack()[0].GetAsText()

    solo_layer = export_prim_to_layer(
        prim, flatten=True, include_session_layer=include_session_layer)
    if solo_layer is None:
        # If include_session_layer was False, it's possible that the prim
        # doesn't exist any more...
        return None
    solo_primspec = solo_layer.GetPrimAtPath(solo_layer.defaultPrim)
    return solo_primspec.GetAsText()


def print_prim(prim: Usd.Prim, flatten=True, include_session_layer=True):
    """
    Print a string representation of the given prim.

    Parameters
    ----------
    prim: Usd.Prim
        The usd object we wish to convert to a string
    flatten: bool
        If True, then print a representation with all USD composition arcs
        flattened. If False, then print the prim definition from the
        strongest composition arc that contributes opinions to this prim.
    include_session_layer: bool
        If True, then include changes from the session layer. If this is
        False, and the prim is ONLY defined in the session layer, then 'None'
        will be printed.

    Note that no parents are included, and xforms are not "flattened" (even if
    `flatten` is True - `flatten` refers to USD composition arcs, not parent
    hierarchy or xforms).
    """
    print(export_prim_to_string(prim, flatten=flatten,
                                include_session_layer=include_session_layer))
