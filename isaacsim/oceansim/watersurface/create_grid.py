"""Warp kernel creating a grid mesh geometry."""

import warp as wp
from typing import Tuple

#   Helpers
# -----------------------------------------------------------------------------


@wp.func
def _define_face(
    face: int,
    vertex_1: int,
    vertex_2: int,
    vertex_3: int,
    vertex_4: int,
    out_face_vertex_indices: wp.array(dtype=int),
):
    out_face_vertex_indices[face * 4 + 0] = vertex_1
    out_face_vertex_indices[face * 4 + 1] = vertex_2
    out_face_vertex_indices[face * 4 + 2] = vertex_3
    out_face_vertex_indices[face * 4 + 3] = vertex_4


@wp.func
def _set_face_normals(
    face: int,
    normal: wp.vec3,
    out_normals: wp.array(dtype=wp.vec3),
):
    out_normals[face * 4 + 0] = normal
    out_normals[face * 4 + 1] = normal
    out_normals[face * 4 + 2] = normal
    out_normals[face * 4 + 3] = normal


@wp.func
def _set_face_uvs(
    face: int,
    uv_1: wp.vec2,
    uv_2: wp.vec2,
    uv_3: wp.vec2,
    uv_4: wp.vec2,
    out_uvs: wp.array(dtype=wp.vec2),
):
    out_uvs[face * 4 + 0] = uv_1
    out_uvs[face * 4 + 1] = uv_2
    out_uvs[face * 4 + 2] = uv_3
    out_uvs[face * 4 + 3] = uv_4


#   Kernel
# -----------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _kernel(
    half_size: wp.vec2,
    res: wp.vec2i,
    update_topology: int,
    dt_pos: wp.vec2,
    dt_uv: wp.vec2,
    out_points: wp.array(dtype=wp.vec3),
    out_face_vertex_indices: wp.array(dtype=int),
    out_normals: wp.array(dtype=wp.vec3),
    out_uvs: wp.array(dtype=wp.vec2),
):
    """Kernel to create a geometry mesh grid."""
    tid = wp.tid()

    i = int(tid % res[0])
    j = int(tid / res[0])

    if i == 0 and j == 0:
        point = 0
        out_points[point] = wp.vec3(
            half_size[0],
            0.0,
            half_size[1],
        )

    if i == 0:
        point = (j + 1) * (res[0] + 1)
        out_points[point] = wp.vec3(
            half_size[0],
            0.0,
            half_size[1] - dt_pos[1] * float(j + 1),
        )

    if j == 0:
        point = i + 1
        out_points[point] = wp.vec3(
            half_size[0] - dt_pos[0] * float(i + 1),
            0.0,
            half_size[1],
        )

    point = (j + 1) * (res[0] + 1) + i + 1
    out_points[point] = wp.vec3(
        half_size[0] - dt_pos[0] * float(i + 1),
        0.0,
        half_size[1] - dt_pos[1] * float(j + 1),
    )

    if update_topology:
        face = tid

        # Face vertex indices.
        vertex_4 = point
        vertex_3 = vertex_4 - 1
        vertex_1 = vertex_3 - res[0]
        vertex_2 = vertex_1 - 1
        _define_face(face, vertex_1, vertex_2, vertex_3, vertex_4, out_face_vertex_indices)

        # Vertex normals.
        _set_face_normals(face, wp.vec3(0.0, 1.0, 0.0), out_normals)

        # Vertex UVs.
        s_0 = 1.0 - dt_uv[0] * float(i)
        s_1 = 1.0 - dt_uv[0] * float(i + 1)
        t_0 = dt_uv[1] * float(j)
        t_1 = dt_uv[1] * float(j + 1)
        _set_face_uvs(
            face,
            wp.vec2(s_1, t_0),
            wp.vec2(s_0, t_0),
            wp.vec2(s_0, t_1),
            wp.vec2(s_1, t_1),
            out_uvs,
        )


#   Launcher
# -----------------------------------------------------------------------------
# dims = np.array([100, 100], dtype=int)
# size = np.array([100, 100],dtype=float)
# update_topology = True


def create_grid(
    dims: Tuple[float, float], 
    size: Tuple[float, float], 
    update_topology: bool = True
    ):
    """
    Create a grid with specified dimensions and size.
    The grid is flat and has extent of ([-50, 0, -50], [50, 0, 50])

    Parameters:
    dims (numpy.ndarray): An array of shape (2,) with integers specifying grid dimensions.
    size (numpy.ndarray): An array of shape (2,) with floats specifying the size of the grid.
    update_topology (bool): Whether to update topology (default is True).

    Returns:
    out_points (numpy.ndarray): shape is (point_count, 3)
    out_face_vertex_indices (numpy.ndarray): shape is (vertex_count, )
    out_face_vertex_counts (numpy.ndarray): shape is (vertex_count, )
    out_normals (numpy.ndarray): shape is (face_count, 3)
    out_uvs (numpy.ndarray): shape is (vertex_count, 2)
    """



    face_count = dims[0] * dims[1]
    vertex_count = face_count * 4
    point_count = (dims[0] + 1) * (dims[1] + 1)

    out_points = wp.array(shape=(point_count,), dtype=wp.vec3, device="cuda:0")
    out_face_vertex_counts = wp.array(shape=(face_count,), dtype=int, device="cuda:0")
    out_face_vertex_indices = wp.array(shape=(vertex_count,), dtype=int, device="cuda:0")
    out_normals = wp.array(shape=(face_count,),dtype=wp.vec3, device="cuda:0")
    out_uvs = wp.array(shape=(vertex_count,), dtype=wp.vec2, device="cuda:0")
    

    half_size = (
        size[0] * 0.5,
        size[1] * 0.5,
    )
    dt_pos = wp.vec2(
        size[0] / float(dims[0]),
        size[1] / float(dims[1]),
    )
    dt_uv = (
        1.0 / float(dims[0]),
        1.0 / float(dims[1]),
    )

    wp.launch(
        kernel=_kernel,
        dim=face_count,
        inputs=[
            half_size,
            dims,
            update_topology,
            dt_pos,
            dt_uv,
        ],
        outputs=[
            out_points,
            out_face_vertex_indices,
            out_normals,
            out_uvs,
        ],
    )

    # All mesh faces have 4 vertices
    out_face_vertex_counts.fill_(4)

    out_points = out_points.numpy()
    out_face_vertex_indices = out_face_vertex_indices.numpy()
    out_face_vertex_counts = out_face_vertex_counts.numpy()
    out_normals = out_normals.numpy()
    out_uvs = out_uvs.numpy()

    return out_points, out_face_vertex_indices, out_face_vertex_counts, out_normals, out_uvs
