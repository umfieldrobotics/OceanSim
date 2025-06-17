from .create_grid import create_grid
from .ocean_deform_kernels import ocean_deform_launch_kernel
import isaacsim.core.utils.prims as prims_utils
# Omniverse import
import carb.settings
import numpy as np
import warp as wp
from pxr import Sdf, UsdLux, Gf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Vt
import omni.kit.commands
import carb

# Isaac sim import
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import get_current_stage, add_reference_to_stage, create_new_stage, open_stage


class Caustics:
    def __init__(self):
        world_prim_path = '/World'
        prims_utils.create_prim(prim_path=world_prim_path, prim_type="Xform")
        settings = carb.settings.get_settings()
        settings.set("/rtx/caustics/enabled", True)
        settings.set("/rtx/raytracing/caustics/photonCountMultiplier", 500)
        settings.set("/rtx/raytracing/caustics/photonMaxBounces", 4)
        settings.set("/rtx/raytracing/caustics/positionPhi", 2.0)
        settings.set("/rtx/raytracing/caustics/normalPhi", 0.8)
        settings.set("/rtx/raytracing/caustics/eawFilteringSteps", 5)

        stage = get_current_stage()
        self._add_light_to_stage()

        # Create ocean surface grid
        ocean_xform_path = world_prim_path + '/ocean'
        prims_utils.create_prim(prim_path=ocean_xform_path, prim_type="Xform", position=[0,0,2])

        oceanSurf_prim_path = ocean_xform_path + '/ocean_surface'

        size = [10, 10]
        dims = [100, 100]

        from isaacsim.oceansim.caustics.create_grid import create_grid
        points, face_vertex_indices, face_vertex_counts, normals, uvs = create_grid(dims, size)
        meshGeom = UsdGeom.Mesh.Define(stage, oceanSurf_prim_path)
        
        self._grid = points
        meshGeom.CreatePointsAttr(points)
        meshGeom.CreateNormalsAttr(normals)
        meshGeom.CreateFaceVertexIndicesAttr(face_vertex_indices)
        meshGeom.CreateFaceVertexCountsAttr(face_vertex_counts)

        
        meshGeom.GetPrim().CreateAttribute("primvars:st", Sdf.ValueTypeNames.TexCoord2fArray, False).Set(uvs)

        meshGeom.AddRotateXYZOp().Set((90, 0, 0))
        meshGeom.GetPrim().CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
        meshGeom.GetPrim().CreateAttribute("primvars:enableShadowTerminatorFix", Sdf.ValueTypeNames.Bool).Set(False)
        
        self.ocean_surface_prim = meshGeom.GetPrim()

        material_prim_path = ocean_xform_path + '/water'
        omni.kit.commands.execute("CreateMdlMaterialPrim", 
                                  mtl_url="/home/haoyu/isaacsim/extsUser/isaacsim.oceansim/isaacsim/oceansim/modules/ocean_python/Water.mdl", 
                                  mtl_name="Water", 
                                  mtl_path=material_prim_path
                )
        mat = UsdShade.Material.Get(stage, material_prim_path)
        binding_api = UsdShade.MaterialBindingAPI.Apply(self.ocean_surface_prim)
        binding_api.Bind(mat)

        deformed_points = ocean_deform_launch_kernel(self._grid, self._time)


        self.ocean_surface_prim.GetAttribute("points").Set(deformed_points.numpy())
        
