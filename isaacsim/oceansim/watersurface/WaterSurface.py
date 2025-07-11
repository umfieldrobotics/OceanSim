from typing import Optional, Sequence
from pxr import Sdf, UsdLux, Gf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
from usdrt import Vt
import omni.kit.commands
import numpy as np
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.api.materials.preview_surface import PreviewSurface
from isaacsim.core.api.materials.visual_material import VisualMaterial
from isaacsim.core.prims import SingleGeometryPrim, SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_at_path, is_prim_path_valid
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.extensions import get_extension_path
from .create_grid import create_grid
from .ocean_deform_kernels import update_profile, update_points
import carb
import warp as wp

#  Internal parameters
PROFILE_EXTENT = 410.0
PROFILE_RES = 8192
PROFILE_WAVENUM = 1000
MIN_WAVE_LENGTH = 0.1
MAX_WAVE_LENGTH = 250.0


class WaterSurface(SingleGeometryPrim):
    """High level wrapper to create/encapsulate a visual water surface

    .. note::

        Visual water surface have no collisions (Collider API) or rigid body dynamics (Rigid Body API)

    Args:
        prim_path (str): prim path of the Prim to encapsulate or create
        name (str, optional): shortname to be used as a key by Scene class.
                                Note: needs to be unique if the object is added to the Scene.
                                Defaults to "visual_capsule".
        position (Optional[Sequence[float]], optional): position in the world frame of the prim. shape is (3, ).
                                                        Defaults to None, which means left unchanged.
        translation (Optional[Sequence[float]], optional): translation in the local frame of the prim
                                                        (with respect to its parent prim). shape is (3, ).
                                                        Defaults to None, which means left unchanged.
        orientation (Optional[Sequence[float]], optional): quaternion orientation in the world/ local frame of the prim
                                                        (depends if translation or position is specified).
                                                        quaternion is scalar-first (w, x, y, z). shape is (4, ).
                                                        Defaults to None, which means left unchanged.
        scale (Optional[Sequence[float]], optional): local scale to be applied to the prim's dimensions. shape is (3, ).
                                                Defaults to None, which means left unchanged.
        visible (bool, optional): set to false for an invisible prim in the stage while rendering. Defaults to True.
        color (Optional[np.ndarray], optional): color of the visual shape. Defaults to None, which means 50% gray
        radius (Optional[float], optional): capsule radius. Defaults to None.
        height (Optional[float], optional): capsule height. Defaults to None.
        visual_material (Optional[VisualMaterial], optional): visual material to be applied to the held prim.
                                Defaults to None. If not specified, a default visual material will be added.

    Example:

    .. code-block:: python

        >>> from isaacsim.core.api.objects import VisualCapsule
        >>> import numpy as np
        >>>
        >>> # create a red visual capsule at the given path
        ... prim = VisualCapsule(
        ...     prim_path="/World/Xform/Capsule",
        ...     radius=0.5,
        ...     height=1.0,
        ...     color=np.array([1.0, 0.0, 0.0])
        ... )
        >>> prim
        <isaacsim.core.api.objects.capsule.VisualCapsule object at 0x7f4ff958b0d0>
    """

    def __init__(
        self,
        prim_path: str,
        name: str = "water_surface",
        position: Optional[Sequence[float]] = None,
        translation: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
        scale: Optional[Sequence[float]] = None,
        size: Optional[Sequence[float]] = [10.0, 10.0],
        dim: Optional[Sequence[int]] = [100, 100],
        visible: Optional[bool] = True,
        visual_material: Optional[VisualMaterial] = None,
        enable_caustics: bool = False,
        cast_shadows: bool = True
    ) -> None :
        self._device = str(wp.get_cuda_device())
        self._caustics_enabled = False
        self._cast_shadows = False
        if is_prim_path_valid(prim_path):
            prim = get_prim_at_path(prim_path)
            if not prim.IsA(UsdGeom.Mesh):
                raise Exception("The prim at path {} cannot be parsed as a water surface object".format(prim_path))
            self.waterSurfGeom = UsdGeom.Mesh(prim)
        else:
            self.waterSurfGeom = UsdGeom.Mesh.Define(get_current_stage(), prim_path)
            if visual_material is None:
                visual_prim_path = find_unique_string_name(
                    initial_name="/World/Looks/Water", is_unique_fn=lambda x: not is_prim_path_valid(x)
                )
                _ext_id = omni.kit.app.get_app().get_extension_manager().get_extension_id_by_module(__name__)
                omni.kit.commands.execute("CreateMdlMaterialPrim", 
                                  mtl_url=get_extension_path(_ext_id) + "/demo/water_material/Water.mdl", 
                                  mtl_name="Water", 
                                  mtl_path=visual_prim_path
                )
                visual_material = UsdShade.Material.Get(get_current_stage(), Sdf.Path(visual_prim_path))

        # define attributes for the mesh
        points, face_vertex_indices, face_vertex_counts, normals, uvs = create_grid(dim, size)
        
        self.pointsAttr = self.waterSurfGeom.GetPrim().CreateAttribute("points", Sdf.ValueTypeNames.Point3fArray)
        self.pointsAttr = self.waterSurfGeom.CreatePointsAttr(points)
        self.waterSurfGeom.CreateNormalsAttr(normals)
        self.waterSurfGeom.CreateFaceVertexIndicesAttr(face_vertex_indices)
        self.waterSurfGeom.CreateFaceVertexCountsAttr(face_vertex_counts) 
        self.waterSurfGeom.GetPrim().CreateAttribute("primvars:st", Sdf.ValueTypeNames.TexCoord2fArray, False).Set(uvs)
        self.waterSurfGeom.AddRotateXYZOp().Set((90, 0, 0))

        SingleGeometryPrim.__init__(
            self,
            prim_path=prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
            collision=False,
        )
        if visual_material is not None:
            binding_api = UsdShade.MaterialBindingAPI.Apply(self.waterSurfGeom.GetPrim())
            binding_api.Bind(visual_material)
        
        self.profile = wp.zeros(PROFILE_RES, dtype=wp.vec3)
        self.grid = wp.from_numpy(points, dtype=wp.vec3f)
        self.deformed_points = wp.empty_like(self.grid)

        self.set_cast_shadows(cast_shadows)
        if enable_caustics:
            self.set_caustics()
            self._caustics_enabled = True
        return
    
    def deform(self, 
            time : float,
            amplitude : float = 1.0,
            cameraPos : np.ndarray = np.array([0.0, 0.0, 0.0]),
            clipmapCellSize : float = 1.0,
            direction : float = 0.0,
            directionality : float = 0.0,
            scale : float = 1.0,
            waterDepth : float = 50.0,
            windSpeed : float = 10.0
                ):
        
        amplitude = max(0.0001, min(1000.0, amplitude))
        direction = direction % 6.28318530718
        directionality = max(0.0, min(1.0, 0.02 * directionality))
        wind_speed = max(0.0, min(30.0, windSpeed))
        water_depth = max(1.0, min(1000.0, waterDepth))
        scale = min(10000.0, max(0.001, scale))
        
        # create 1D profile buffer for this timestep using wave parameters

        wp.launch(
            kernel=update_profile,
            dim=(PROFILE_RES,),
            inputs=(
                self.profile,
                PROFILE_RES,
                PROFILE_WAVENUM,
                MIN_WAVE_LENGTH,
                MAX_WAVE_LENGTH,
                PROFILE_EXTENT,
                time,
                wind_speed,
                water_depth,
            ),
        )



        # Update point positions using the profile buffer created above
        wp.launch(
            kernel=update_points,
            dim=len(self.grid),
            inputs=(
                self.grid,
                self.profile,
                PROFILE_RES,
                PROFILE_EXTENT * scale,
                amplitude,
                directionality,
                direction,
                cameraPos,
                clipmapCellSize,
            ),
            outputs=(self.deformed_points,),
        )

        self.pointsAttr.Set(Vt.Vec3fArray(self.deformed_points.numpy()))
    

    def set_caustics(self, 
                     photonCountMultiplier: int = 500,
                     photonMaxBounces: int = 4,
                     positionPhi: float = 2.0,
                     normalPhi: float = 0.8,
                     eawFilteringSteps: int = 5):
        settings = carb.settings.get_settings()
        if not self._caustics_enabled:
            settings.set("/rtx/caustics/enabled", True)
            self._caustics_enabled = True

        photonCountMultiplier = np.clip(photonCountMultiplier, 1, 5000)
        photonMaxBounces = np.clip(photonMaxBounces, 1, 20)
        positionPhi = np.clip(positionPhi, 0.1, 50)
        normalPhi = np.clip(normalPhi, 0.3, 1)
        eawFilteringSteps = np.clip(eawFilteringSteps, 0, 10)

        settings.set("/rtx/raytracing/caustics/photonMaxBounces", photonMaxBounces)
        settings.set("/rtx/raytracing/caustics/positionPhi", positionPhi)
        settings.set("/rtx/raytracing/caustics/normalPhi", normalPhi)
        settings.set("/rtx/raytracing/caustics/eawFilteringSteps", eawFilteringSteps)
        settings.set("/rtx/raytracing/caustics/photonCountMultiplier", int(photonCountMultiplier)) # This is the craziest bug I have seen.


    def set_cast_shadows(self, cast_shadows: bool):
        if cast_shadows:
            self.waterSurfGeom.GetPrim().CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(False)
            self.waterSurfGeom.GetPrim().CreateAttribute("primvars:enableShadowTerminatorFix", Sdf.ValueTypeNames.Bool).Set(True)
            self._cast_shadows = True
        else:
            self.waterSurfGeom.GetPrim().CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
            self.waterSurfGeom.GetPrim().CreateAttribute("primvars:enableShadowTerminatorFix", Sdf.ValueTypeNames.Bool).Set(False)
            self._cast_shadows = False


# class FixedCapsule(VisualCapsule):
#     """High level wrapper to create/encapsulate a fixed capsule

#     .. note::

#         Fixed capsules (Capsule shape) have collisions (Collider API) but no rigid body dynamics (Rigid Body API)

#     Args:
#         prim_path (str): prim path of the Prim to encapsulate or create
#         name (str, optional): shortname to be used as a key by Scene class.
#                                 Note: needs to be unique if the object is added to the Scene.
#                                 Defaults to "fixed_capsule".
#         position (Optional[Sequence[float]], optional): position in the world frame of the prim. shape is (3, ).
#                                                         Defaults to None, which means left unchanged.
#         translation (Optional[Sequence[float]], optional): translation in the local frame of the prim
#                                                         (with respect to its parent prim). shape is (3, ).
#                                                         Defaults to None, which means left unchanged.
#         orientation (Optional[Sequence[float]], optional): quaternion orientation in the world/ local frame of the prim
#                                                         (depends if translation or position is specified).
#                                                         quaternion is scalar-first (w, x, y, z). shape is (4, ).
#                                                         Defaults to None, which means left unchanged.
#         scale (Optional[Sequence[float]], optional): local scale to be applied to the prim's dimensions. shape is (3, ).
#                                                 Defaults to None, which means left unchanged.
#         visible (bool, optional): set to false for an invisible prim in the stage while rendering. Defaults to True.
#         color (Optional[np.ndarray], optional): color of the visual shape. Defaults to None, which means 50% gray
#         radius (Optional[float], optional): capsule radius. Defaults to None.
#         height (Optional[float], optional): capsule height. Defaults to None.
#         visual_material (Optional[VisualMaterial], optional): visual material to be applied to the held prim.
#                                 Defaults to None. If not specified, a default visual material will be added.
#         physics_material (Optional[PhysicsMaterial], optional): physics material to be applied to the held prim.
#                                 Defaults to None. If not specified, a default physics material will be added.

#     Example:

#     .. code-block:: python

#         >>> from isaacsim.core.api.objects import FixedCapsule
#         >>> import numpy as np
#         >>>
#         >>> # create a red fixed capsule at the given path
#         >>> prim = FixedCapsule(
#         ...     prim_path="/World/Xform/Capsule",
#         ...     radius=0.5,
#         ...     height=1.0,
#         ...     color=np.array([1.0, 0.0, 0.0])
#         ... )
#         >>> print(prim)
#         <isaacsim.core.api.objects.capsule.FixedCapsule object at 0x7f520c0d4790>
#     """

#     def __init__(
#         self,
#         prim_path: str,
#         name: str = "fixed_capsule",
#         position: Optional[np.ndarray] = None,
#         translation: Optional[np.ndarray] = None,
#         orientation: Optional[np.ndarray] = None,
#         scale: Optional[np.ndarray] = None,
#         visible: Optional[bool] = None,
#         color: Optional[np.ndarray] = None,
#         radius: Optional[np.ndarray] = None,
#         height: Optional[float] = None,
#         visual_material: Optional[VisualMaterial] = None,
#         physics_material: Optional[PhysicsMaterial] = None,
#     ) -> None:
#         if not is_prim_path_valid(prim_path):
#             # set default values if no physics material given
#             if physics_material is None:
#                 static_friction = 0.2
#                 dynamic_friction = 1.0
#                 restitution = 0.0
#                 physics_material_path = find_unique_string_name(
#                     initial_name="/World/Physics_Materials/physics_material",
#                     is_unique_fn=lambda x: not is_prim_path_valid(x),
#                 )
#                 physics_material = PhysicsMaterial(
#                     prim_path=physics_material_path,
#                     dynamic_friction=dynamic_friction,
#                     static_friction=static_friction,
#                     restitution=restitution,
#                 )
#         VisualCapsule.__init__(
#             self,
#             prim_path=prim_path,
#             name=name,
#             position=position,
#             translation=translation,
#             orientation=orientation,
#             scale=scale,
#             visible=visible,
#             color=color,
#             radius=radius,
#             height=height,
#             visual_material=visual_material,
#         )
#         SingleGeometryPrim.set_collision_enabled(self, True)
#         if physics_material is not None:
#             FixedCapsule.apply_physics_material(self, physics_material)
#         return


# class DynamicCapsule(SingleRigidPrim, FixedCapsule):
#     """High level wrapper to create/encapsulate a dynamic capsule

#     .. note::

#         Dynamic capsules (Capsule shape) have collisions (Collider API) and rigid body dynamics (Rigid Body API)

#     Args:
#         prim_path (str): prim path of the Prim to encapsulate or create
#         name (str, optional): shortname to be used as a key by Scene class.
#                                 Note: needs to be unique if the object is added to the Scene.
#                                 Defaults to "dynamic_capsule".
#         position (Optional[Sequence[float]], optional): position in the world frame of the prim. shape is (3, ).
#                                                         Defaults to None, which means left unchanged.
#         translation (Optional[Sequence[float]], optional): translation in the local frame of the prim
#                                                         (with respect to its parent prim). shape is (3, ).
#                                                         Defaults to None, which means left unchanged.
#         orientation (Optional[Sequence[float]], optional): quaternion orientation in the world/ local frame of the prim
#                                                         (depends if translation or position is specified).
#                                                         quaternion is scalar-first (w, x, y, z). shape is (4, ).
#                                                         Defaults to None, which means left unchanged.
#         scale (Optional[Sequence[float]], optional): local scale to be applied to the prim's dimensions. shape is (3, ).
#                                                 Defaults to None, which means left unchanged.
#         visible (bool, optional): set to false for an invisible prim in the stage while rendering. Defaults to True.
#         color (Optional[np.ndarray], optional): color of the visual shape. Defaults to None, which means 50% gray
#         radius (Optional[float], optional): capsule radius. Defaults to None.
#         height (Optional[float], optional): capsule height. Defaults to None.
#         visual_material (Optional[VisualMaterial], optional): visual material to be applied to the held prim.
#                                 Defaults to None. If not specified, a default visual material will be added.
#         physics_material (Optional[PhysicsMaterial], optional): physics material to be applied to the held prim.
#                                 Defaults to None. If not specified, a default physics material will be added.
#         mass (Optional[float], optional): mass in kg. Defaults to None.
#         density (Optional[float], optional): density. Defaults to None.
#         linear_velocity (Optional[np.ndarray], optional): linear velocity in the world frame. Defaults to None.
#         angular_velocity (Optional[np.ndarray], optional): angular velocity in the world frame. Defaults to None.

#     Example:

#     .. code-block:: python

#         >>> from isaacsim.core.api.objects import DynamicCapsule
#         >>> import numpy as np
#         >>>
#         >>> # create a red fixed capsule of mass 1kg at the given path
#         >>> prim = DynamicCapsule(
#         ...     prim_path="/World/Xform/Capsule",
#         ...     radius=0.5,
#         ...     height=1.0,
#         ...     color=np.array([1.0, 0.0, 0.0]),
#         ...     mass=1.0
#         ... )
#         >>> prim
#         <isaacsim.core.api.objects.capsule.DynamicCapsule object at 0x7f4ff915f8e0>
#     """

#     def __init__(
#         self,
#         prim_path: str,
#         name: str = "dynamic_capsule",
#         position: Optional[np.ndarray] = None,
#         translation: Optional[np.ndarray] = None,
#         orientation: Optional[np.ndarray] = None,
#         scale: Optional[np.ndarray] = None,
#         visible: Optional[bool] = None,
#         color: Optional[np.ndarray] = None,
#         radius: Optional[np.ndarray] = None,
#         height: Optional[np.ndarray] = None,
#         visual_material: Optional[VisualMaterial] = None,
#         physics_material: Optional[PhysicsMaterial] = None,
#         mass: Optional[float] = None,
#         density: Optional[float] = None,
#         linear_velocity: Optional[Sequence[float]] = None,
#         angular_velocity: Optional[Sequence[float]] = None,
#     ) -> None:
#         if not is_prim_path_valid(prim_path):
#             if mass is None:
#                 mass = 0.02
#         FixedCapsule.__init__(
#             self,
#             prim_path=prim_path,
#             name=name,
#             position=position,
#             translation=translation,
#             orientation=orientation,
#             scale=scale,
#             visible=visible,
#             color=color,
#             radius=radius,
#             height=height,
#             visual_material=visual_material,
#             physics_material=physics_material,
#         )
#         SingleRigidPrim.__init__(
#             self,
#             prim_path=prim_path,
#             name=name,
#             position=position,
#             translation=translation,
#             orientation=orientation,
#             scale=scale,
#             visible=visible,
#             mass=mass,
#             density=density,
#             linear_velocity=linear_velocity,
#             angular_velocity=angular_velocity,
#         )
