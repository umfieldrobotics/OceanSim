import math
import os
import random
from itertools import chain
from collections import defaultdict
import json
from pathlib import Path
from collections import OrderedDict
from typing import Optional, List

import omni.kit.app
import omni.kit.commands
import omni.physx
import omni.replicator.core as rep
import omni.timeline
import omni.usd
import carb.settings
from isaacsim.core.utils.semantics import  remove_labels, add_labels, upgrade_prim_semantics_to_labels
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdLux
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.transformations import get_relative_transform



def get_material_prims(parent_prim: Usd.Prim) -> list[Usd.Prim]:
    """Recursively search for material prims under the given parent prim."""
    material_prims = []
    for child in parent_prim.GetChildren():
        if child.IsA(UsdShade.Material):
            material_prims.append(child)
        else:
            material_prims.extend(get_material_prims(child))
    return material_prims


def get_UsdUVTexture_shaders(parent_prim: Usd.Prim) -> list[Usd.Prim]:
    """Recursively search for UsdUVTexture shader prims under the given parent prim."""
    uv_texture_shaders = []
    for child in parent_prim.GetChildren():
        if child.IsA(UsdShade.Shader) and child.GetAttribute("info:implementationSource").Get() == "id" and child.GetAttribute("info:id").Get() == "UsdUVTexture":
            uv_texture_shaders.append(UsdShade.Shader(child))
            UsdShade.Shader(child).CreateInput("bias", Sdf.ValueTypeNames.Float4)
            UsdShade.Shader(child).CreateInput("scale", Sdf.ValueTypeNames.Float4)

    return uv_texture_shaders
stage = omni.usd.get_context().get_stage()
mPrims = get_material_prims(stage.GetDefaultPrim())
uv_tex_shaders = get_UsdUVTexture_shaders(mPrims[0])
for shader in uv_tex_shaders:
    print(shader.GetPath())
    shader.GetInput("bias").Set((0.1, 0.1, 0.1, 0.0))
    shader.GetInput("scale").Set((1.0, 1.0, 1.0, 1.0))
