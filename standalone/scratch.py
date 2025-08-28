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





