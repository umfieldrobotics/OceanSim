# This file is to fix an artifact from the shader exported from unreal engine USD plugin
# in which the material's opacity mask does not appear transparent.
# This can be fixed by accessing the BRDF shader in the exported material and set 
# opacity_threshold to be non-zero.


import argparse
import os

# Check if there are any config files (yaml or json) are passed as arguments
parser = argparse.ArgumentParser()
parser.add_argument("--assets", required=True, help="Path to the texture folder")
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

from pxr import Sdf, UsdShade, Usd, UsdGeom, Gf, UsdPhysics, UsdLux
import os
import omni.usd




def main():
    def fix_opacity_threshold(url) -> None:
        nonlocal fixed_num
        CHANGE = False
        # A script to set the opacity threshold of all the shaders to 0.5
        omni.usd.get_context().open_stage(url)
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
                    # NOTE: This should correctly give the BRDF shader (not unreal BRDF nor the output shader)

                    # Get the 'opacityThreshold' input
                    opacity_threshold_input = shader.GetInput("opacityThreshold")
                    # If the input exists, set its value
                    if opacity_threshold_input:
                        opacity_threshold_input.Set(0.5)
                        print(f"  > Set 'opacityThreshold' to {0.5}")
                        CHANGE = True
                    else:
                        # If the input doesn't exist, create it and set the value
                        shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.5)
                        print(f"  > Created and set 'opacityThreshold' to {0.5}")
                        CHANGE = True

        if CHANGE:
            saved = omni.usd.get_context().save_stage()
            if saved:
                print('Fixed usd file saved.')
                fixed_num += 1
            else:
                print('Fixed usd file failed to be saved.')
        else:
            print('Nothing changed in this file. No need to save any thing.')
    
    
    assets_url = args.assets
    fixed_num = 0
    idx = 0
    for fname in os.listdir(assets_url):
        fpath = os.path.join(assets_url, fname)
        if os.path.isfile(fpath) and fname.lower().endswith(('.usd', '.usdc', '.usda', '.usdz')):
            # Just fix the material usd should be fine    
            if "MI_" in fname:
                print(f'[{idx}] Fixing {fpath}.')
                fix_opacity_threshold(fpath)
                idx += 1

    print(f"Total {idx} number of MI_USD detected and {fixed_num} are fixed in {assets_url}.")



main()
simulation_app.close()