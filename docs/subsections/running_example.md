# Run Examples in OceanSim
In this document, we will provide simple guidelines in using existing features in OceanSim and Nvidia Isaac Sim that faciliate building underwater digital twins in our framework.

## Sensor Example
OceanSim provides an example also formatted as an extension to demonstrate the usage of underwater sensors and modify their parameters.

Navigate to `OceanSim - Examples - Sensor Example` to open the module.

The module provides self-explanatory UI in which you can choose which sensor to use and corresponding data visualization will be automatically available. User may test this module in their own USD scenes otherwise a default one is used. 

We do not recommend user to perform digital twin experiment on this extension. This is an example involves boilerplate code and less performent, which is only for demonstration purpose.

For more instructions when using this example, refer to [information panel](../../isaacsim/oceansim/modules/SensorExample_python/global_variables.py) in the extension UI.

## Color Picker
OceanSim provies a handy UI tool to accelerate the process of recreating underwater column effects similar to the robot's actual working environment by selecting the appropriate image formation parameters ([Akkaynak, Derya, and Tali Treibitz. "A revised underwater image formation model"](https://ieeexplore.ieee.org/document/8578801).)

Navigate to `OceanSim - Color Picker` to open the module.

This widget allows user to visualize the rendered result in any USD scene while tunning parameters in real time. 

For more instructions when using this example, refer to [information panel](../../isaacsim/oceansim/modules/colorpicker_python/global_variables.py) in the extension UI.

## Tuning Object Reflectivity for Imaging Sonar
User can adjust reflectivity of objects in the sonar perception via adding semantic label to the object. 

Semantic type must be `reflectivity` as string. 
And corresponding semantic data must be float, eg. `0.2`.

Semantic configuration can either be performed by code:
<!-- configure Prim Semantics by code -->
```bash
from isaacsim.core.utils.semantics import add_update_semantics
add_update_semantics(prim=<object_prim>,
                    type_label='reflectivity',
                    semantic_label='1.0')
```
Or with UI provided in `semantics.schema.editor` ([Semantic Schema Editor](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/semantics_schema_editor.html) should be auto loaded as Isaac Sim starts up). A simple tutorial is as followed:
<!-- (../../media/semantic_editor.gif) -->
![Add reflectivity by Semantic Editor](../../media/semantic_editor.gif)

## Adding Caustics from Wave Deformation
Notice the below way of adding water caustics into the USD scene is still in exploration and thus may lead to performance issud and crash during the simulation.

To turn on rendering water caustics, `Render Settings - Ray Tracing - Caustics` will be set `on`, and `Enable Caustics` in the UsdLux that supports caustics will be set `on` for the light source.

Next we assign `transparent materials` (eg. Water, glass) to any mesh surface that we wish to [deflect photons](https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-2-rendering-water-caustics) and create caustics.

Lastly to simulate water caustics, we will deform the surface according to realistic water surface deformation.

A USD file containing the caustic settings and surface deformation powered by a Warp kernel can be found in the [OceanSim assets]()`(TODO)` we published. And the corresponding demo video is provided below:

<!-- (../../media/caustics.gif) -->
![How to turn on Caustics](../../media/caustics.gif)







