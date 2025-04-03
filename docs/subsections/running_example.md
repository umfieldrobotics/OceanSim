# Run Examples in OceanSim
In this document, we will provide simple guidelines for using existing features in OceanSim and Nvidia Isaac Sim to facilitate building underwater digital twins in our framework. The following are explanations of the examples that you can run:

## Sensor Example
OceanSim provides an example also formatted as an extension to demonstrate the usage of underwater sensors and modify their parameters.

Navigate to `OceanSim - Examples - Sensor Example` to open the module. Select the sensors you wish to simulate and point the "Path to USD" to your own USD scene or the example MHL scene in the `OceanSim_assets` directory.

The module provides self-explanatory UI in which you can choose which sensor to use and corresponding data visualization will be automatically available. User may test this module in their own USD scenes otherwise a default one is used. 

We do not recommend user to perform digital twin experiments on this extension. This is example involves boilerplate code which is only for demonstration purposes.
### Instructions
For more instructions when using this example, refer to the following, which you can also find in the [information panel](../../isaacsim/oceansim/modules/SensorExample_python/global_variables.py) in the extension UI:
- This is a unified example that demonstrates various OceanSim UW sensors.
- For users interested in making edits to the sensor parameters, click on the `Open Source Code` icon, which also looks like the "edit" icon. This will bring up the source code where many parameters can be edited, such as:
  - water surface height
  - sonar fov
  - camera rendering parameters
  - guides on developing your own digital twins.
- User can test this demo on their own scene by copying the USD file path to `Path to USD`, otherwise a default one is loaded.
- For DVL sensor, scene has to be toggled with static collider for beam interaction!
- Manual control:

<div align="center">

| Key      | Control |
|----------|---------|
| W/w      | +x      |
| S/s      | -x      |
| A/a      | +y      |
| D/d      | -y      |
| Up key   | +z      |
| Down key | -z      |
| I/i      | +pitch  |
| K/k      | -pitch  |
| J/j      | +yaw    |
| L/l      | -yaw    |
| Left key | +roll   |
| Right key| -roll   |

</div>

- Automatic Control:
  - `Straight line`: Robot will travel to local x direction with v=0.5m/s
- No control: Robot will remain static.

### Running the example
To run the example after selecting the sensors and their configurations, first `Load`, and then click `Run`.

## Color Picker
OceanSim provides a handy UI tool to accelerate the process of recreating underwater column effects similar to the robot's actual working environment by selecting the appropriate image formation parameters ([Akkaynak, Derya, and Tali Treibitz. "A revised underwater image formation model"](https://ieeexplore.ieee.org/document/8578801).)

Navigate to `OceanSim - Color Picker` to open the module.

This widget allows the user to visualize the rendered result in any USD scene while tuning parameters in real time. 

For more instructions when using this example, refer to [information panel](../../isaacsim/oceansim/modules/colorpicker_python/global_variables.py) in the extension UI.

## Tuning Object Reflectivity for Imaging Sonar
The user can adjust the reflectivity of objects in the sonar perception by adding a semantic label to the object. 

The semantic type must be `"reflectivity"` as a string. 
And corresponding semantic data must be float, eg. `0.2`.

Semantic configuration can either be performed by code during scene setup:
<!-- configure Prim Semantics by code -->
```bash
from isaacsim.core.utils.semantics import add_update_semantics
add_update_semantics(prim=<object_prim>,
                    type_label='reflectivity',
                    semantic_label='1.0')
```
Or with UI provided in `semantics.schema.editor` ([Semantic Schema Editor](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/semantics_schema_editor.html) should be auto loaded as Isaac Sim starts up). 

A simple tutorial is as follows:
<!-- (../../media/semantic_editor.gif) -->
![Add reflectivity by Semantic Editor](../../media/semantic_editor.gif)

As demonstrated by this workflow, developers are freely to add more modeling parameters as a new semantic type to improve sonar fidelity.  

## Adding Water Caustics
Note that the addition of water caustics into the USD scene is still under development and thus may lead to performance issues and crash during the simulation.

To turn on rendering caustics, `Render Settings - Ray Tracing - Caustics` will be set `on`, and `Enable Caustics` in the UsdLux that supports caustics will be set `on` for the light source.

Next we assign `transparent materials` (eg. Water, glass) to any mesh surface that we wish to [deflect photons](https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-2-rendering-water-caustics) and create caustics.

Lastly to simulate water caustics, we will deform the surface according to realistic water surface deformation.

A USD file containing the caustic settings and surface deformation powered by a Warp kernel can be found in the OceanSim assets `~\OceanSim_assets\collected_MHL\mhl_water.usd` we published. 

And the corresponding demo video is provided below:

<!-- (../../media/caustics.gif) -->
![How to turn on Caustics](../../media/caustics.gif)







