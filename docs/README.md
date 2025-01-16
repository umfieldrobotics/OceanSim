# Progress
**Dynmics** 
- Implemented the exact same Torpedo AUV fossen example in Issac Sim environment. Comparsion of the simulation result is saved in demo\fossen folder. 

**Sensor**
- DVL extension(finished, haven't well encapsulated): Being able to read gt velocity, measure the distance return of four beams. Noise and uncertainty corrections are from Holoocean.

- Imaging Sonar extention(in progress): 
Developed based on [paper](https://ieeexplore.ieee.org/document/7404349). 
Being able to return point cloud, scan info (fov, resolution), intensity (**problem now**: $${\color{red}always\space gives\space full\space intensity\space 255\space upon\space hitting\space, need\space intermediate\space value\space 0\space to\space 255\space to\space get\space image\space noise\space}$$)

- Sidescan Sonar (haven't started)
- Baro (haven't started)

**Graphics\render**  
(haven't started)
# Extension Install
1. Install Omniverse launcher
2. Install Isaac Sim 4.2.0 in Omniverse Launcher
   - If you stick to Nvidia's tutorial, you can also download Omniverse Cache (managing cache) and Nucleus Navigator (managing assets). 
3. Open the software once for it to setup the environment automatically
   - To check if your Isaac Sim is installed properly, **make sure no any $${\color{red}Errors}$$ in terminals**, $${\color{orange}Warnings}$$ are fine.
4. Clone this repo to  
   - **/{path-to-your-isaac-sim}/isaac-sim-4.2.0/extsUser**
     - Mine installed defaultly on 24.04 **Ubuntu** is:  
      /home/haoyu-ma/.local/share/ov/pkg/isaac-sim-4.2.0/extsUser
6. To enable this extension, go to **Window-Extensions**, and search for **OceanSim**.  
   On the right, click the enable **switch** and **AUTOLOAD**

# Usage
On top tool bar you can see the OceanSim menu, and hover on it you can see the contained utilities.  

In the repo, the folder **demo\usd_scenes** contains my demo scenes (.usd) and assets in proposal.   
**I haven't linked them to the UI.** But before opening it through **File-Open**, turn on the following extensions:
  - Mimic real ocean deformation
    - omni.warp.core
    - omni.warp
  - Apply forcings to collective object
    - omni.physx.forcefields
    - omni.usd.schema.forcefield

I would expect some assets linking error. I never tested on other machines.  
Let me know.  
They are exclusively implemented through blueprint, so no code body included.
