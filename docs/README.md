# Progress
**OceanSim for Issac Sim 4.5.0**
- **Forward looking imaging sonar**.
  - Based on Omniverse replicator to achieve high performance scene query
  - Gaussian and range Rayleigh noise are added
  - Simple interface based on semantics segmentation to acheive material property configuration

**OceanSim for Issac Sim 4.2.0**
- **Sensor**
   - DVL extension: Read gt velocity, measure the distance return of four beams. Noise and uncertainty corrections are from Holoocean.
   - Next step: further characterize noise based on range readings from beam range sensor.
- **Graphics\render**  
   - Caustics example is saved in demo\usd_scenes
   - Next step: implementing slide bar tools for users to pick visually accurate underwater image formation model parameters.
- **Fossen Dynmics** 
   - Implemented the same Holoocean Torpedo AUV fossen example in Issac Sim environment. Comparsion of the simulation result is saved in demo\fossen folder. 

# Extension Install 
**4.2.0**
1. Install Omniverse launcher
2. Install Isaac Sim 4.2.0 in Omniverse Launcher
   - If you stick to Nvidia's tutorial, you can also download Omniverse Cache (managing cache) and Nucleus Navigator (managing assets). 
3. Open the software once for it to setup the environment automatically
   - To check if your Isaac Sim is installed properly, **make sure no any $${\color{red}Errors}$$ in terminals**, $${\color{orange}Warnings}$$ are fine.
4. Clone this repo to  
   - **/{path-to-your-isaac-sim}/isaac-sim-4.2.0/extsUser**
     - Mine installed defaultly on 24.04 **Ubuntu** is:  
      /home/haoyu-ma/.local/share/ov/pkg/isaac-sim-4.2.0/extsUser
5. To enable this extension, go to **Window-Extensions**, and search for **OceanSim**.  
   On the right, click the enable **switch** and **AUTOLOAD**

**4.5.0**
This version no longer needs the launcher.
After you install issac sim, you can directly skip to step 4.


# Usage
On top tool bar you can see the OceanSim menu, and hover on it you can see the contained utilities.  

<!-- 
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
-->