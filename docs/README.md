# Extension Install
1. Install Omniverse launcher
2. Install Isaac Sim 4.2.0 in Omniverse Launcher
   - If you stick to Nvidia's tutorial, you can also download Omniverse Cache (managing cache) and Nucleus Navigator (managing assets). 
3. Open the software once for it to setup the environment automatically
   - To check if your Isaac Sim is installed properly, **make sure no any $${\color{red}Errors}$$ in terminals**, $${\color{orange}Warnings}$$ are fine.
4. Clone this repo to  
   - **/{path-to-your-isaac-sim}/isaac-sim-4.2.0/extsUser**
     - Mine installed defaultly on newest Ubuntu is:  
      /home/haoyu-ma/.local/share/ov/pkg/isaac-sim-4.2.0/extsUser
6. To enable this extension, go to **Window-Extensions**, and search for **OceanSim**.  
   On the right, click the enable **switch** and **AUTOLOAD**

# Usage
On top tool bar you can see the OceanSim menu, and hover on it you can see the contained utilities.  
For now it only contains a **DVL** and **more** (template for adding more utilities)

In the repo, the folder **demo_usd** contains my demo scenes (.usd) and assets in proposal.   
**I haven't linked them to the UI.** But before opening it directly through **File-Open**, turn on the following extensions:
  - Mimic real ocean deformation
    - omni.warp.core
    - omni.warp
  - Apply forcings to collective object
    - omni.physx.forcefields
    - omni.usd.schema.forcefield

I would expect some assets linking error if opened the above .usd files. I never tested on other machines.  
Let me know.  
They exclusively implemented through blueprint, so no code body included.
