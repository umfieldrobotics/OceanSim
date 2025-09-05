# OceanSim Installation Documentation
We design OceanSim as an extension package for NVIDIA Isaac Sim. This design allows better integration with Isaac Sim and users can pair OceanSim with other Isaac Sim extensions. This document provides a step-by-step guide to install OceanSim.

## Prerequisites
OceanSim does not enforce any additional prerequisites beyond those required by Isaac Sim. Please refer to the [official Isaac Sim documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html#system-requirements) for the prerequisites.

OceanSim is now compatible with Isaac Sim 5.0. Due to the changes in Isaac Sim 5.0 compared to previous versions, OceanSim main branch release may not work with older versions of Isaac Sim.

We have tested OceanSim on Ubuntu 20.04, 22.04, and 24.04. We have also tested OceanSim using various GPUs, including NVIDIA RTX 3090, RTX A6000, and RTX 4080 Super, TX 5070Ti. 

## Installation
For Isaac Sim 5.0, we build from their [source code](https://github.com/isaac-sim/IsaacSim).



Clone this repository to your local machine. We recommend cloning the repository to the Isaac Sim workspace directory.
```bash
cd /path/to/isaacsim/extsUser
git clone https://github.com/umfieldrobotics/OceanSim.git
```
`/extsUser` folder is guaranteed that the extension is discoverable in the extension browser of Isaac Sim.

Download `OceanSim_assets` from [Google Drive](https://drive.google.com/drive/folders/1qg4-Y_GMiybnLc1BFjx0DsWfR0AgeZzA?usp=sharing) which contains USD assets of robot and environment.

Then, run the following to configure OceanSim to point to your asset path:

```bash
cd /path/to/OceanSim
python3 config/register_asset_path.py /path/to/OceanSim_assets
```
For Isaac Sim 4.5, we follow the official [workstation installation guide](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html).

**NOTE**: The main branch is always the latest release and does not have backward compatibility due to Omniverse being a fast evolving ecosystem. 
Please download previous release and the installation is exactly the same as above.

## Launching OceanSim
There is no separate building process needed for OceanSim, as it is an extension. To load OceanSim: 
- IsaacSim, follow `Window -> Extensions`
- On the window that shows up, remove the `@feature` filter that comes by default
- Activate `OCEANSIM`
- You can now exit the `Extensions` window, and OceanSim should be an option on the IsaacSim panel. You can freely import OceanSim sensors and modules into your own Isaac Sim workflow.
