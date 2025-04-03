# OceanSim Installation Documentation
We design OceanSim as an extension package for NVIDIA Isaac Sim. This design allows better integration with Isaac Sim and users can pair OceanSim with other Isaac Sim extensions. This document provides a step-by-step guide to install OceanSim.

## Prerequisites
OceanSim does not enforce any additional prerequisites beyond those required by Isaac Sim. Please refer to the [official Isaac Sim documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html#system-requirements) for the prerequisites.

OceanSim is developed with Isaac Sim 4.5. Due to the changes in Isaac Sim 4.5 compared to previous versions, OceanSim may not work with older versions of Isaac Sim.

We have tested OceanSim on Ubuntu 20.04, 22.04, and 24.04. We have also tested OceanSim using various GPUs, including NVIDIA RTX 3090, RTX A6000, and RTX 4080 Super.

## Installation
Install NVIDIA Isaac Sim 4.5. We follow the official [workstation installation guide](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html) to install Isaac Sim.

Clone this repository to your local machine. We recommend cloning the repository to the Isaac Sim workspace directory.
```bash
cd /path/to/isaacsim/extsUser
git clone https://github.com/umfieldrobotics/OceanSim.git
```

Download `OceanSim_assets` from [Google Drive](https://drive.google.com/drive/folders/1qg4-Y_GMiybnLc1BFjx0DsWfR0AgeZzA?usp=sharing) which contains USD assets of robot and environment.

And change function `get_oceansim_assets_path()` in [~/isaacsim/extsUser/OceanSim/isaacsim/oceansim/utils/assets_utils.py](../../isaacsim/oceansim/utils/assets_utils.py) to return the path to the installed assets folder.
```bash
def get_oceansim_assets_path() -> str:
    return "/path/to/downloaded/assets/OceanSim_assets"
```

Launch Isaac Sim following this [guide](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html#isaac-sim-short-app-selector).

## Launching OceanSim
There is no separate building process needed for OceanSim, as it is an extension. To load OceanSim: 
- IsaacSim, follow `Window -> Extensions`
- On the window that shows up, remove the `@feature` filter that comes by default
- Activate `OCEANSIM`
- You can now exit the `Extensions` window, and OceanSim should be an option on the IsaacSim panel
