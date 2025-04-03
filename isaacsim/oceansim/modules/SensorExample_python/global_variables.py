# Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

EXTENSION_TITLE = "Sensor Example"

EXTENSION_DESCRIPTION = "This is unified example demonstrating various OceanSim UW sensors." \
"User should click 'Open Source Code' icon located on the extension title for any sensor parameter tunning, \n" \
'eg: water surface height, sonar fov, camera rendering parameters..., and get consulted to develop own digital twins. \n' \
"User may test this demo on their own scene by copying the USD file path to 'Path to USD',\n" \
"otherwise a default one is loaded.\n" \
"For DVL sensor, scene has to be toggled with static collider for beam interaction! \n" \
'Manual control: w:x+, s:x-, a:y+, d:y-, up:z+, down:z-, \n' \
'i:pitch-, k:pitch+, j:yaw+, l:yaw-, left:row-, right:row+. \n' \
'Straight line: Robot will travel to local x direction with v=0.5m/s. \n' \
'No control: Robot will remain static.'


EXTENSION_LINK = "https://umfieldrobotics.github.io/OceanSim/"