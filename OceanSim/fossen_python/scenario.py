# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#


class ScenarioTemplate:
    def __init__(self):
        pass

    def setup_scenario(self):
        pass

    def teardown_scenario(self):
        pass

    def update_scenario(self):
        pass


import numpy as np
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.dynamic_control import _dynamic_control
from .vehicle_dynamics import *
from .dynamics import *
"""
This scenario takes in a robot Articulation and makes it move through its joint DOFs.
Additionally, it adds a cuboid prim to the stage that moves in a circle around the robot.

The particular framework under which this scenario operates should not be taken as a direct
recomendation to the user about how to structure their code.  In the simple example put together
in this template, this particular structure served to improve code readability and separate
the logic that runs the example from the UI design.
"""


class FossenScenario(ScenarioTemplate):
    def __init__(self):
        self._rob_prim_path = None
        self._articulation = None

        self._dc = _dynamic_control.acquire_dynamic_control_interface()

        self._running_scenario = False
        self._time = 0.0
        
        # Fossen scenario config
        self._ticks_per_sec = 20
        initial_location = [0,0,-10] #Translation in NWU coordinate system
        initial_rotation = [0,0,0] #Roll, pitch, Yaw in Euler angle order ZYX and in degrees NWU coordinate system

        self._scenario_config = {
        "name": "torpedo_dynamics",
        "package_name": "Ocean",
        "world": "OpenWater",
        "main_agent": "auv0",
        "ticks_per_sec": self._ticks_per_sec,
        "agents": [
            {
                "agent_name": "auv0",
                "agent_type": "TorpedoAUV",
                "sensors": [
                    {
                        "sensor_type": "DynamicsSensor",
                        "configuration": {
                            "UseCOM": True,
                            "UseRPY": False  # Use quaternion for dynamics
                        }
                    },
                ],
                "control_scheme": 1,  # Control scheme 1 is how custom dynamics are applied to TAUV
                "location": initial_location,
                "rotation": initial_rotation,
                "dynamics": 
                    {
                        "mass": 16,
                        "length": 1.6,
                        "rho": 1026,
                        "diam": 0.19,
                        "r_bg": [0, 0, 0.02],
                        "r_bb": [0, 0, 0],
                        "r44": 0.3,
                        "Cd": 0.42,
                        "T_surge": 20,
                        "T_sway": 20,
                        "zeta_roll": 0.3,
                        "zeta_pitch": 0.8,
                        "T_yaw": 1,
                        "K_nomoto": 5.0 / 20.0
                    },
                "actuator": 
                    {
                        "fin_area": 0.00665,
                        "deltaMax_fin_deg": 15,
                        "nMax": 1525,
                        "T_delta": 0.1,
                        "T_n": 0.1,
                        "CL_delta_r": 0.5,
                        "CL_delta_s": 0.7
                    },
                "autopilot": 
                    {
                        'depth': {
                            'wn_d_z': 0.2,
                            'Kp_z': 0.08,
                            'T_z': 100,
                            'Kp_theta': 4.0,
                            'Kd_theta': 2.3,
                            'Ki_theta': 0.3,
                            'K_w':  5.0,
                        },
                        'heading': {
                            'wn_d': 1.2,
                            'zeta_d': 0.8,
                            'r_max': 0.9,
                            'lam': 0.1,
                            'phi_b': 0.1,
                            'K_d': 0.5,
                            'K_sigma': 0.05,
                        }
                    },
                
                }
            ]
        }
        

    def setup_scenario(self, articulation, rob_prim_path):
        self._articulation = articulation
        self._rob_prim_path = rob_prim_path
        vehicle = fourFinDep(scenario=self._scenario_config,
                             vehicle_name='auv0',
                             controlSystem='manualControl')
        ticks_per_second = 25 
        period = 1.0/ticks_per_second
        self._torpedo_dynamics = FossenDynamics(vehicle=vehicle, 
                                          sample_period=period)
        self._running_scenario = True


    def teardown_scenario(self):
        self._time = 0.0


    def update_scenario(self, step: float):
        if not self._running_scenario:
            return

        self._time += step
        # Use the top fin rigid body as measurement of dynamical properties
        rob_body = self._dc.get_rigid_body(self._rob_prim_path + "/Object_27/mesh_0/Geometry") 
        # pose = rob_body.Transform.pose
        # print(pose)

