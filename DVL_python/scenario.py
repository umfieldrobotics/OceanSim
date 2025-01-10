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


# DVL scenario implementation
import numpy as np
from omni.isaac.dynamic_control import _dynamic_control
import carb




class DVLScenario(ScenarioTemplate):
    def __init__(self):
        self._rob_xform = None
        self._ls = None
        self._beam_paths = None
        self._articulation = None

        # self._rob_frequency = 0.25  # Hz
        self._dc = _dynamic_control.acquire_dynamic_control_interface()


        self._running_scenario = False
        self._time = 0.0



    def setup_scenario(self, rob, ls, beam_paths, articulation):
        self._rob = rob
        self._ls = ls
        self._beam_paths = beam_paths
        self._articulation = articulation

        # self._initial_rob_position = self._rob_xform.get_world_pose()[0]
        # self._initial_rob_phase = np.arctan2(self._initial_rob_position[1], self._initial_rob_position[0])
        # self._rob_radius = np.linalg.norm(self._initial_rob_position[:2])

        self._running_scenario = True




    def teardown_scenario(self):
        self._rob = None
        self._ls = None
        self._beam_paths = None
        self._articulation = None


        self._running_scenario = False
        self._time = 0.0


    def update_scenario(self, step: float):
        if not self._running_scenario:
            return

        self._time += step

        # x = self._rob_radius * np.cos(self._initial_rob_phase + self._time * self._rob_frequency * 2 * np.pi)
        # y = self._rob_radius * np.sin(self._initial_rob_phase + self._time * self._rob_frequency * 2 * np.pi)
        # z = self._initial_rob_position[2]

        # self._rob_xform.set_world_pose(np.array([x, y, z]))

        depth = []
        hit_pos = []
        beam_hit = []
        for beam_path in self._beam_paths:
            depth.append(self._ls.get_linear_depth_data(beam_path).squeeze())
            hit_pos.append(self._ls.get_hit_pos_data(beam_path).squeeze())
            beam_hit.append(self._ls.get_beam_hit_data(beam_path).astype(bool).squeeze())
        
        rob_body = self._dc.get_rigid_body("/rob")
        if (self._time < 1):
            self._dc.apply_body_force(rob_body, carb.Float3(0.1,0,0),carb.Float3(0,0,0), 0)
        vel = self._dc.get_rigid_body_linear_velocity(rob_body)
        print(vel)


    