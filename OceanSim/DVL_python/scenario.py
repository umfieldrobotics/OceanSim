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
from ..utils.MultivariateNormal import MultivariateNormal
import omni.ui as ui
from omni.isaac.ui.ui_utils import get_style
from omni.isaac.ui.element_wrappers import XYPlot




class DVLScenario(ScenarioTemplate):
    def __init__(self):
        self._rob = None
        self._ls = None
        self._beam_paths = None
        self._articulation = None

        self._dc = _dynamic_control.acquire_dynamic_control_interface()

        self._elevation = None
        self._vel_sigma = None

        self._running_scenario = False
        self._time = 0.0

        self._mvn_vel = MultivariateNormal(4)
        
        self._vel_buffer = []
        self._time_buffer = []


    def setup_scenario(self, rob, ls, beam_paths, articulation):
        self._rob = rob
        self._ls = ls
        self._beam_paths = beam_paths
        self._articulation = articulation

        self._elevation = 22.5 # deg
        self._vel_cov = np.array([1,2,3])

        self._mvn_vel.uncertain = True
        self._mvn_vel.init_cov(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
        

        sinElev = np.sin(np.deg2rad(self._elevation))
        cosElev = np.cos(np.deg2rad(self._elevation))
        self._transform = np.array([[1/(2*sinElev), 0, -1/(2*sinElev), 0],
                                    [0, 1/(2*sinElev), 0, -1/(2*sinElev)],
                                    [1/(4*cosElev), 1/(4*cosElev), 1/(4*cosElev), 1/(4*cosElev)]
                                    ])

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

        depth = []
        hit_pos = []
        beam_hit = []
        for beam_path in self._beam_paths:
            depth.append(self._ls.get_linear_depth_data(beam_path).squeeze())
            hit_pos.append(self._ls.get_hit_pos_data(beam_path).squeeze())
            beam_hit.append(self._ls.get_beam_hit_data(beam_path).astype(bool).squeeze())
        
        rob_body = self._dc.get_rigid_body("/rob")
        if (self._time < 1):
            self._dc.apply_body_force(rob_body, carb.Float3(0.1,0,0), carb.Float3(0,0,0), 0)
        vel = self._dc.get_rigid_body_linear_velocity(rob_body)

        if (self._mvn_vel.is_uncertain()):
            sample = self._mvn_vel.sample_array()
            for i in range(4):
                for j in range(3):
                    vel[j] += self._transform[j][i] * sample[i] 

        self._time_buffer.append(self._time)
        self._vel_buffer.append(vel)
        self._time_buffer = self._time_buffer[-25:]
        self._vel_buffer = self._vel_buffer[-25:]

    def update_ui(self,ui_frame):

        with ui_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):

                plot = XYPlot(
                    "DVL readings",
                    tooltip="Press mouse over the plot for data label",
                    x_data=[self._time_buffer, self._time_buffer, self._time_buffer],
                    y_data=[list(arr) for arr in np.array(self._vel_buffer).T ],
                    x_min=None,  # Use default behavior to fit plotted data to entire frame
                    x_max=None,
                    y_min=-10,
                    y_max=10,
                    x_label="step",
                    y_label="vel [m/s]",
                    # plot_height=10,
                    legends=["X", "Y", "Z"],
                    show_legend=True,
                    plot_colors=[
                        [255, 0, 0],
                        [0, 255, 0],
                        [0, 100, 200],
                    ],  # List of [r,g,b] values; not necessary to specify
                )

    


