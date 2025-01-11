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
        self._dvl = None
        self._articulation = None

        self._dc = _dynamic_control.acquire_dynamic_control_interface()

        self._running_scenario = False
        self._time = 0.0
        
        self._vel_buffer = []
        self._time_buffer = []


    def setup_scenario(self, rob, dvl, articulation):
        self._rob = rob
        self._dvl = dvl
        self._articulation = articulation

        self._running_scenario = True




    def teardown_scenario(self):
        self._rob = None
        self._articulation = None
        self._dvl = None

        self._running_scenario = False
        self._time = 0.0


    def update_scenario(self, step: float):
        if not self._running_scenario:
            return

        self._time += step
        
        rob_body = self._dc.get_rigid_body("/rob")
        if (self._time < 1):
            self._dc.apply_body_force(rob_body, carb.Float3(0.1,0,0), carb.Float3(0,0,0), 0)


        self._time_buffer.append(self._time)
        self._vel_buffer.append(self._dvl.get_linear_vel())
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

    


