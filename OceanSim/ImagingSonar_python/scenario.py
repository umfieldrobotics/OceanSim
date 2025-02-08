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


# ImagingSonar scenario implementation
import numpy as np
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.core.prims import XFormPrim

class ImagingSonarScenario(ScenarioTemplate):
    def __init__(self):
        self._rob = None
        self._sonar = None

        self._running_scenario = False
        self._time = 0.0
        self._output_dir = "/home/haoyu-ma/Desktop/_sonar_data"




    def setup_scenario(self, rob, sonar):
        self._rob = rob
        self._sonar = sonar        
        self._sonar.initialize(self._output_dir)

        self._running_scenario = True




    def teardown_scenario(self):
        self._rob = None
        if self._sonar is not None:
            self._sonar.close()
        
        self._running_scenario = False
        self._time = 0.0



    def update_scenario(self, step: float):
        if not self._running_scenario:
            return
        self._time += step
        self._sonar.scan()
        self._sonar.make_sonar_data()
