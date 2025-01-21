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
import carb
import omni.ui as ui
import matplotlib.pyplot as plt


class ImagingSonarScenario(ScenarioTemplate):
    def __init__(self):
        self._rob = None
        self._sonar = None
        self._dc = _dynamic_control.acquire_dynamic_control_interface()
        self._running_scenario = False
        self._time = 0.0



    def setup_scenario(self, rob, sonar):
        self._rob = rob
        self._sonar = sonar
        self._running_scenario = True




    def teardown_scenario(self):
        self._rob = None
        self._sonar = None

        self._running_scenario = False
        self._time = 0.0


    def update_scenario(self, step: float):
        self._sonar.clear_debug_lines()
        
        if not self._running_scenario:
            return

        self._time += step
        self._sonar.ray_cast()
        self._sonar.draw_debug_lines()






    def plot(self):
        saved_path = '/home/haoyu-ma/Desktop'

        np.save(saved_path+'/cart_coord.npy', self.pcl_cart)
        np.save(saved_path+'/sonar_map.npy', self.sonar_map)
        import matplotlib.pyplot as plt 

        plt.figure()
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        colors = plt.cm.viridis(self.sonar_map[:,2].flatten())
        ax.scatter(self.pcl_cart[:,0], self.pcl_cart[:,1], self.pcl_cart[:,2], c=colors)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_xlim([0,5])
        fig.set_figwidth(10)
        fig.set_figheight(10)
        plt.grid(True)
        plt.savefig(saved_path + '/pcl_colored.png')
 
        
        
        plt.figure()
        fig, ax = plt.subplots()
        filterd_sonar_map = self.sonar_map[self.sonar_map[:,2]!=0]
        ax.scatter(filterd_sonar_map[:,0], filterd_sonar_map[:,1])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        fig.set_figwidth(10)
        fig.set_figheight(10)
        plt.grid(True)
        plt.savefig(saved_path + '/sonar.png')


        print(f"Plot and data has been save to {saved_path}")
        
        plt.close()

    

    def update_ui(outputs_frame):
        pass
