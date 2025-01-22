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
from omni.isaac.ui.element_wrappers import XYPlot
from omni.isaac.range_sensor import _range_sensor
import matplotlib.pyplot as plt


class ImagingSonarScenario(ScenarioTemplate):
    def __init__(self):
        self._rob = None
        self._articulation = None

        self._dc = _dynamic_control.acquire_dynamic_control_interface()
        self._ul = _range_sensor.acquire_ultrasonic_sensor_interface()
        self._running_scenario = False
        self._time = 0.0



    def setup_scenario(self, rob, rotations, sonar_path):
        self._rob = rob
        self._rotations = rotations
        self._sonar_path = sonar_path
        self._running_scenario = True




    def teardown_scenario(self):
        self._rob = None
        self._articulation = None

        self._running_scenario = False
        self._time = 0.0


    def update_scenario(self, step: float):
        if not self._running_scenario:
            return

        self._time += step
        self.envelope_array = self._ul.get_envelope_array(self._sonar_path)
        self.bin_stamp = np.linspace(0, 224, 224)

        # depth = self._ul.get_linear_depth_data(self._sonar_path, 0)
        # azimuth = self._ul.get_azimuth_data(self._sonar_path)
        # zenith = self._ul.get_zenith_data(self._sonar_path)
        # num_points = azimuth.shape[0] * zenith.shape[0]
        # theta, phi = np.meshgrid(azimuth, zenith, indexing="ij")
        # self.pcl_spher = np.column_stack((depth.ravel(), theta.ravel(), phi.ravel()))
        # self.pcl_cart = np.zeros([num_points, 3])

        # intensity = self._ul.get_intensity_data(self._sonar_path, 0)
        # self._intensity = intensity.ravel()
        # self.sonar_map = np.zeros([num_points, 3])
        # map_width = 1
        # map_height = 1
        # hori_fov = np.abs(azimuth[-1] - azimuth[0])
        # max_range = 4.5
        # self.envelope_array = self._ul.get_envelope_array(self._sonar_path)
        
        # for i in range(num_points):
        #     # self.pcl_cart[i,:] = [self.pcl_spher[i,0] * np.cos(self.pcl_spher[i,1]) * np.sin(self.pcl_spher[i,2]),
        #     #                  self.pcl_spher[i,0] * np.sin(self.pcl_spher[i,1]) * np.sin(self.pcl_spher[i,2]),
        #     #                  self.pcl_spher[i,0] * np.cos(self.pcl_spher[i,2])]
        #     self.pcl_cart[i,:] = [self.pcl_spher[i,0] * np.cos(self.pcl_spher[i,2]) * np.cos(self.pcl_spher[i,1]),
        #                      self.pcl_spher[i,0] * np.cos(self.pcl_spher[i,2]) * np.sin(self.pcl_spher[i,1]),
        #                      self.pcl_spher[i,0] * np.sin(self.pcl_spher[i,2])]
        #     self.sonar_map[i,:] = [map_width/2 - (self.pcl_cart[i,1]/(np.sin(hori_fov) * max_range)) * np.sin(hori_fov/2) * map_height,
        #                       (self.pcl_cart[i,0]/max_range) * map_height,
        #                       self._intensity[i]/255]
            


    def save(self):
        saved_path = '/home/haoyu-ma/Desktop'

        np.save(saved_path+'/rotations.npy', self._rotations)
        np.save(saved_path+'/bin_stamp.npy', self.bin_stamp)
        np.save(saved_path+'/envelope.npy', self.envelope_array)

        print(f"Data has been save to {saved_path}")
        
        plt.close()

    

    def update_ui(outputs_frame):
        pass



# # Add dynamic cylinders as obstacles
#         obstacle_path = ["/obstacle_0", "/obstacle_1"]
#         self._obstacle = DynamicCylinder(
#             prim_path=obstacle_path[0],
#             translation=np.array([5,0,5]),
#             radius=0.5,
#             height=10,
#         )
#         obstacle_prim = prims_utils.get_prim_at_path(prim_path=obstacle_path[0])
#         obstacle_rigidBody_API = PhysxSchema.PhysxRigidBodyAPI.Apply(obstacle_prim)
#         obstacle_rigidBody_API.CreateDisableGravityAttr(True)
#         obstacle_rigidBody_API.GetLinearDampingAttr().Set(0.0)
#         obstacle_rigidBody_API.GetAngularDampingAttr().Set(0.0)

#         self._obstacle = DynamicCuboid(
#             prim_path=obstacle_path[1],
#             translation=np.array([5,2,2]),
#             size=1
#         )
#         obstacle_prim = prims_utils.get_prim_at_path(prim_path=obstacle_path[1])
#         obstacle_rigidBody_API = PhysxSchema.PhysxRigidBodyAPI.Apply(obstacle_prim)
#         obstacle_rigidBody_API.CreateDisableGravityAttr(True)
#         obstacle_rigidBody_API.GetLinearDampingAttr().Set(0.0)
#         obstacle_rigidBody_API.GetAngularDampingAttr().Set(0.0)