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
        self._camera = None

        self._dc = _dynamic_control.acquire_dynamic_control_interface()
        self._running_scenario = False
        self._time = 0.0




    def setup_scenario(self, rob, sonar):
        self._rob = rob
        self._sonar = sonar
        self._running_scenario = True




    def teardown_scenario(self):
        self._rob = None
        self._camera = None

        self._running_scenario = False
        self._time = 0.0



    def update_scenario(self, step: float):
        if not self._running_scenario:
            return
        self._time += step
        self._frame += 1

        if (self._frame < 50):
            XFormPrim('/rob').set_world_pose(position=[-1.5 + 0.2 * self._frame, -0.5, 6])
        sonar_data = self._sonar.make_sonar_data()
        print(sonar_data)


    # def make_sonar_data(self, pcl:np.ndarray, normals:np.ndarray, viewTransform:np.ndarray, ) -> np.ndarray:
        

    #     def cartesian_to_spherical(cart_coords):
    #         x, y, z = cart_coords[:, 0], cart_coords[:, 1], cart_coords[:, 2]
    #         r = np.sqrt(x**2 + y**2 + z**2)
    #         theta = np.arctan2(y, x)
    #         phi = np.arccos(z / r)
    #         return np.vstack((r, theta, phi)).T


    #     def bin_intensity(num_r_bins, num_azi_bins, pcl, intensity):
    #         min_r = pcl[:,0].min()
    #         max_r = pcl[:,0].max()
    #         min_azi = pcl[:,1].min()
    #         max_azi = pcl[:,1].max()
    #         r_bins = np.linspace(min_r, max_r, num_r_bins, endpoint=True)
    #         azi_bins = np.linspace(min_azi, max_azi, num_azi_bins, endpoint=True)

    #         intensity_binned, r_edges, azi_edges, _ = binned_statistic_2d(pcl[:,0], pcl[:,1], intensity, statistic='mean', bins=[r_bins, azi_bins])
    #         r_mid = (r_edges[:-1] + r_edges[1:]) / 2  
    #         azi_mid = (azi_edges[:-1] + azi_edges[1:]) / 2
    #         r, azi = np.meshgrid(r_mid, azi_mid, indexing='ij')
    #         return np.stack((r, azi, intensity_binned), axis=-1).reshape(-1,3)

        
    #     max_range = 1
    #     base_intensity = 255
    #     reflectivity = 1
    #     attenuation = 0.1

    #     normals = np.delete(arr=normals, obj=3, axis=1)
    #     viewTransform = viewTransform.reshape(4,4).T
    #     render_trans = -(viewTransform[:3,:3].T @ viewTransform[:3,3])
    #     dist = np.linalg.norm(pcl-render_trans, axis=1)
    #     directs = pcl - render_trans
    #     unit_directs = directs/np.linalg.norm(directs)

    #     theta = np.arccos(np.sum(unit_directs * normals, axis=1))
    #     # Formula to calculate the intensity 
    #     intensity = base_intensity * reflectivity * np.abs(np.cos(theta)) * (1/max_range)**2 * np.exp(-attenuation * 2 * dist)
    #     # Pre-multiplication to produce transform with respect to world frame
    #     pcl_local = (viewTransform @ np.hstack((pcl, np.ones([pcl.shape[0], 1]))).T).T 
    #     # Change the axis location to make z pointing upwards  and x pointing forwards for spherical coordinate transformation
    #     pcl_local = np.delete(pcl_local, obj=3, axis=1) @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

    #     pcl_spher_local = cartesian_to_spherical(pcl_local)
    #     sonar_data = bin_intensity(1024, 1024, pcl_spher_local, intensity)
    #     # Remove all the nan intensity entries
    #     sonar_data = sonar_data[~np.isnan(sonar_data[:,2])]
    #     # Normalized the intensity
    #     normalized_intensity = (sonar_data[:,2] - sonar_data[:,2].min()) / (sonar_data[:,2].max() - sonar_data[:,2].min())
    #     # Convert back to cartesian corrdiantes and map normalized intensity to 0-255 uint8
    #     sonar_data = np.array([sonar_data[:,0] * np.cos(sonar_data[:,1]), 
    #                         sonar_data[:,0] * np.sin(sonar_data[:,1]),
    #                         np.round(normalized_intensity * 255)
    #                         ]).T
    #     print(f'{sonar_data.shape[0]} sonar points generated')
    #     return sonar_data