import numpy as np
import random
# import omni.kit.raycast.query
from omni.physx import get_physx_scene_query_interface
from pxr import Gf
from omni.isaac.core.prims import BaseSensor
from omni.isaac.dynamic_control import _dynamic_control
import omni.isaac.core.utils.rotations as rotations_utils
from omni.isaac.debug_draw import _debug_draw
import carb

class SonarSensor():
    def __init__(self):
        hori_fov = 90 # deg
        vert_fov = 45 # deg
        hori_res = 3 # deg
        vert_res = 2 # deg
        self.max_range = 4.5 # unit multiplies with the ray unit vector 
        self.min_range = 0.5 # unit multiplies with the ray unit vector 
        self.base_intensity = 255

        self.local2world_rot = None
        self.local2world_tran = None
        
        
        
        self.draw = _debug_draw.acquire_debug_draw_interface()
        # self.raycast = omni.kit.raycast.query.acquire_raycast_query_interface()
        self.raycast = get_physx_scene_query_interface()


        self._hit_normal = []
        self._hit_distance = []
        self._hit_material_path = []
    
        
        azi = np.deg2rad(np.arange(-hori_fov/2, hori_fov/2, hori_res))
        zen = np.deg2rad(np.arange(90-vert_fov/2, 90+vert_fov/2, vert_res))
        self.numRays = azi.shape[0] * zen.shape[0]
        # self.azi, self.zen = np.meshgrid(azi, zen)
        self.origins_local = np.zeros([azi.shape[0], zen.shape[0], 3])
        self.unit_vec_local = np.zeros([azi.shape[0], zen.shape[0], 3])
        for i in range(azi.shape[0]):
            for j in range(zen.shape[0]):
                self.unit_vec_local[i,j] = np.array([np.sin(zen[j])*np.cos(azi[i]),
                                               np.sin(zen[j])*np.sin(azi[i]),
                                               np.cos(zen[j])])

        self.origins_local = (self.origins_local + self.unit_vec_local * self.min_range).reshape(-1,3)
        self.unit_vec_local = self.unit_vec_local.reshape(-1,3)


        self._rigid_body_path = None
        self._dc = _dynamic_control.acquire_dynamic_control_interface()
    
    def attachSonar(self, rigid_body_path: str, location:np.ndarray = np.array([0.0, 0.0, 0.0])):
        self._rigid_body_path = rigid_body_path
        sensor_prim_path = rigid_body_path + "/Sonar"
        self._sonar = BaseSensor(prim_path=sensor_prim_path,translation=location)
        

    def ray_cast(self):
        world_pose, world_orien_quat = self._sonar.get_world_pose()
        self.local2world_tran = world_pose
        self.local2world_rot = rotations_utils.quat_to_rot_matrix(world_orien_quat)
        self.origins_world = self.origins_local @ self.local2world_rot.T + self.local2world_tran
        self.unit_vec_world = self.unit_vec_local @ self.local2world_rot.T
    
        for i in range(self.numRays):
            origin = carb.Float3(*self.origins_world[i,:])
            direction = carb.Float3(*self.unit_vec_world[i,:])
            hit_info = self.raycast.raycast_closest(origin, direction, (self.max_range-self.min_range))

            if hit_info['hit']:
                self._hit_normal.append(hit_info['normal'])
                self._hit_distance.append(hit_info['distance'])
                self._hit_material_path.append(hit_info['material'])
        

    
    
    def draw_debug_lines(self):
        colors = [(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), 1) for _ in range(self.numRays)]
        sizes = [3 for _ in range(self.numRays)]
        self.draw.draw_lines(self.origins_world, 
                             self.origins_world + self.unit_vec_world * (self.max_range - self.min_range), 
                             colors, 
                             sizes)
        


    def clear_debug_lines(self):
        self.draw.clear_lines()







# def spher2cart(spher_coor: np.ndarray):
#     cart_coor = np.zeros(spher_coor.shape)
#     for i in range(spher_coor.shape[0]):
#         cart_coor[i,:] = [spher_coor[i,0] * np.cos(spher_coor[i,1]) * np.sin(spher_coor[i,2]),
#                         spher_coor[i,0] * np.sin(spher_coor[i,1]) * np.sin(spher_coor[i,2]),
#                         spher_coor[i,0] * np.cos(spher_coor[i,2])]
#     return cart_coor

# def apply_transformation(pts: np.ndarray, transformation: np.ndarray):
#     homogeneous_pts = np.hstack((pts, np.ones((pts.shape[0],1))))
#     transformed_homogeneous_pts = np.dot(homogeneous_pts, transformation.T)
#     return transformed_homogeneous_pts[:, :3]

# if __name__ == "__main__":
#     a = SonarSensor()
    
