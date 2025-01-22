import numpy as np
import random
import omni.kit.raycast.query
from pxr import Gf
from omni.isaac.core.prims import BaseSensor
from omni.isaac.dynamic_control import _dynamic_control
import omni.isaac.core.utils.rotations as rotations_utils
from omni.isaac.debug_draw import _debug_draw
import carb

class SonarSensor():
    def __init__(self):
        self.hori_fov = 90 # deg
        self.vert_fov = 45 # deg
        self.hori_res = 0.9 # deg
        self.vert_res = 0.45 # deg
        self.max_range = 4.5 # unit multiplies with the ray unit vector 
        self.min_range = 0.5 # unit multiplies with the ray unit vector 
        
        self.base_intensity = 255
        self.attenuation = 0.25 # I = I₀ * e^(-μt)

        self.local2world_rot = None
        self.local2world_tran = None
        
        self.map_width = 1
        self.map_height = 1
        
        self.draw = _debug_draw.acquire_debug_draw_interface()
        self.raycast = omni.kit.raycast.query.acquire_raycast_query_interface()
        self._rigid_body_path = None
        self._dc = _dynamic_control.acquire_dynamic_control_interface()
    
        
        self.azi = np.deg2rad(np.arange(-self.hori_fov/2, self.hori_fov/2, self.hori_res))
        self.zen = np.deg2rad(np.arange(90-self.vert_fov/2, 90+self.vert_fov/2, self.vert_res))
        self.numRays = self.azi.shape[0] * self.zen.shape[0]
        self.origins_local = np.zeros([self.azi.shape[0], self.zen.shape[0], 3])
        self.unit_vec_local = np.zeros([self.azi.shape[0], self.zen.shape[0], 3])
        for i in range(self.azi.shape[0]):
            for j in range(self.zen.shape[0]):
                self.unit_vec_local[i,j] = np.array([np.sin(self.zen[j])*np.cos(self.azi[i]),
                                               np.sin(self.zen[j])*np.sin(self.azi[i]),
                                               np.cos(self.zen[j])])

        self.origins_local = self.origins_local.reshape(-1,3)
        self.unit_vec_local = self.unit_vec_local.reshape(-1,3)



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

        hit_position = []
        def query_callback(ray, result):
            print(result.hit_position)
            hit_position.append(result.hit_position)

        for i in range(self.numRays):
            ray = omni.kit.raycast.query.Ray(origin=self.origins_world,
                                       direction=self.unit_vec_world,
                                       min_t=self.min_range,
                                       max_t=self.max_range)
            self.raycast.submit_raycast_query(ray,query_callback)

           
        

    def get_azi(self):
        return self.azi
    
    def get_zen(self):
        return self.zen
    
    def draw_debug_lines(self):
        colors = [(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), 1) for _ in range(self.numRays)]
        sizes = [1.5 for _ in range(self.numRays)]
        self.draw.draw_lines(self.origins_world + self.min_range * self.unit_vec_world, 
                             self.origins_world + self.unit_vec_world * self.max_range, 
                             colors, 
                             sizes)

    def clear_debug_lines(self):
        self.draw.clear_lines()


    def clear_hit_pts(self):
        self.draw.clear_points()




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
    
