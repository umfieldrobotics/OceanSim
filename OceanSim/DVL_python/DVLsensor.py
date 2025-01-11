import numpy as np
from pxr import Gf
from omni.isaac.core.prims import BaseSensor
import omni.isaac.core.utils.rotations as rotations_utils
from omni.isaac.sensor import _sensor
import omni.kit.commands
from omni.isaac.dynamic_control import _dynamic_control
from ..utils.MultivariateNormal import MultivariateNormal
import carb
class DVLsensor:
    def __init__(self,
                 elevation:float = 22.5, # deg
                 vel_cov = np.array([1,1,1,1]),
                 depth_cov = np.array([1,1,1,1]),
                 min_range: float = 0,
                 max_range: float = 10,
                 ):
        self._elevation = elevation
        self._min_range = min_range
        self._max_range = max_range
        self._mvn_vel = MultivariateNormal(4)
        self._mvn_vel.init_cov(vel_cov)

        self._mvn_dep = MultivariateNormal(4)
        self._mvn_dep.init_cov(depth_cov)
        
        self._rigid_body_path = None
        self._dc = _dynamic_control.acquire_dynamic_control_interface()
        self._beam_paths = []

        sinElev = np.sin(np.deg2rad(self._elevation))
        cosElev = np.cos(np.deg2rad(self._elevation))
        self._transform = np.array([[1/(2*sinElev), 0, -1/(2*sinElev), 0],
                                    [0, 1/(2*sinElev), 0, -1/(2*sinElev)],
                                    [1/(4*cosElev), 1/(4*cosElev), 1/(4*cosElev), 1/(4*cosElev)]
                                    ])
        

    def attachDVL(self, rigid_body_path:str, location:np.ndarray = np.array([0.0, 0.0, 0.0])):
        self._rigid_body_path = rigid_body_path
        sensor_prim_path = rigid_body_path + "/DVL"
        self._DVL = BaseSensor(prim_path=sensor_prim_path,translation=location)
        
        elevation = np.deg2rad(self._elevation)
        orients_euler = np.array([[elevation, 0.0, 0.0], [0.0, elevation, 0.0], [-elevation, 0.0, 0.0], [0.0, -elevation, 0.0]])

        orients_quat = []
        for i in range(orients_euler.shape[0]):
            orients_quat.append(rotations_utils.euler_angles_to_quat(orients_euler[i,:]))
            self.beam_paths.append(sensor_prim_path + "/beam" + str(i))

            result, sensor = omni.kit.commands.execute(
                "IsaacSensorCreateLightBeamSensor",
                path=self.beam_paths[i],
                parent=None,
                min_range=self._min_range,
                max_range=self._max_range,
                translation=Gf.Vec3d(0, 0, 0),
                orientation=Gf.Quatd(*orients_quat[i]),
                forward_axis=Gf.Vec3d(0, 0, -1),
                num_rays=1,
                ) 
        if result:
            self._DVL_interface = _sensor.acquire_lightbeam_sensor_interface()
        else:
            carb.log_error("Beam Sensor fails to be loaded")
    

    def get_DVL_interface(self):
        return self._DVL_interface
    
    def get_baseSensor(self):
        return self._DVL
    
    def get_beam_paths(self):
        return self._beam_paths
    
    def get_depth(self):
        depth = []
        for beam_path in self._beam_paths:
            depth.append(self._DVL_interface.get_linear_depth_data(beam_path).squeeze())
        
        if (self._mvn_dep.is_uncertain()):
            for i in range(4):
                sample = self._mvn_dep.sample_array()
                depth[i] += sample[i]
        
        return depth
    
    def get_hit_pos(self):
        hit_pos = []
        for beam_path in self._beam_paths:
            hit_pos.append(self._DVL_interface.get_hit_pos_data(beam_path).squeeze())
        return hit_pos
    
    def get_beam_hit(self):
        beam_hit = []
        for beam_path in self._beam_paths:
            beam_hit.append(self._DVL_interface.get_beam_hit_data(beam_path).astype(bool).squeeze())
        return beam_hit
    
    def get_linear_vel(self):
        rob_body = self._dc.get_rigid_body(self._rigid_body_path)
        
        vel = self._dc.get_rigid_body_linear_velocity(rob_body)

        if (self._mvn_vel.is_uncertain()):
            sample = self._mvn_vel.sample_array()
            for i in range(4):
                for j in range(3):
                    vel[j] += self._transform[j][i] * sample[i] 