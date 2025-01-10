import numpy as np
from pxr import Gf
from omni.isaac.core.prims import BaseSensor
import omni.isaac.core.utils.rotations as rotations_utils
from omni.isaac.sensor import _sensor
import omni.kit.commands


class DVLsensor:
    def __init__(self):
        self._elevation = 22.5 # deg
        self._vel_cov = 0
        self._vel_uncertainty = True
        self._min_range = 0.5
        self._max_range = 10
        self._beam_paths = []
        

    def attachDVL(self, rigid_body_path:str, location:np.ndarray = np.array([0.0, 0.0, 0.0])):
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
    
    def acquire_DVL_interface(self):
        return _sensor.acquire_lightbeam_sensor_interface()
    
    def get_beam_paths(self):
        return self._beam_paths
    
    