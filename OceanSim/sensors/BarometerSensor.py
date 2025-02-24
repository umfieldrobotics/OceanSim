# Omniverse import
import numpy as np
from pxr import Gf
import omni.kit.commands
import omni.graph.core as og
import carb

# Isaac sim import
from isaacsim.core.api.sensors import BaseSensor
from isaacsim.core.api.physics_context import PhysicsContext

# Custom import
from ..utils.MultivariateNormal import MultivariateNormal

# TODO: Can not automatically resolve water surface height, need to write a separate class for water surface 
# (essentially a warp kernel connecting to a plane mesh)
class BarometerSensor(BaseSensor):
    def __init__(self, 
                 prim_path, 
                 name = "barometer", 
                 position = None, 
                 translation = None, 
                 orientation = None, 
                 scale = None, 
                 visible = None,
                 water_density: float = 1000.0,     # kg/m^3 (default for water)
                 g: float = 9.81,                   # m/s^2, user-defined gravitational acceleration
                 noise_cov: float = 0.0,            # noise covariance for pressure measurement
                 water_surface_z: float = 0.0,      # z coordinate of the water surface
                 atmosphere_pressure: float = 101325.0  # atmospheric pressure in Pascals
                 ) -> None:
        
        super().__init__(prim_path, name, position, translation, orientation, scale, visible)
        self._prim_path = prim_path
        self._water_density = water_density
        self._g = g
        self._mvn_press = MultivariateNormal(1)
        self._mvn_press.init_cov(noise_cov)
        self._water_surface_z = water_surface_z
        self._atmosphere_pressure = atmosphere_pressure  


    
        physics_context = PhysicsContext()
        g_dir, scene_g = physics_context.get_gravity()
        if np.abs(self._g - np.abs(scene_g)) > 0.1:
            carb.log_warn('Detected USD scene gravity is different from user definition. Reduced to user definition.')
        

    
    def get_pressure(self) -> float:

        if self.get_world_pose()[0][2] < self._water_surface_z:
            depth = self._water_surface_z - self.get_world_pose()[0][2]
        else:
            depth = 0.0
        
        # Compute hydrostatic pressure.
        pressure = self._atmosphere_pressure + self._water_density * self._g * depth
        
        # Add noise if defined.
        if self._mvn_press.is_uncertain():
            # The noise sample is a one-element array since our sensor is 1D.
            noise = self._mvn_press.sample_array()[0]
            pressure += noise
        
        return pressure