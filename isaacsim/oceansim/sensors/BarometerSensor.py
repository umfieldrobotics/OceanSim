# Omniverse import
import numpy as np
import carb

# Isaac sim import
from isaacsim.core.api.sensors import BaseSensor
from isaacsim.core.api.physics_context import PhysicsContext

# Custom import
from isaacsim.oceansim.utils.MultivariateNormal import MultivariateNormal


class BarometerSensor(BaseSensor):
    def __init__(self, 
                 prim_path, 
                 name = "baro", 
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
        
        """Initialize a barometer sensor with configurable physical properties and noise characteristics.

        .. note::

            This class is inheritied from ``BaseSensor``.

        Args:
            prim_path (str): prim path of the Prim to encapsulate or create.
            name (str, optional): shortname to be used as a key by Scene class.
                                    Note: needs to be unique if the object is added to the Scene.
                                    Defaults to "baro".
            position (Optional[Sequence[float]], optional): position in the world frame of the prim. shape is (3, ).
                                                        Defaults to None, which means left unchanged.
            translation (Optional[Sequence[float]], optional): translation in the local frame of the prim
                                                            (with respect to its parent prim). shape is (3, ).
                                                            Defaults to None, which means left unchanged.
            orientation (Optional[Sequence[float]], optional): quaternion orientation in the world/ local frame of the prim
                                                            (depends if translation or position is specified).
                                                            quaternion is scalar-first (w, x, y, z). shape is (4, ).
                                                            Defaults to None, which means left unchanged.
            scale (Optional[Sequence[float]], optional): local scale to be applied to the prim's dimensions. shape is (3, ).
                                                    Defaults to None, which means left unchanged.
            visible (bool, optional): set to false for an invisible prim in the stage while rendering. Defaults to True.
            water_density (float, optional): Fluid density in kg/m³. Defaults to 1000.0 (fresh water).
            g (float, optional): Gravitational acceleration in m/s². Defaults to 9.81.
            noise_cov (float, optional): Covariance for pressure measurement noise (0 = no noise). Defaults to 0.0.
            water_surface_z (float, optional): Z-coordinate of water surface in world frame. Defaults to 0.0.
            atmosphere_pressure (float, optional): Atmospheric pressure at surface in Pascals. Defaults to 101325.0 (1 atm).

        Raises:
            Exception: if translation and position defined at the same time
        """
        
        super().__init__(prim_path, name, position, translation, orientation, scale, visible)
        self._name = name
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
            carb.log_warn(f'[{self._name}] Detected USD scene gravity is different from user definition. Reduced to user definition.')
        

    
    def get_pressure(self) -> float:
        """Calculate the total pressure at the sensor's current position, including hydrostatic pressure and noise.

        Returns:
            float: Total pressure in Pascals (Pa), composed of:
                - Atmospheric pressure (constant)
                - Hydrostatic pressure (if submerged, calculated as ρgh)
                - Gaussian noise (if noise_cov > 0)
                
        Note:
            The sensor returns only atmospheric pressure when above water surface (z-position ≥ water_surface_z).
            When submerged (z-position < water_surface_z), hydrostatic pressure is added based on depth.
        """

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