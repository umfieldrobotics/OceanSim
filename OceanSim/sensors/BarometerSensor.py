# Omniverse import
import numpy as np
from pxr import Gf
import omni.kit.commands
import omni.graph.core as og
import carb

# Isaac sim import
from isaacsim.core.api.sensors import BaseSensor
from isaacsim.core.prims import SingleRigidPrim

# Custom import
from ..utils.MultivariateNormal import MultivariateNormal

# TODO: @Haoyu - we should try to not let user define the g and water_surface_z.
# for g we should get it from scene definition
# for water_surface_z we should get it from the water surface you added
class BarometerSensor:
    def __init__(self,
                 water_density: float = 1000.0,     # kg/m^3 (default for water)
                 g: float = 9.81,                   # m/s^2, user-defined gravitational acceleration
                 noise_cov: float = 0.0,            # noise covariance for pressure measurement
                 water_surface_z: float = 0.0,      # z coordinate of the water surface
                 atmosphere_pressure: float = 101325.0  # atmospheric pressure in Pascals
                 ):
        """
        Initialize the barometer sensor.
        
        Args:
            water_density (float): Density of the water (kg/m^3).
            g (float): Gravitational acceleration (m/s^2).
            noise_cov (float): Covariance of the sensor noise.
            water_surface_z (float): The z-coordinate of the water surface.
            atmosphere_pressure (float): The atmospheric pressure (Pascals).
        """
        self._water_density = water_density
        self._g = g
        self._mvn_press = MultivariateNormal(1)
        self._mvn_press.init_cov(noise_cov)
        self._water_surface_z = water_surface_z
        self._atmosphere_pressure = atmosphere_pressure
        self._rigid_body_path = None
        self._sensor = None

    def attachBarometer(self, rigid_body_path: str, location: np.ndarray = np.array([0.0, 0.0, 0.0])):
        """
        Attach the barometer sensor to a rigid body.
        
        Args:
            rigid_body_path (str): The prim path of the rigid body.
            location (np.ndarray): Local translation offset of the sensor.
        """
        self._rigid_body_path = rigid_body_path
        self._rigid_body_prim = SingleRigidPrim(prim_path=rigid_body_path)
        sensor_prim_path = rigid_body_path + "/Barometer"
        self._sensor = BaseSensor(prim_path=sensor_prim_path, translation=location)

    def get_pressure(self) -> float:
        """
        Compute the current pressure measurement.
        
        The pressure is computed as:
            pressure = atmosphere_pressure + water_density * g * depth
        where depth is defined as the vertical distance below the water surface (if the sensor is below the surface).
        Noise is added based on the defined covariance.
        
        Returns:
            float: The pressure measurement (in Pascals).
        """
        # Get the current sensor (rigid body) pose.
        # We assume that get_local_pose() returns an object with a 'translation' attribute (e.g., a Gf.Vec3d).
        pose = self._rigid_body_prim.get_local_pose()
        sensor_pos = pose.translation  # sensor_pos is assumed to be a 3D vector (x, y, z)
        
        # Calculate depth based on the user-defined water surface z.
        # If the sensor is below the water surface (sensor z < water_surface_z), depth = water_surface_z - sensor_z.
        # Otherwise, if the sensor is above the water, depth is 0.
        if sensor_pos[2] < self._water_surface_z:
            depth = self._water_surface_z - sensor_pos[2]
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